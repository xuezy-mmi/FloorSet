"""
dit_utils_hd.py - HouseDiffusion-style utilities for the VLSI floorplan DiT.

Mirrors the working `dit_utils.py` (linear beta schedule) used by the
original DiT, and adds the masked forward/reverse diffusion helpers
originally ported from `house_diffusion.gaussian_diffusion`.

The single most important design choice: we use a positive, bounded
normalization (`x / norm_factor`) rather than z-score. The z-score
variant centred the data on the mean, which produced *negative* x/y
positions after denormalization (the user's main complaint). With a
plain positive divisor, the denormalized output is non-negative and we
can clip the predicted x0 to `[0, 1]` in normalized space to guarantee
that no negative w, h, x, y ever leak through the reverse process.
"""
import math
from typing import List, Optional, Tuple

import torch


# ----------------------------------------------------------------------------
# Linear beta schedule (matches the working `dit_utils.DiffusionScheduler`)
# ----------------------------------------------------------------------------
class DiffusionScheduler:
    """Linear-noise DDPM beta schedule (Ho et al., 2020)."""

    def __init__(self, n_steps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        self.n_steps = n_steps
        self.beta = torch.linspace(beta_start, beta_end, n_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        # alpha_cumprod_prev: shift by 1, with 1.0 at index 0
        self.alpha_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), self.alpha_cumprod[:-1]]
        )
        # For DDPM posterior: posterior_mean_coef1 / 2 etc.
        self.posterior_variance = (
            self.beta * (1.0 - self.alpha_cumprod_prev)
            / (1.0 - self.alpha_cumprod).clamp(min=1e-12)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([
                self.posterior_variance[1:2],
                self.posterior_variance[1:],
            ])
        )

    def to(self, device):
        for name in [
            "beta", "alpha", "alpha_cumprod", "alpha_cumprod_prev",
            "posterior_variance", "posterior_log_variance_clipped",
        ]:
            setattr(self, name, getattr(self, name).to(device))
        return self


# ----------------------------------------------------------------------------
# Forward diffusion (mask-aware)
# ----------------------------------------------------------------------------
def q_sample_masked(
    x0: torch.Tensor,
    t: torch.Tensor,
    alpha_cumprod: torch.Tensor,
    mask: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
):
    """Forward diffusion with valid-block mask [B, N, 1].

    Returns (x_t * mask, noise * mask) so the model only sees real
    signals on valid positions and the loss is well-defined.
    """
    if noise is None:
        noise = torch.randn_like(x0)
    if alpha_cumprod.device != t.device:
        alpha_cumprod = alpha_cumprod.to(t.device)
    a_t = alpha_cumprod[t].view(-1, 1, 1).clamp(min=0.0, max=1.0)
    x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1.0 - a_t) * noise
    return x_t * mask, noise * mask


# ----------------------------------------------------------------------------
# Reverse diffusion: DDPM single step (with strict x0 clip to [0, 1])
# ----------------------------------------------------------------------------
@torch.no_grad()
def p_sample_ddpm(
    model_out_eps: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    t_prev: torch.Tensor,
    schedule: DiffusionScheduler,
    mask: torch.Tensor,
    clip_x0: bool = True,
    clip_lo: float = 0.0,
    clip_hi: float = 1.0,
) -> torch.Tensor:
    """One DDPM reverse step. Mirrors `GaussianDiffusion.p_sample`.

    `t_prev` is the *previous* timestep index (t-1, or 0 at t=0).
    `t_prev` may be -1 to mean "no noise at the last step".
    `clip_lo` / `clip_hi` bound the predicted x0 in *normalized* space
    (default [0, 1] — this is the key fix that keeps all positions
    non-negative after un-normalization).
    """
    a_t = schedule.alpha_cumprod[t].view(-1, 1, 1)
    a_prev = (
        schedule.alpha_cumprod_prev[t_prev.clamp(min=0)].view(-1, 1, 1)
        if (t_prev >= 0).all() else
        torch.ones_like(a_t)
    )
    beta_t = (1.0 - a_t / a_prev).clamp(min=1e-8, max=0.999)

    # predicted x0
    x0_pred = (x_t - torch.sqrt(1.0 - a_t) * model_out_eps) / torch.sqrt(a_t)
    if clip_x0:
        x0_pred = x0_pred.clamp(clip_lo, clip_hi)
    x0_pred = x0_pred * mask

    # posterior mean (DDPM)
    mean = (
        torch.sqrt(a_prev) * beta_t / (1.0 - a_t).clamp(min=1e-8)
    ) * x0_pred + (
        (1.0 - a_prev) * torch.sqrt(1.0 - beta_t) / (1.0 - a_t).clamp(min=1e-8)
    ) * x_t

    # add noise unless t == 0
    nonzero = (t != 0).float().view(-1, 1, 1)
    log_var = schedule.posterior_log_variance_clipped[t].view(-1, 1, 1)
    noise = nonzero * torch.exp(0.5 * log_var) * torch.randn_like(x_t)
    x_prev = (mean + noise) * mask
    return x_prev


# ----------------------------------------------------------------------------
# Reverse diffusion: DDIM single step (faster, deterministic if eta=0)
# ----------------------------------------------------------------------------
@torch.no_grad()
def ddim_step(
    model_out_eps: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    t_prev: torch.Tensor,
    schedule: DiffusionScheduler,
    mask: torch.Tensor,
    eta: float = 0.0,
    clip_x0: bool = True,
    clip_lo: float = 0.0,
    clip_hi: float = 1.0,
) -> torch.Tensor:
    """One DDIM reverse step. Mirrors `GaussianDiffusion.ddim_sample`."""
    a_t = schedule.alpha_cumprod[t].view(-1, 1, 1)
    a_prev = torch.where(
        t_prev >= 0,
        schedule.alpha_cumprod_prev[t_prev.clamp(min=0)].view(-1, 1, 1),
        torch.ones_like(a_t),
    )

    x0_pred = (x_t - torch.sqrt(1.0 - a_t) * model_out_eps) / torch.sqrt(a_t)
    if clip_x0:
        x0_pred = x0_pred.clamp(clip_lo, clip_hi)
    x0_pred = x0_pred * mask

    sigma = eta * torch.sqrt(
        (1.0 - a_prev) / (1.0 - a_t).clamp(min=1e-8) *
        (1.0 - a_t / a_prev.clamp(min=1e-8))
    )
    dir_xt = torch.sqrt((1.0 - a_prev - sigma ** 2).clamp(min=0)) * model_out_eps
    noise = sigma * torch.randn_like(x_t) if eta > 0 else 0.0
    x_prev = torch.sqrt(a_prev) * x0_pred + dir_xt + noise
    return x_prev * mask


# ----------------------------------------------------------------------------
# Masked MSE loss (HouseDiffusion `mean_flat` with padding mask)
# ----------------------------------------------------------------------------
def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE averaged over valid (mask==1) entries only.

    `mask` is broadcast-compatible with `pred`/`target`. The denominator
    is `mask.sum()` (or 1.0 if mask is empty), giving a per-element MSE
    that is well-defined for variable-length sequences.
    """
    diff2 = (pred - target) ** 2
    diff2 = diff2 * mask
    n = mask.sum().clamp(min=1.0)
    return diff2.sum() / n


# ----------------------------------------------------------------------------
# EMA (Exponential Moving Average) of trainable parameters
# ----------------------------------------------------------------------------
class EMAModel:
    """Maintain EMA copies of a model's trainable parameters.

    Mirrors `house_diffusion.nn.update_ema` and the `ema_params` logic
    in `house_diffusion.train_util.TrainLoop`.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(
                param.detach(), alpha=1.0 - self.decay
            )

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].data)

    def state_dict(self) -> dict:
        return {k: v.cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict):
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device))


# ----------------------------------------------------------------------------
# Post-processing: hard constraints & overlap removal
# ----------------------------------------------------------------------------
def enforce_hard_constraints(
    positions: torch.Tensor,           # [N, 4] in (x, y, w, h) raw space
    target_positions: torch.Tensor,   # [N, 4] (-1 for free)
    constraints: torch.Tensor,         # [N, 5]
) -> torch.Tensor:
    """Overwrite fixed/preplaced entries in `positions` to match targets.

    Hard constraints (per PDF):
      - fixed-shape:   w, h must equal target (x, y free)
      - preplaced:     x, y, w, h must all equal target
    """
    if target_positions is None:
        return positions
    out = positions.clone()
    if constraints is None:
        return out
    ncols = constraints.shape[1] if constraints.dim() > 1 else 0
    for i in range(out.shape[0]):
        is_fixed = ncols > 0 and constraints[i, 0] != 0
        is_preplaced = ncols > 1 and constraints[i, 1] != 0
        if is_fixed or is_preplaced:
            tx, ty, tw, th = target_positions[i].tolist()
            if is_fixed and tw > 0:
                out[i, 2] = tw
                out[i, 3] = th
            if is_preplaced and tx >= 0:
                out[i, 0] = tx
                out[i, 1] = ty
                out[i, 2] = tw
                out[i, 3] = th
    return out


def remove_overlaps(
    positions: List[Tuple[float, float, float, float]],
    locked: Optional[List[bool]] = None,
    n_iters: int = 50,
) -> List[Tuple[float, float, float, float]]:
    """Greedy pair-wise push-apart. Padded positions are untouched."""
    pos = [list(p) for p in positions]
    n = len(pos)
    if locked is None:
        locked = [False] * n
    for _ in range(n_iters):
        moved = False
        for i in range(n):
            if locked[i]:
                continue
            for j in range(i + 1, n):
                if locked[j]:
                    continue
                x1, y1, w1, h1 = pos[i]
                x2, y2, w2, h2 = pos[j]
                ox = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
                oy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
                if ox > 1e-6 and oy > 1e-6:
                    # push the larger block to the right/down a bit
                    if w1 * h1 >= w2 * h2:
                        pos[j][0] = x2 + ox + 1.0
                    else:
                        pos[i][0] = x1 + ox + 1.0
                    moved = True
            if moved:
                break
        if not moved:
            break
    return [tuple(p) for p in pos]


def fix_area_tolerance(
    positions: List[Tuple[float, float, float, float]],
    area_targets: torch.Tensor,
    constraints: torch.Tensor,
    tolerance: float = 0.01,
) -> List[Tuple[float, float, float, float]]:
    """Scale w,h so that |w*h - target| / target <= tolerance.

    Skips fixed-shape and preplaced blocks (which have their own hard
    constraint enforced elsewhere).
    """
    pos = [list(p) for p in positions]
    n = len(pos)
    for i in range(n):
        if constraints is not None and i < len(constraints):
            if constraints[i, 0] != 0 or constraints[i, 1] != 0:
                continue
        if i >= len(area_targets) or area_targets[i] <= 0:
            continue
        target = float(area_targets[i])
        w, h = pos[i][2], pos[i][3]
        actual = w * h
        if abs(actual - target) / target > tolerance:
            scale = math.sqrt(target / max(actual, 1e-6))
            pos[i][2] = w * scale
            pos[i][3] = h * scale
    return [tuple(p) for p in pos]
