"""
dit_utils.py - Utilities for OLD v1 DiT model (matching train_dit.py checkpoint).

This matches the original v1 architecture that was trained:
  - dim=512, depth=8, heads=8, cond_dim=128
  - Standard DDPM with linear beta schedule
"""
import math

import torch
import torch.nn.functional as F


class DiffusionScheduler:
    """Linear-beta DDPM scheduler (matching original v1 training)."""
    def __init__(self, n_steps, beta_start=1e-4, beta_end=0.02):
        self.n_steps = n_steps
        self.beta = torch.linspace(beta_start, beta_end, n_steps)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)


def q_sample(x0, t, alpha_cumprod, noise=None):
    """Forward diffusion: add noise to x0 at step t."""
    if noise is None:
        noise = torch.randn_like(x0)
    if alpha_cumprod.device != x0.device:
        alpha_cumprod = alpha_cumprod.to(x0.device)
    alpha_cumprod_t = alpha_cumprod[t].view(-1, 1, 1)
    alpha_cumprod_t = torch.clamp(alpha_cumprod_t, min=0.0, max=1.0)
    sqrt_alpha = torch.sqrt(alpha_cumprod_t)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_cumprod_t)
    return sqrt_alpha * x0 + sqrt_one_minus * noise, noise


def q_sample_masked(x0: torch.Tensor, t: torch.Tensor,
                    alpha_cumprod: torch.Tensor,
                    mask: torch.Tensor,
                    noise: torch.Tensor = None):
    """Forward diffusion with valid-block mask [B, N, 1]."""
    if noise is None:
        noise = torch.randn_like(x0)
    if alpha_cumprod.device != x0.device:
        alpha_cumprod = alpha_cumprod.to(x0.device)
    a_t = alpha_cumprod[t].view(-1, 1, 1).clamp(min=1e-6)
    x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1.0 - a_t) * noise
    return x_t * mask, noise * mask


def aggregate_graph_features_v1(area_target: torch.Tensor,
                                 b2b_conn: torch.Tensor,
                                 p2b_conn: torch.Tensor,
                                 pins_pos: torch.Tensor,
                                 constraints: torch.Tensor) -> torch.Tensor:
    """
    v1 feature aggregation: log(area) per block -> [B, N, 1].
    This is what the OLD v1 model was trained with.
    """
    B, N = area_target.shape
    device, dtype = area_target.device, area_target.dtype

    # log(area) — single channel, matching original v1 cond_dim=128 input
    # The model will expand this to cond_dim=128 internally
    safe = torch.where(area_target > 0, area_target, torch.ones_like(area_target))
    log_area = torch.log(safe) * (area_target != -1).float()

    return log_area.unsqueeze(-1)  # [B, N, 1]


def vectorized_diff_loss(positions: torch.Tensor,        # [N, 4] = (x, y, w, h)
                          b2b_conn: torch.Tensor,
                          p2b_conn: torch.Tensor,
                          pins_pos: torch.Tensor,
                          area_targets: torch.Tensor,
                          baseline_metrics: torch.Tensor,
                          alpha: float = 0.5,
                          beta: float = 2.0) -> torch.Tensor:
    """
    Vectorized contest cost surrogate (differentiable).
    Cost = (1 + α·(HPWL_gap + Area_gap)) × exp(β·V_soft)
    """
    N = positions.shape[0]
    x, y, w, h = positions[:, 0], positions[:, 1], positions[:, 2], positions[:, 3]
    cx, cy = x + w / 2, y + h / 2
    device, dtype = positions.device, positions.dtype

    # HPWL b2b
    hpwl_b2b = torch.zeros((), device=device, dtype=dtype)
    if b2b_conn is not None and b2b_conn.numel() > 0:
        m = b2b_conn[:, 0] >= 0
        if m.any():
            e = b2b_conn[m]
            i, j = e[:, 0].long(), e[:, 1].long()
            v = (i < N) & (j < N)
            i, j, wt = i[v], j[v], e[v, 2]
            if i.numel():
                hpwl_b2b = (wt * (torch.abs(cx[i] - cx[j]) + torch.abs(cy[i] - cy[j]))).sum()

    # HPWL p2b
    hpwl_p2b = torch.zeros((), device=device, dtype=dtype)
    if p2b_conn is not None and p2b_conn.numel() > 0:
        m = p2b_conn[:, 0] >= 0
        if m.any():
            e = p2b_conn[m]
            pin_idx, blk_idx = e[:, 0].long(), e[:, 1].long()
            wt = e[:, 2]
            v = (pin_idx < pins_pos.shape[0]) & (blk_idx < N)
            pin_idx, blk_idx, wt = pin_idx[v], blk_idx[v], wt[v]
            if pin_idx.numel():
                px = pins_pos[pin_idx, 0]
                py = pins_pos[pin_idx, 1]
                hpwl_p2b = (wt * (torch.abs(cx[blk_idx] - px) + torch.abs(cy[blk_idx] - py))).sum()

    hpwl_total = hpwl_b2b + hpwl_p2b

    # BBox area
    if N > 0:
        bbox_area = (x.max() - x.min()).clamp(min=0) * (y.max() - y.min()).clamp(min=0)
    else:
        bbox_area = torch.zeros((), device=device, dtype=dtype)

    # Overlap (upper-triangle)
    if N > 1:
        xi, xj = x.unsqueeze(0), x.unsqueeze(1)
        yi, yj = y.unsqueeze(0), y.unsqueeze(1)
        wi, wj = w.unsqueeze(0), w.unsqueeze(1)
        hi, hj = h.unsqueeze(0), h.unsqueeze(1)
        ox = torch.relu(torch.min(xi + wi, xj + wj) - torch.max(xi, xj))
        oy = torch.relu(torch.min(yi + hi, yj + hj) - torch.max(yi, yj))
        tri = torch.triu(torch.ones(N, N, device=device, dtype=dtype), diagonal=1)
        overlap_area = ((ox * oy) * tri).sum()
    else:
        overlap_area = torch.zeros((), device=device, dtype=dtype)
    total_block_area = (w * h).sum().clamp(min=1e-6)
    overlap_v = overlap_area / total_block_area

    # Area tolerance (1% rule)
    valid_a = area_targets > 0
    if valid_a.any():
        err = torch.zeros_like(area_targets)
        err[valid_a] = torch.abs(w[valid_a] * h[valid_a] - area_targets[valid_a]) / area_targets[valid_a]
        area_v = torch.relu(err - 0.01).sum() / (valid_a.sum().float() + 1e-6)
    else:
        area_v = torch.zeros((), device=device, dtype=dtype)

    V_soft = torch.clamp(overlap_v + area_v, max=5.0)

    # Gaps
    baseline_area = baseline_metrics[0]
    baseline_hpwl = baseline_metrics[6] + baseline_metrics[7]
    hpwl_gap = torch.relu((hpwl_total - baseline_hpwl) / (baseline_hpwl + 1e-6))
    area_gap = torch.relu((bbox_area - baseline_area) / (baseline_area + 1e-6))

    return (1.0 + alpha * (hpwl_gap + area_gap)) * torch.exp(beta * V_soft)


def hard_violation_loss(pos: torch.Tensor,
                         target_pos: torch.Tensor,
                         constraints: torch.Tensor,
                         area_targets: torch.Tensor) -> torch.Tensor:
    """
    Penalty for hard constraint violations.
    = 0 when all hard constraints satisfied, > 0 when violated.
    """
    x, y, w, h = pos[:, 0], pos[:, 1], pos[:, 2], pos[:, 3]
    loss = torch.zeros((), device=pos.device, dtype=pos.dtype)

    # Fixed blocks: (w, h) must match target
    fixed = constraints[:, 0] > 0
    if fixed.any() and target_pos is not None:
        loss = loss + ((w[fixed] - target_pos[fixed, 2]) ** 2 +
                        (h[fixed] - target_pos[fixed, 3]) ** 2).mean()

    # Preplaced: (x, y, w, h) must all match target
    pp = constraints[:, 1] > 0
    if pp.any() and target_pos is not None:
        loss = loss + ((x[pp] - target_pos[pp, 0]) ** 2 +
                        (y[pp] - target_pos[pp, 1]) ** 2).mean()
        loss = loss + ((w[pp] - target_pos[pp, 2]) ** 2 +
                        (h[pp] - target_pos[pp, 3]) ** 2).mean()

    # Area tolerance on free blocks
    free = (constraints[:, 0] == 0) & (constraints[:, 1] == 0) & (area_targets > 0)
    if free.any():
        rel = (w[free] * h[free] - area_targets[free]).abs() / (area_targets[free] + 1e-6)
        loss = loss + torch.relu(rel - 0.01).mean() * 10.0

    return loss


def compute_norm_stats(dataloader, max_batches: int = 64):
    """Per-channel mean/std of (w, h, x, y) across the dataset."""
    ws, hs, xs, ys = [], [], [], []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        _, _, _, _, _, _, fp_sol, _ = batch
        fp = fp_sol.squeeze(0)
        valid = (fp != -1).all(dim=-1)
        fv = fp[valid]
        if fv.numel() == 0:
            continue
        ws.append(fv[:, 0]); hs.append(fv[:, 1])
        xs.append(fv[:, 2]); ys.append(fv[:, 3])
    w = torch.cat(ws).float(); h = torch.cat(hs).float()
    x = torch.cat(xs).float(); y = torch.cat(ys).float()
    mu = torch.stack([w.mean(), h.mean(), x.mean(), y.mean()])
    sigma = torch.stack([
        w.std().clamp(min=1.0),
        h.std().clamp(min=1.0),
        x.std().clamp(min=1.0),
        y.std().clamp(min=1.0),
    ])
    return mu, sigma