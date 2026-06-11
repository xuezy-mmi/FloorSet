#!/usr/bin/env python3
"""
dit_optimizer_hd.py - HouseDiffusion-style inference for the ICCAD 2026
floorplan optimizer.

This is the eval-side companion to `train_dit_hd.py`. It:

  1. Loads the trained `HouseDiffusionDiT` (preferring the EMA copy
     saved as `diffusion_ema_final.pt`, falling back to
     `diffusion_final.pt`).
  2. Loads the matching training meta (`meta.json`) which contains
     `norm_factor`, `n_steps`, `beta_start`, `beta_end`, etc.
  3. Runs DDPM (default) or DDIM reverse sampling — both reuse the
     helper functions in `dit_utils_hd.py`, which mirror the working
     `dit_utils.DiffusionScheduler` and HouseDiffusion's
     `gaussian_diffusion.{p_sample, ddim_sample}`.
  4. **Crucially**, after denormalization the raw positions are clipped
     to `[0, 1e6]` to guarantee non-negative x/y/w/h — this is the
     same trick the working original `dit_optimizer.py` uses to avoid
     negative positions.
  5. Enforces hard constraints (fixed-shape and preplaced) and
     post-processes the layout to remove overlaps and fix area
     tolerance.
  6. Falls back to the SA baseline in `optimizer_template.py` if no
     checkpoint is found.

Usage (from `iccad2026contest/`):

    python iccad2026_evaluate.py --evaluate dit_optimizer_hd.py

The model file is auto-discovered in this order:
    /home/xzy/eda/model/hd_v1/diffusion_ema_final.pt
    /home/xzy/eda/model/hd_v1/diffusion_final.pt
    ./diffusion_ema_final.pt
    ./diffusion_final.pt
"""
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))
from iccad2026_evaluate import FloorplanOptimizer
from dit_model_hd import HouseDiffusionDiT
from dit_utils_hd import (
    DiffusionScheduler,
    ddim_step,
    enforce_hard_constraints,
    fix_area_tolerance,
    p_sample_ddpm,
    remove_overlaps,
)


# ----------------------------------------------------------------------------
# Helper: list of candidate checkpoint directories
# ----------------------------------------------------------------------------
_CANDIDATE_DIRS = [
    Path("/home/xzy/eda/model/hd_v1"),
    Path("/home/xzy/eda/model/hd_default"),
    Path(__file__).parent / "checkpoints_hd",
    Path(__file__).parent,
]


def _find_ckpt() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Return (ema_ckpt, model_ckpt, meta_path) — whichever exist first."""
    for d in _CANDIDATE_DIRS:
        if not d.exists():
            continue
        ema = d / "diffusion_ema_final.pt"
        mdl = d / "diffusion_final.pt"
        meta = d / "meta.json"
        if ema.exists() or mdl.exists():
            return (
                ema if ema.exists() else None,
                mdl if mdl.exists() else None,
                meta if meta.exists() else None,
            )
    return None, None, None


# ----------------------------------------------------------------------------
# HouseDiffusionDiT optimizer
# ----------------------------------------------------------------------------
class MyOptimizer(FloorplanOptimizer):
    """HouseDiffusion-style diffusion optimizer."""

    def __init__(self, verbose: bool = False, sampler: str = "ddpm",
                 num_inference_steps: int = 1000, ddim_eta: float = 0.0):
        super().__init__(verbose)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampler = sampler
        self.num_inference_steps = num_inference_steps
        self.ddim_eta = ddim_eta

        self.model: Optional[HouseDiffusionDiT] = None
        self.schedule: Optional[DiffusionScheduler] = None
        self.norm_factor: float = 1000.0
        self.n_steps: int = 1000
        self._load()

    # ----- model loading --------------------------------------------------
    def _load(self):
        ema_ckpt, model_ckpt, meta_path = _find_ckpt()
        ckpt_path = ema_ckpt or model_ckpt
        if ckpt_path is None:
            if self.verbose:
                print("[HD] No checkpoint found, will fall back to SA baseline")
            return

        # detect arch / norm_factor from meta.json if present
        if meta_path is not None:
            import json
            meta = json.loads(meta_path.read_text())
            dim = int(meta.get("dim", 256))
            depth = int(meta.get("depth", 6))
            heads = int(meta.get("heads", 4))
            cond_in = int(meta.get("cond_in", 12))
            max_blocks = int(meta.get("max_blocks", 200))
            self.n_steps = int(meta.get("n_steps", 1000))
            self.norm_factor = float(meta.get("norm_factor", 1000.0))
            beta_start = float(meta.get("beta_start", 1e-4))
            beta_end = float(meta.get("beta_end", 0.02))
        else:
            dim, depth, heads, cond_in, max_blocks = 256, 6, 4, 12, 200
            self.n_steps = 1000
            self.norm_factor = 1000.0
            beta_start, beta_end = 1e-4, 0.02

        self.model = HouseDiffusionDiT(
            dim=dim, depth=depth, heads=heads, cond_in=cond_in,
            n_steps=self.n_steps, max_blocks=max_blocks,
        ).to(self.device)

        state = torch.load(ckpt_path, map_location=self.device)
        # The EMA checkpoint is a {name: tensor} dict of trainable
        # *parameters* only (no buffers). The plain model checkpoint IS
        # a state_dict and includes buffers like `pos_encoder.pe`.
        # We dispatch on whether the dict contains any non-parameter
        # key (anything that doesn't end in `weight` or `bias`); the
        # EMA shadow never does.
        keys = list(state.keys())
        has_buffer_like = any(
            not (k.endswith("weight") or k.endswith("bias")) for k in keys
        )
        if has_buffer_like:
            # model state_dict; load with strict=True
            self.model.load_state_dict(state)
        else:
            # EMA shadow
            sd = self.model.state_dict()
            for k, v in state.items():
                if k in sd and sd[k].shape == v.shape:
                    sd[k] = v.to(self.device)
            self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        if self.verbose:
            print(f"[HD] Loaded model from {ckpt_path}")

        # schedule
        self.schedule = DiffusionScheduler(
            n_steps=self.n_steps,
            beta_start=beta_start,
            beta_end=beta_end,
        ).to(self.device)
        if self.verbose:
            print(f"[HD] sampler={self.sampler} n_steps={self.n_steps} "
                  f"norm_factor={self.norm_factor} "
                  f"inference_steps={self.num_inference_steps}")

    # ----- solve ---------------------------------------------------------
    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Tuple[float, float, float, float]]:

        if self.model is None:
            return self._solve_sa_baseline(
                block_count, area_targets, b2b_connectivity,
                p2b_connectivity, pins_pos, constraints, target_positions,
            )

        try:
            return self._solve_diffusion(
                block_count, area_targets, b2b_connectivity,
                p2b_connectivity, pins_pos, constraints, target_positions,
            )
        except Exception as e:
            if self.verbose:
                import traceback
                print(f"[HD] Diffusion failed ({e!r}), falling back to SA")
                traceback.print_exc()
            return self._solve_sa_baseline(
                block_count, area_targets, b2b_connectivity,
                p2b_connectivity, pins_pos, constraints, target_positions,
            )

    # ----- diffusion inference -------------------------------------------
    def _solve_diffusion(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_conn: torch.Tensor,
        p2b_conn: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
    ) -> List[Tuple[float, float, float, float]]:

        # pad inputs to model.max_blocks so the model doesn't see a
        # 21-block tensor when it was trained on sequences of 200.
        max_blocks = self.model.max_blocks
        N = max(block_count, max_blocks)
        device = self.device

        area = self._pad_1d(area_targets, N, -1.0).unsqueeze(0).to(device)
        b2b = self._pad_edges(b2b_conn, N, dim_size=3).unsqueeze(0).to(device)
        p2b = self._pad_edges(p2b_conn, N, dim_size=3).unsqueeze(0).to(device)
        if pins_pos is None or pins_pos.numel() == 0:
            pins = torch.zeros(1, 0, 2, device=device)
        else:
            pins = pins_pos.unsqueeze(0).to(device)
        const = self._pad_2d(constraints, N, fill=0.0).unsqueeze(0).to(device)

        valid = (area != -1).float()  # [1, N]
        mask = valid.unsqueeze(-1)    # [1, N, 1]

        # start from noise
        x = torch.randn(1, N, 4, device=device)
        x = x * mask

        # decide inference steps (subsample for speed)
        n_full = self.n_steps
        n_inf = max(1, min(self.num_inference_steps, n_full))
        step_indices = torch.linspace(n_full - 1, 0, n_inf + 1).long().tolist()

        # strict x0 clip in normalized space: [0, 1] ensures all
        # positions are non-negative after un-normalization.
        with torch.no_grad():
            for k in range(len(step_indices) - 1):
                t_cur = step_indices[k]
                t_prev = step_indices[k + 1]
                t_tensor = torch.full(
                    (1,), t_cur, device=device, dtype=torch.long
                )
                pred_eps = self.model(
                    x, t_tensor, area, b2b, p2b, pins, const,
                )
                if self.sampler == "ddim":
                    x = ddim_step(
                        pred_eps, x, t_tensor,
                        torch.tensor([t_prev], device=device),
                        self.schedule, mask, eta=self.ddim_eta,
                        clip_lo=0.0, clip_hi=1.0,
                    )
                else:
                    x = p_sample_ddpm(
                        pred_eps, x, t_tensor,
                        torch.tensor([t_prev], device=device),
                        self.schedule, mask,
                        clip_lo=0.0, clip_hi=1.0,
                    )

        # de-normalize. Model output is in (w, h, x, y) order matching
        # `fp_sol` from training; the optimizer API expects (x, y, w, h).
        x = x * self.norm_factor       # [1, N, 4] in (w, h, x, y) order
        x = x.clamp(0.0, 1e6)          # safety: no negative positions
        x_raw = x[0, :, [2, 3, 0, 1]]  # [N, 4] = (x, y, w, h)
        x_raw = x_raw.cpu()
        # keep only valid blocks
        x_raw = x_raw[:block_count]

        # enforce hard constraints (fixed-shape / preplaced)
        if target_positions is not None:
            x_raw = enforce_hard_constraints(
                x_raw, target_positions[:block_count],
                constraints[:block_count],
            )

        # safety: w, h must be at least 1.0 to keep area meaningful
        w = x_raw[:, 2].clamp(min=1.0)
        h = x_raw[:, 3].clamp(min=1.0)
        x_raw = torch.stack([x_raw[:, 0], x_raw[:, 1], w, h], dim=-1)

        positions = [tuple(map(float, p)) for p in x_raw.tolist()]

        # post-process
        positions = remove_overlaps(positions, n_iters=50)
        positions = fix_area_tolerance(
            positions, area_targets[:block_count], constraints[:block_count],
            tolerance=0.01,
        )

        return positions

    # ----- padding helpers -----------------------------------------------
    @staticmethod
    def _pad_1d(x: torch.Tensor, n: int, fill: float) -> torch.Tensor:
        x = x.flatten()
        if x.shape[0] >= n:
            return x[:n]
        out = torch.full((n,), fill, dtype=x.dtype, device=x.device)
        out[: x.shape[0]] = x
        return out

    @staticmethod
    def _pad_2d(x: torch.Tensor, n: int, fill: float = 0.0) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.shape[0] >= n:
            return x[:n]
        out = torch.full((n, x.shape[1]), fill, dtype=x.dtype, device=x.device)
        out[: x.shape[0]] = x
        return out

    @staticmethod
    def _pad_edges(x: torch.Tensor, n: int, dim_size: int = 3) -> torch.Tensor:
        if x is None or x.numel() == 0:
            return torch.zeros(0, dim_size, dtype=torch.float32)
        if x.shape[0] == 0:
            return torch.zeros(0, dim_size, dtype=x.dtype)
        m = (x[..., 0] >= 0) & (x[..., 0] < n) & (x[..., 1] >= 0) & (x[..., 1] < n)
        if not m.all():
            x = x.clone()
            x[~m] = -1
        return x

    # ----- SA fallback (used only if no checkpoint) ----------------------
    def _solve_sa_baseline(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_conn: torch.Tensor,
        p2b_conn: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
    ) -> List[Tuple[float, float, float, float]]:

        widths, heights = [], []
        for i in range(block_count):
            if target_positions is not None and i < target_positions.shape[0] \
                    and target_positions[i, 2] != -1:
                w = float(target_positions[i, 2])
                h = float(target_positions[i, 3])
            else:
                area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
                w = h = math.sqrt(area)
            widths.append(w)
            heights.append(h)

        total_area = sum(w * h for w, h in zip(widths, heights))
        canvas_size = math.sqrt(total_area) * 1.5

        positions: List[Tuple[float, float, float, float]] = []
        for i in range(block_count):
            if target_positions is not None and i < target_positions.shape[0] \
                    and target_positions[i, 0] != -1:
                x = float(target_positions[i, 0])
                y = float(target_positions[i, 1])
            else:
                x = random.uniform(0, max(0, canvas_size - widths[i]))
                y = random.uniform(0, max(0, canvas_size - heights[i]))
            positions.append((x, y, widths[i], heights[i]))

        # greedy overlap push
        for _ in range(100):
            moved = False
            for i in range(block_count):
                for j in range(i + 1, block_count):
                    x1, y1, w1, h1 = positions[i]
                    x2, y2, w2, h2 = positions[j]
                    ox = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    oy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    if ox > 1e-6 and oy > 1e-6:
                        new_x = x2 + ox + 1
                        new_y = y2 + oy + 1
                        positions[j] = (new_x, new_y, w2, h2)
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                break

        # enforce hard constraints
        if target_positions is not None:
            pos_t = torch.tensor(positions, dtype=torch.float32)
            pos_t = enforce_hard_constraints(
                pos_t, target_positions[:block_count],
                constraints[:block_count],
            )
            positions = [tuple(map(float, p)) for p in pos_t.tolist()]

        # fix area tolerance
        positions = fix_area_tolerance(
            positions, area_targets[:block_count], constraints[:block_count],
            tolerance=0.01,
        )
        return positions
