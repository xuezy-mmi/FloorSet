#!/usr/bin/env python3
"""
dit_optimizer_v2.py - DiT inference (v2)

Companion of train_dit_v2.py. Loads the v2 checkpoint and uses DDIM
sampling (50 steps) + z-score unnormalization + hard-constraint override
at inference. EMA weights are preferred if present.
"""
import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))
from iccad2026_evaluate import FloorplanOptimizer
from dit_model import DiffusionTransformer


CKPT_PATH = Path("/home/xzy/eda/model/v2/diffusion_final.pth")
N_DDIM_STEPS = 100


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.mu = None
        self.sigma = None
        self.alpha_cumprod = None
        self.n_train_steps = 1000
        self._load()

    def _load(self):
        if not CKPT_PATH.exists():
            if self.verbose:
                print(f"[v2] No checkpoint at {CKPT_PATH}")
            return
        ckpt = torch.load(CKPT_PATH, map_location=self.device, weights_only=False)
        kw = ckpt.get('model_kwargs', {'dim': 512, 'depth': 8, 'heads': 8, 'cond_dim': 128, 'n_steps': 1000})
        self.model = DiffusionTransformer(**kw).to(self.device)
        state = ckpt.get('ema_state_dict') or ckpt.get('model_state_dict')
        self.model.load_state_dict(state)
        self.model.eval()

        ns = ckpt['norm_stats']
        self.mu = ns['mu'].to(self.device)
        self.sigma = ns['sigma'].to(self.device)
        self.n_train_steps = ckpt.get('n_steps', 1000)

        from dit_utils import DiffusionScheduler
        sched = DiffusionScheduler(self.n_train_steps)
        self.alpha_cumprod = sched.alpha_cumprod.to(self.device)
        if self.verbose:
            print(f"[v2] Loaded ckpt from {CKPT_PATH}, mu={self.mu.tolist()}, sigma={self.sigma.tolist()}")

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _ddim_sample(self, area, b2b, p2b, pins, constr, n_steps: int) -> torch.Tensor:
        """Returns x0 in real scale, shape [1, N, 4] = (w, h, x, y)."""
        N = area.shape[1]
        x = torch.randn(1, N, 4, device=self.device)

        # Pick a subsequence of n_steps timesteps (linear schedule from T-1 -> 0)
        T = self.n_train_steps
        ts = torch.linspace(T - 1, 0, n_steps + 1).long()  # n_steps+1 entries
        for i in range(n_steps):
            t_cur = int(ts[i].item())
            t_prev = int(ts[i + 1].item()) if i + 1 < n_steps else -1
            t_tensor = torch.full((1,), t_cur, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t_tensor, area, b2b, p2b, pins, constr)

            a_cur = self.alpha_cumprod[t_cur]
            a_prev = self.alpha_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)
            a_cur = a_cur.clamp(min=1e-6)
            a_prev = a_prev.clamp(min=0.0)

            # predicted x0
            x0_hat = (x - torch.sqrt(1 - a_cur) * pred_noise) / torch.sqrt(a_cur)
            x0_hat = x0_hat.clamp(-5.0, 5.0)  # critical: keep predicted x0 within trained range
            # direction to x_t
            dir_xt = torch.sqrt((1 - a_prev).clamp(min=0.0)) * pred_noise
            x = torch.sqrt(a_prev) * x0_hat + dir_xt
            # mask invalid blocks back to 0 in z-score space
            valid = (area != -1)
            x = x * valid.unsqueeze(-1).float()

        # final x0 from last step
        x0_real = x * self.sigma + self.mu
        x0_real = x0_real.clamp(min=0.0)
        return x0_real

    # -----------------------------------------------------------------
    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity,
              pins_pos, constraints, target_positions=None):
        if self.model is None:
            return self._fallback(block_count, area_targets, constraints, target_positions)

        # Pad/unsqueeze to batch=1
        area = area_targets.unsqueeze(0).to(self.device)
        b2b = (b2b_connectivity.unsqueeze(0).to(self.device)
               if b2b_connectivity is not None and b2b_connectivity.numel()
               else torch.zeros(1, 0, 3, device=self.device))
        p2b = (p2b_connectivity.unsqueeze(0).to(self.device)
               if p2b_connectivity is not None and p2b_connectivity.numel()
               else torch.zeros(1, 0, 3, device=self.device))
        pins = (pins_pos.unsqueeze(0).to(self.device)
                if pins_pos is not None and pins_pos.numel()
                else torch.zeros(1, 0, 2, device=self.device))
        constr = constraints.unsqueeze(0).to(self.device)

        x0_real = self._ddim_sample(area, b2b, p2b, pins, constr, N_DDIM_STEPS)

        # Slice to actual block_count (DDIM used N = area.shape[1])
        x0_real = x0_real[0, :block_count]  # (block_count, 4) = (w, h, x, y)
        w = x0_real[:, 0].cpu().numpy()
        h = x0_real[:, 1].cpu().numpy()
        x = x0_real[:, 2].cpu().numpy()
        y = x0_real[:, 3].cpu().numpy()

        # Build positions, with hard-constraint override
        positions = []
        for i in range(block_count):
            wi = max(float(w[i]), 1e-3)
            hi = max(float(h[i]), 1e-3)
            xi = float(x[i])
            yi = float(y[i])
            if target_positions is not None:
                tp = target_positions[i]
                if tp[2] != -1:  # fixed-shape -> exact w, h
                    wi = float(tp[2])
                    hi = float(tp[3])
                if tp[0] != -1:  # preplaced -> exact (x, y, w, h)
                    xi = float(tp[0])
                    yi = float(tp[1])
                    wi = float(tp[2])
                    hi = float(tp[3])
            positions.append((xi, yi, wi, hi))

        # Light post-processing: de-overlap soft blocks via a simple sweep
        positions = self._deoverlap(positions, constraints)

        return positions

    # -----------------------------------------------------------------
    def _deoverlap(self, positions, constraints):
        """Sweep a few passes: if two soft blocks overlap, push the later one
        in +x. Hard constraint blocks (fixed/preplaced) are never moved."""
        n = len(positions)
        pos = [list(p) for p in positions]
        skip = set()
        if constraints is not None:
            for i in range(min(n, len(constraints))):
                if constraints[i, 0] != 0 or constraints[i, 1] != 0:
                    skip.add(i)
        for _ in range(20):
            moved = False
            for i in range(n):
                if i in skip:
                    continue
                for j in range(i + 1, n):
                    if j in skip:
                        continue
                    x1, y1, w1, h1 = pos[i]
                    x2, y2, w2, h2 = pos[j]
                    ox = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    oy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    if ox > 1e-6 and oy > 1e-6:
                        pos[j][0] = x2 + ox + 1.0
                        pos[j][1] = y2
                        moved = True
            if not moved:
                break
        return [tuple(p) for p in pos]

    def _fallback(self, block_count, area_targets, constraints, target_positions):
        positions = []
        for i in range(block_count):
            a = float(area_targets[i]) if area_targets[i] > 0 else 1.0
            wi = hi = math.sqrt(a)
            xi = yi = 0.0
            if target_positions is not None:
                if target_positions[i, 2] != -1:
                    wi = float(target_positions[i, 2])
                    hi = float(target_positions[i, 3])
                if target_positions[i, 0] != -1:
                    xi = float(target_positions[i, 0])
                    yi = float(target_positions[i, 1])
            positions.append((xi, yi, wi, hi))
        return positions
