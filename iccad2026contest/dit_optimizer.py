#!/usr/bin/env python3
"""
dit_optimizer.py - DiT inference with multi-sample selection.

Uses v3 checkpoint at: /home/xzy/eda/model/v3/diffusion_final.pth
  - dim=256, depth=6, heads=8, cond_in=8
  - Pre-LN Transformer + edge-bias attention
  - Complete with norm_stats for proper denormalization
  - CosineSchedule for DDPM

Multi-sample DDIM: runs N_SAMPLES DDIM with different seeds and picks
the one with lowest overlap ratio.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from iccad2026_evaluate import FloorplanOptimizer, check_overlap
from dit_model_v3 import DiffusionTransformer
from dit_utils_v3 import CosineSchedule

CKPT_PATH = Path("/home/xzy/eda/model/v3/diffusion_final.pth")
N_DDIM_STEPS = 50
N_SAMPLES = 4  # number of DDIM samples to try per case


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
                print(f"[dit] No checkpoint at {CKPT_PATH}")
            return
        ckpt = torch.load(CKPT_PATH, map_location=self.device, weights_only=False)

        kw = ckpt.get('model_kwargs', {'dim': 256, 'depth': 6, 'heads': 8, 'cond_in': 8, 'n_steps': 1000})
        self.model = DiffusionTransformer(**kw).to(self.device)

        # Try EMA state first, fall back to model state
        state = ckpt.get('ema_state_dict') or ckpt.get('model_state_dict')
        self.model.load_state_dict(state)
        self.model.eval()

        ns = ckpt['norm_stats']
        self.mu = ns['mu'].to(self.device)
        self.sigma = ns['sigma'].to(self.device)
        self.n_train_steps = ckpt.get('n_steps', 1000)

        sched = CosineSchedule(self.n_train_steps)
        self.alpha_cumprod = sched.alpha_cumprod.to(self.device)

        if self.verbose:
            print(f"[dit] Loaded v3 checkpoint from {CKPT_PATH}")
            print(f"  mu    = {self.mu.tolist()}")
            print(f"  sigma = {self.sigma.tolist()}")

    @torch.no_grad()
    def _ddim_sample(self, area, b2b, p2b, pins, constr, n_steps: int) -> torch.Tensor:
        """DDIM sampling with cosine schedule."""
        N = area.shape[1]
        x = torch.randn(1, N, 4, device=self.device)
        T = self.n_train_steps
        ts = torch.linspace(T - 1, 0, n_steps + 1).long()

        for i in range(n_steps):
            t_cur = int(ts[i].item())
            t_prev = int(ts[i + 1].item()) if i + 1 < n_steps else -1
            t_tensor = torch.full((1,), t_cur, device=self.device, dtype=torch.long)

            pred_noise = self.model(x, t_tensor, area, b2b, p2b, pins, constr)

            a_cur = self.alpha_cumprod[t_cur].clamp(min=1e-6)
            a_prev = (self.alpha_cumprod[t_prev].clamp(min=0.0)
                      if t_prev >= 0 else torch.tensor(1.0, device=self.device))

            x0_hat = (x - torch.sqrt(1.0 - a_cur) * pred_noise) / torch.sqrt(a_cur)
            x0_hat = x0_hat.clamp(-5.0, 5.0)
            dir_xt = torch.sqrt((1.0 - a_prev).clamp(min=0.0)) * pred_noise
            x = torch.sqrt(a_prev) * x0_hat + dir_xt

            valid = (area != -1)
            x = x * valid.unsqueeze(-1).float()

        # Denormalize
        x0_real = x * self.sigma + self.mu
        x0_real = x0_real.clamp(min=0.0)
        # Zero out invalid blocks (they were masked to 0 during sampling,
        # so they would otherwise get mu after denormalization)
        valid = (area != -1)
        x0_real = x0_real * valid.unsqueeze(-1).float()
        return x0_real

    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity,
              pins_pos, constraints, target_positions=None):
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

        # Multi-sample DDIM: run N_SAMPLES and pick lowest-overlap
        best_positions = None
        best_overlaps = float('inf')

        for seed in range(N_SAMPLES):
            torch.manual_seed(seed)
            x0_real = self._ddim_sample(area, b2b, p2b, pins, constr, N_DDIM_STEPS)
            x0_real = x0_real[0, :block_count]

            positions = []
            for i in range(block_count):
                w = float(x0_real[i, 0].cpu().item())
                h = float(x0_real[i, 1].cpu().item())
                x = float(x0_real[i, 2].cpu().item())
                y = float(x0_real[i, 3].cpu().item())

                # Hard constraint override
                if target_positions is not None and constraints is not None:
                    nc = constraints.shape[1] if constraints.dim() > 1 else 0
                    is_fixed = nc > 0 and float(constraints[i, 0]) != 0
                    is_preplaced = nc > 1 and float(constraints[i, 1]) != 0

                    if is_preplaced:
                        x = float(target_positions[i, 0])
                        y = float(target_positions[i, 1])
                        w = float(target_positions[i, 2])
                        h = float(target_positions[i, 3])
                    elif is_fixed:
                        w = float(target_positions[i, 2])
                        h = float(target_positions[i, 3])

                if w <= 0:
                    w = 1.0
                if h <= 0:
                    h = 1.0

                positions.append((x, y, w, h))

            overlaps = check_overlap(positions)
            if overlaps < best_overlaps:
                best_overlaps = overlaps
                best_positions = positions

            if best_overlaps == 0:
                break

        return best_positions if best_positions else positions