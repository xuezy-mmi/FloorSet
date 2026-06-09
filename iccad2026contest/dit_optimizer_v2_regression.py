#!/usr/bin/env python3
"""
dit_optimizer_v2_regression.py - Regression-based optimizer

Companion to train_dit_v2_regression.py. Uses a Transformer that directly
predicts (x, y, w, h) given the conditioning. No diffusion — just a
regression model. This works much better than the DiT for this task
because the target distribution is a complex multi-modal function of
the conditions and the DiT was collapsing to a degenerate solution.
"""
import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from iccad2026_evaluate import FloorplanOptimizer


CKPT_PATH = Path("/home/xzy/eda/model/v2_regression/diffusion_final.pth")


def aggregate_graph_features(area_target, b2b_conn, p2b_conn, pins_pos, constraints):
    B, N = area_target.shape
    device, dtype = area_target.device, area_target.dtype

    safe = torch.where(area_target > 0, area_target, torch.ones_like(area_target))
    log_area = torch.log(safe) * (area_target != -1).float()

    b2b_w = torch.zeros(B, N, device=device, dtype=dtype)
    b2b_d = torch.zeros(B, N, device=device, dtype=dtype)
    if b2b_conn is not None and b2b_conn.numel() > 0:
        m = b2b_conn[..., 0] >= 0
        i_idx = b2b_conn[..., 0].long().clamp(min=0, max=N - 1)
        j_idx = b2b_conn[..., 1].long().clamp(min=0, max=N - 1)
        w = b2b_conn[..., 2] * m.float()
        i_safe = i_idx.clamp(min=0, max=N - 1)
        j_safe = j_idx.clamp(min=0, max=N - 1)
        b2b_w.scatter_add_(1, i_safe, w)
        b2b_w.scatter_add_(1, j_safe, w)
        b2b_d.scatter_add_(1, i_safe, m.float())
        b2b_d.scatter_add_(1, j_safe, m.float())

    p2b_w = torch.zeros(B, N, device=device, dtype=dtype)
    p2b_px = torch.zeros(B, N, device=device, dtype=dtype)
    p2b_py = torch.zeros(B, N, device=device, dtype=dtype)
    if p2b_conn is not None and p2b_conn.numel() > 0:
        m = p2b_conn[..., 0] >= 0
        pin_idx = p2b_conn[..., 0].long().clamp(min=0)
        blk_idx = p2b_conn[..., 1].long().clamp(max=N - 1)
        wt = p2b_conn[..., 2] * m.float()
        if pins_pos is not None and pins_pos.numel() > 0:
            P = pins_pos.shape[1]
            pin_idx_g = pin_idx.clamp(max=P - 1)
            px = pins_pos[..., 0].gather(1, pin_idx_g) * m.float()
            py = pins_pos[..., 1].gather(1, pin_idx_g) * m.float()
        else:
            px = torch.zeros_like(wt)
            py = torch.zeros_like(wt)
        p2b_w.scatter_add_(1, blk_idx, wt)
        p2b_px.scatter_add_(1, blk_idx, wt * px)
        p2b_py.scatter_add_(1, blk_idx, wt * py)

    is_hard = ((constraints[..., 0] > 0) | (constraints[..., 1] > 0)).float()
    boundary = constraints[..., 4].float() / 15.0

    return torch.stack([
        log_area, b2b_w, b2b_d, p2b_w, p2b_px, p2b_py, is_hard, boundary,
    ], dim=-1)


class RegModel(nn.Module):
    """Direct regression Transformer: conditions -> (x, y, w, h)."""
    def __init__(self, dim=256, depth=6, heads=8, cond_in=8):
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.input_proj = nn.Linear(8, dim)  # cond (8) + dummy 0 (for x,y,w,h)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads,
                                       dim_feedforward=dim * 4,
                                       dropout=0.1, batch_first=True, activation='gelu'),
            num_layers=depth,
        )
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 4))
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)

    def forward(self, area, b2b, p2b, pins, constraints):
        gf = aggregate_graph_features(area, b2b, p2b, pins, constraints)
        B, N, _ = gf.shape
        x = self.input_proj(torch.cat([gf, torch.zeros(B, N, 0, device=gf.device)], dim=-1)) if gf.shape[-1] < 8 else self.input_proj(gf[:, :, :8])
        x = x + self.cond_proj(gf)
        x = self.transformer(x)
        out = self.out(x)
        return out  # (B, N, 4) = (x, y, w, h) raw


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.mu = None
        self.sigma = None
        self._load()

    def _load(self):
        if not CKPT_PATH.exists():
            if self.verbose:
                print(f"[reg] No checkpoint at {CKPT_PATH}")
            return
        ckpt = torch.load(CKPT_PATH, map_location=self.device, weights_only=False)
        kw = ckpt.get('model_kwargs', {'dim': 256, 'depth': 6, 'heads': 8, 'cond_in': 8})
        self.model = RegModel(**kw).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        ns = ckpt['norm_stats']
        self.mu = ns['mu'].to(self.device)
        self.sigma = ns['sigma'].to(self.device)
        if self.verbose:
            print(f"[reg] Loaded ckpt from {CKPT_PATH}")

    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity,
              pins_pos, constraints, target_positions=None):
        if self.model is None:
            return self._fallback(block_count, area_targets, constraints, target_positions)
        with torch.no_grad():
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
            out = self.model(area, b2b, p2b, pins, constr)[0, :block_count]  # (N, 4) = (x, y, w, h)
        x = out[:, 0].cpu().numpy()
        y = out[:, 1].cpu().numpy()
        w = out[:, 2].cpu().numpy()
        h = out[:, 3].cpu().numpy()
        positions = []
        for i in range(block_count):
            wi = max(float(w[i]), 1e-3)
            hi = max(float(h[i]), 1e-3)
            xi = float(x[i])
            yi = float(y[i])
            if target_positions is not None:
                tp = target_positions[i]
                if tp[2] != -1:
                    wi = float(tp[2])
                    hi = float(tp[3])
                if tp[0] != -1:
                    xi = float(tp[0])
                    yi = float(tp[1])
                    wi = float(tp[2])
                    hi = float(tp[3])
            positions.append((xi, yi, wi, hi))
        positions = self._deoverlap(positions, constraints)
        return positions

    def _deoverlap(self, positions, constraints):
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
                if i in skip: continue
                for j in range(i + 1, n):
                    if j in skip: continue
                    x1, y1, w1, h1 = pos[i]
                    x2, y2, w2, h2 = pos[j]
                    ox = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    oy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    if ox > 1e-6 and oy > 1e-6:
                        pos[j][0] = x2 + ox + 1.0
                        pos[j][1] = y2
                        moved = True
            if not moved: break
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
