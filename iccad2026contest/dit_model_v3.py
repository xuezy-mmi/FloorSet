#!/usr/bin/env python3
"""
dit_model_v3.py - DiT with proper graph conditioning (v3)

Differs from dit_model.py by:
  - Permutation-equivariant (no pos_embed that breaks optimizer state).
  - Conditioning aggregates b2b weights, p2b weighted pin positions, and
    placement constraints into a per-block feature vector before projection.
  - Self-attention over blocks (so the model can model inter-block HPWL).
"""
import math

import torch
import torch.nn as nn


def aggregate_graph_features(area_target, b2b_conn, p2b_conn, pins_pos, constraints):
    """Build per-block conditioning features [B, N, F].

    Features per block (8 channels):
        0  log(area)   (log of target area; 0 if invalid)
        1  b2b weight sum  (sum of b2b edge weights incident to this block)
        2  b2b degree      (number of b2b edges incident to this block)
        3  p2b weight sum  (sum of p2b edge weights)
        4  p2b weighted pin x
        5  p2b weighted pin y
        6  is_fixed_or_preplaced (0/1)
        7  boundary code (0/15, then /15)
    """
    B, N = area_target.shape
    device = area_target.device
    dtype = area_target.dtype

    # log area (treat -1 / 0 as 0 contribution)
    safe = torch.where(area_target > 0, area_target, torch.ones_like(area_target))
    log_area = torch.log(safe)
    valid = (area_target != -1).float()
    log_area = log_area * valid

    # b2b aggregation: scatter_add weights and counts to both endpoints
    b2b_w = torch.zeros(B, N, device=device, dtype=dtype)
    b2b_d = torch.zeros(B, N, device=device, dtype=dtype)
    if b2b_conn is not None and b2b_conn.numel() > 0:
        edge_mask = b2b_conn[..., 0] >= 0   # [B, E]
        i_idx = b2b_conn[..., 0].long().clamp(min=0)
        j_idx = b2b_conn[..., 1].long().clamp(min=0)
        w = b2b_conn[..., 2] * edge_mask.float()
        # clamp block indices to N-1 for safe scatter
        i_idx = i_idx.clamp(max=N - 1)
        j_idx = j_idx.clamp(max=N - 1)
        b2b_w.scatter_add_(1, i_idx, w)
        b2b_w.scatter_add_(1, j_idx, w)
        b2b_d.scatter_add_(1, i_idx, edge_mask.float())
        b2b_d.scatter_add_(1, j_idx, edge_mask.float())

    # p2b aggregation
    p2b_w = torch.zeros(B, N, device=device, dtype=dtype)
    p2b_px = torch.zeros(B, N, device=device, dtype=dtype)
    p2b_py = torch.zeros(B, N, device=device, dtype=dtype)
    if p2b_conn is not None and p2b_conn.numel() > 0:
        edge_mask = p2b_conn[..., 0] >= 0
        pin_idx = p2b_conn[..., 0].long().clamp(min=0)
        blk_idx = p2b_conn[..., 1].long().clamp(min=0, max=N - 1)
        wt = p2b_conn[..., 2] * edge_mask.float()
        # gather pin positions
        if pins_pos is not None and pins_pos.numel() > 0:
            P = pins_pos.shape[1]
            pin_idx_g = pin_idx.clamp(max=P - 1)
            px = pins_pos[..., 0].gather(1, pin_idx_g) * edge_mask.float()
            py = pins_pos[..., 1].gather(1, pin_idx_g) * edge_mask.float()
        else:
            px = torch.zeros_like(wt)
            py = torch.zeros_like(wt)
        p2b_w.scatter_add_(1, blk_idx, wt)
        p2b_px.scatter_add_(1, blk_idx, wt * px)
        p2b_py.scatter_add_(1, blk_idx, wt * py)

    # constraint features
    is_hard = (constraints[..., 0] > 0) | (constraints[..., 1] > 0)
    boundary = constraints[..., 4].float() / 15.0

    feats = torch.stack([
        log_area,
        b2b_w,
        b2b_d,
        p2b_w,
        p2b_px,
        p2b_py,
        is_hard.float(),
        boundary,
    ], dim=-1)  # [B, N, 8]
    return feats


class DiffusionTransformer(nn.Module):
    """v3: permutation-equivariant DiT with graph conditioning."""
    def __init__(self, dim: int = 256, depth: int = 6, heads: int = 8,
                 cond_in: int = 8, n_steps: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.cond_in = cond_in
        self.n_steps = n_steps

        # input: 4 (layout) + cond_in (graph features)
        self.input_proj = nn.Linear(4 + cond_in, dim)

        # cond projection on graph features alone (residual)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)

    def _time_embedding(self, t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
        args = t.unsqueeze(-1).float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, x, t, area_target, b2b_conn, p2b_conn, pins_pos, constraints):
        """
        x: [B, N, 4] noisy layout
        t: [B] timesteps
        area_target, b2b_conn, p2b_conn, pins_pos, constraints: conditioning
        """
        B, N, _ = x.shape

        # build per-block graph features
        gf = aggregate_graph_features(area_target, b2b_conn, p2b_conn, pins_pos, constraints)
        # [B, N, cond_in]

        # mask invalid blocks (area_target == -1)
        valid = (area_target != -1).float().unsqueeze(-1)  # [B, N, 1]
        x = x * valid

        tokens = self.input_proj(torch.cat([x, gf], dim=-1))         # [B, N, dim]
        tokens = tokens + self.cond_proj(gf)                          # residual
        t_emb = self._time_embedding(t, self.dim)                    # [B, dim]
        tokens = tokens + t_emb.unsqueeze(1)                         # broadcast to [B, N, dim]

        h = self.transformer(tokens)                                 # [B, N, dim]
        pred_noise = self.output_proj(h)                             # [B, N, 4]
        pred_noise = pred_noise * valid                              # mask output
        return pred_noise
