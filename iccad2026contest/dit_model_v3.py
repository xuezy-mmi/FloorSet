"""
dit_model_v3.py - v3 DiT for floorplan optimization.

Improvements over previous versions:
  - Pre-LN TransformerEncoder (more stable training).
  - LayerNorm in input/output projections.
  - Edge-bias in self-attention: b2b weights become additive attention bias.
  - Per-block embeddings carry an "is_locked" indicator so the model can
    differentiate hard-constrained vs. free blocks.
"""
import math

import torch
import torch.nn as nn

from dit_utils_v3 import aggregate_graph_features


class DiffusionTransformer(nn.Module):
    """v3: pre-LN DiT with edge-bias self-attention."""
    def __init__(self, dim: int = 256, depth: int = 6, heads: int = 8,
                 cond_in: int = 8, n_steps: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.cond_in = cond_in
        self.n_steps = n_steps

        # input: 4 (layout in z-score) + cond_in (graph features)
        self.input_proj = nn.Sequential(
            nn.Linear(4 + cond_in, dim),
            nn.LayerNorm(dim),
        )
        # residual cond projection
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in, dim), nn.SiLU(), nn.Linear(dim, dim),
        )
        # time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )

        # pre-LN transformer (more stable for deep nets)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, activation='gelu',
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, 4),
        )

        # Edge bias projection: scalar b2b weight per ordered pair -> single bias
        # Applied to attention logits as (B, N, N), broadcast across heads.
        self.edge_bias = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.edge_bias.weight)

        for p in self.parameters():
            if p.dim() > 1 and p is not self.edge_bias.weight:
                nn.init.xavier_uniform_(p, gain=0.02)

    def _time_embedding(self, t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
        args = t.unsqueeze(-1).float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def _attn_with_edge_bias(self, tokens, b2b_conn):
        """Custom transformer forward that injects b2b edge bias into attention."""
        B, N, D = tokens.shape
        # Build [B, N, N] scalar edge weight matrix from b2b_conn [B, E, 3]
        edge_w = torch.zeros(B, N, N, device=tokens.device, dtype=tokens.dtype)
        if b2b_conn is not None and b2b_conn.numel() > 0:
            m = b2b_conn[..., 0] >= 0
            i = b2b_conn[..., 0].long().clamp(min=0, max=N - 1)
            j = b2b_conn[..., 1].long().clamp(min=0, max=N - 1)
            w = (b2b_conn[..., 2] * m.float())
            edge_w[torch.arange(B).unsqueeze(1), i, j] = w
            edge_w[torch.arange(B).unsqueeze(1), j, i] = w

        # Edge bias: project to (B, N, N), then expand across heads to (B*H, N, N)
        bias = self.edge_bias(edge_w.unsqueeze(-1)).squeeze(-1)  # [B, N, N]
        bias = bias.unsqueeze(1).expand(-1, self.heads, -1, -1).reshape(B * self.heads, N, N)

        for layer in self.transformer.layers:
            # Pre-LN
            src = layer.norm1(tokens)
            # Standard MHA with attn_mask
            attn_out, _ = layer.self_attn(
                src, src, src, attn_mask=bias, need_weights=False,
            )
            tokens = tokens + layer.dropout1(attn_out)
            src2 = layer.norm2(tokens)
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(src2))))
            tokens = tokens + layer.dropout2(ff_out)
        return tokens

    def forward(self, x, t, area_target, b2b_conn, p2b_conn, pins_pos, constraints):
        """
        x: [B, N, 4] noisy layout (z-score normalized)
        t: [B] timesteps
        area_target, b2b_conn, p2b_conn, pins_pos, constraints: conditioning
        Returns: pred_noise [B, N, 4] (z-score space, masked by valid)
        """
        B, N, _ = x.shape
        gf = aggregate_graph_features(area_target, b2b_conn, p2b_conn, pins_pos, constraints)
        valid = (area_target != -1).float().unsqueeze(-1)
        x = x * valid

        tokens = self.input_proj(torch.cat([x, gf], dim=-1))
        tokens = tokens + self.cond_proj(gf)
        t_emb = self._time_embedding(t, self.dim)
        tokens = tokens + t_emb.unsqueeze(1)

        h = self._attn_with_edge_bias(tokens, b2b_conn)
        pred_noise = self.output_proj(h)
        return pred_noise * valid
