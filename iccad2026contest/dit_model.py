"""
dit_model.py - OLD v1 DiT model (matching trained checkpoint exactly).

Architecture (from checkpoint analysis):
  - dim=512, depth=8, heads=8, cond_dim=128
  - Learnable pos_embed [1, 1000, 512]
  - time_embed: Linear(dim, dim*4) → SiLU → Linear(dim*4, dim)
  - cond_proj: Linear(cond_dim, dim) → SiLU → Linear(dim, dim)
  - input_proj: Linear(4, dim)
  - 8-layer Post-LN TransformerEncoder (standard, not Pre-LN)
  - output_proj: Linear(dim, dim) → GELU → Linear(dim, 4)
  - No edge_bias attention

This matches the checkpoint at: iccad2026contest/model/diffusion_final.pth
"""
import math

import torch
import torch.nn as nn

from dit_utils import aggregate_graph_features_v1


class DiffusionTransformer(nn.Module):
    """
    OLD v1 DiT: matches the trained checkpoint architecture exactly.
    Standard Post-LN Transformer with learnable positional embeddings.
    """
    def __init__(self, dim: int = 512, depth: int = 8, heads: int = 8,
                 cond_dim: int = 128, n_steps: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.cond_dim = cond_dim
        self.n_steps = n_steps

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, 256, dim) * 0.02, requires_grad=True
        )

        # Time embedding: dim → dim*4 → dim
        self.time_embed = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

        # Condition projection: cond_dim → dim → dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        # Input projection: 4 → dim
        self.input_proj = nn.Linear(4, dim)

        # 8-layer Post-LN Transformer (standard, not Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )

        # Output projection: dim → dim → 4
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 4)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization (matching original v1 training)."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)

    def _time_embedding(self, t, dim):
        """Sinusoidal time embedding."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t.unsqueeze(-1).float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, x, t, area_target, b2b_conn, p2b_conn, pins_pos, constraints):
        """
        x: [B, N, 4] noisy layout
        t: [B] timesteps
        area_target: [B, N] target areas
        Returns: pred_noise [B, N, 4]
        """
        B, N, _ = x.shape
        device = x.device

        # 1. Time embedding
        t_emb = self._time_embedding(t, self.dim)  # [B, dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, dim]

        # 2. Condition encoding (log_area per block, expanded to cond_dim)
        valid_mask = (area_target != -1).float()  # [B, N]
        safe_area = torch.where(area_target > 0, area_target, torch.ones_like(area_target))
        log_area = torch.log(safe_area) * valid_mask  # [B, N]
        cond_feat = log_area.unsqueeze(-1).expand(-1, -1, self.cond_dim)  # [B, N, cond_dim]
        cond_emb = self.cond_proj(cond_feat)  # [B, N, dim]

        # 3. Input embedding
        x_emb = self.input_proj(x)  # [B, N, dim]

        # 4. Combine + positional embedding
        tokens = x_emb + cond_emb + t_emb
        # Extend pos_embed if N > 1000
        if self.pos_embed.shape[1] < N:
            new_embed = torch.randn(1, N - self.pos_embed.shape[1], self.dim) * 0.02
            self.pos_embed = nn.Parameter(
                torch.cat([self.pos_embed.detach(), new_embed.to(device)], dim=1),
                requires_grad=True
            )
        tokens = tokens + self.pos_embed[:, :N, :]

        # 5. Transformer
        out = self.transformer(tokens)  # [B, N, dim]

        # 6. Predict noise
        pred_noise = self.output_proj(out)  # [B, N, 4]

        # Mask invalid blocks
        valid = (area_target != -1).float().unsqueeze(-1)
        return pred_noise * valid