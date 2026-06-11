"""
dit_model_hd.py - HouseDiffusion-inspired DiT for VLSI floorplanning.

Adapted from HouseDiffusion (Shabani et al., 2022). Key ideas borrowed:

  1. **3-branch masked multi-head attention** (HouseDiffusion's
     door_attn + self_attn + gen_attn). We replace the per-room masks
     with three VLSI-relevant ones:
        - `wire_attn` with `wire_mask`  : attention only between blocks
           that share a b2b edge (high-wire blocks should communicate).
        - `clust_attn` with `clust_mask` : attention only between blocks
           in the same MIB / cluster group (so they coordinate shape /
           abutment).
        - `glob_attn` with `glob_mask`  : full attention, used as global
           context (analogous to HouseDiffusion's `gen_attn`).

  2. **Per-block conditioning embeddings** (HouseDiffusion's
     room_types/corner_indices/room_indices). We use 12 features built
     from the 5 constraint columns + per-block graph features.

  3. **Sinusoidal positional encoding** (HouseDiffusion's
     PositionalEncoding) rather than a learnable per-block embedding.

  4. **Pre-LN TransformerEncoder** for training stability.

  5. **Mask semantics** are reversed from PyTorch's `attn_mask`:
     `mask == 1` means *block* (do not attend), `mask == 0` means
     *attend*. Score is filled with the *dtype's min* (not `-1e9`) so
     the layer works under fp16 autocast. This is exactly HouseDiffusion's
     convention.

Key change from the previous HD attempt (VLSI-specific):
  - **Positive, bounded normalization** (`x / norm_factor`, default 1000)
    rather than z-score. With z-score centering, denormalizing the
    reverse process produced *negative* x/y positions; the plain
    positive divisor (matching the working original DiT) makes
    `clamp(x0, 0, 1)` in normalized space equivalent to
    `clamp(0, norm_factor)` in raw space — i.e. all w, h, x, y are
    guaranteed non-negative.
  - Hard constraints are injected at *inference time* by overwriting
    fixed/preplaced dimensions in the denoised sample, not by adding
    them to the loss.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Sinusoidal positional encoding (HouseDiffusion's PositionalEncoding)
# ----------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Standard sin/cos PE matching HouseDiffusion's PositionalEncoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        x = x + self.pe[0:1, : x.size(1)]
        return self.dropout(x)


# ----------------------------------------------------------------------------
# Time embedding (HouseDiffusion uses guided-diffusion's timestep_embedding)
# ----------------------------------------------------------------------------
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000):
    """Sinusoidal time embedding, matching guided-diffusion."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


# ----------------------------------------------------------------------------
# Multi-head attention with mask == 1 → block convention
# (HouseDiffusion-style)
# ----------------------------------------------------------------------------
class MaskedMultiHeadAttention(nn.Module):
    """Multi-head attention where `mask == 1` blocks attention.

    Mirrors HouseDiffusion.house_diffusion.transformer.MultiHeadAttention
    but written to be importable on its own and easier to read.
    """

    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """q, k, v: [B, S, D]; mask: [B, S, S] (1 = block, 0 = attend)."""
        B, S, _ = q.shape
        q_ = self.q_linear(q).view(B, S, self.heads, self.d_k).transpose(1, 2)
        k_ = self.k_linear(k).view(B, S, self.heads, self.d_k).transpose(1, 2)
        v_ = self.v_linear(v).view(B, S, self.heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q_, k_.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask == 1 means block: fill with -inf. Use dtype min so it
            # works under fp16 autocast (where -1e9 overflows).
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask.unsqueeze(1) == 1, neg_inf)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v_)  # [B, H, S, d_k]
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out(out)


# ----------------------------------------------------------------------------
# Encoder layer with 3 parallel masked attentions
# (HouseDiffusion's EncoderLayer, ported to pre-LN)
# ----------------------------------------------------------------------------
class ThreeBranchEncoderLayer(nn.Module):
    """Pre-LN encoder block with three parallel masked attentions.

    Layout (per layer):
        x = x + wire_attn(norm(x), wire_mask)
              + clust_attn(norm(x), clust_mask)
              + glob_attn(norm(x), glob_mask)
        x = x + ff(norm(x))
    """

    def __init__(
        self, d_model: int, heads: int = 4, dropout: float = 0.1,
        activation: nn.Module = None,
    ):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.wire_attn = MaskedMultiHeadAttention(d_model, heads, dropout)
        self.clust_attn = MaskedMultiHeadAttention(d_model, heads, dropout)
        self.glob_attn = MaskedMultiHeadAttention(d_model, heads, dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            activation if activation is not None else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        wire_mask: torch.Tensor,
        clust_mask: torch.Tensor,
        glob_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm_1(x)
        x = x + self.dropout(self.wire_attn(h, h, h, wire_mask))
        x = x + self.dropout(self.clust_attn(h, h, h, clust_mask))
        x = x + self.dropout(self.glob_attn(h, h, h, glob_mask))

        h2 = self.norm_2(x)
        x = x + self.dropout(self.ff(h2))
        return x


# ----------------------------------------------------------------------------
# Top-level HouseDiffusion-inspired DiT
# ----------------------------------------------------------------------------
class HouseDiffusionDiT(nn.Module):
    """DiT for VLSI floorplanning with HouseDiffusion-style 3-branch attention.

    Forward signature matches the existing `DiffusionTransformer` so the
    same training plumbing can call it.

    Input/output convention (matches the working original DiT):
        - x: [B, N, 4] in (w, h, x, y) order, already divided by
             `norm_factor` (e.g. 1000.0). With norm_factor=1000 and the
             typical FloorSet-Lite layout range, normalized values land
             in roughly [0, 0.5].
        - The model predicts the noise added at timestep t (epsilon
          parameterization). During inference, the optimizer re-runs the
          reverse process to recover a clean x0, then un-normalizes via
          `x0_raw = x0_norm * norm_factor` and clamps to [0, +inf).
    """

    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        heads: int = 4,
        cond_in: int = 12,
        n_steps: int = 1000,
        dropout: float = 0.1,
        max_blocks: int = 200,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.cond_in = cond_in
        self.n_steps = n_steps
        self.max_blocks = max_blocks

        # Input: 4-D noisy layout (normalized) + per-block conditioning
        self.input_proj = nn.Sequential(
            nn.Linear(4 + cond_in, dim),
            nn.LayerNorm(dim),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in, dim), nn.SiLU(), nn.Linear(dim, dim),
        )
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )

        # Sinusoidal PE (HouseDiffusion style)
        self.pos_encoder = SinusoidalPositionalEncoding(
            dim, dropout=dropout, max_len=max_blocks
        )

        # Stack of 3-branch encoder layers
        self.layers = nn.ModuleList([
            ThreeBranchEncoderLayer(
                d_model=dim, heads=heads, dropout=dropout, activation=nn.GELU(),
            )
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(dim)

        # Output projection (4 channels: noise in (w, h, x, y) normalized space)
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim), nn.Linear(dim, 4),
        )

        # Init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _time_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        return timestep_embedding(t, dim)

    def _build_wire_mask(
        self,
        b2b_conn: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Mask that allows attention only between blocks sharing a b2b edge.

        Returns a [B, N, N] mask with 1=block, 0=attend.
        """
        B, N = valid.shape
        device = b2b_conn.device if b2b_conn is not None else valid.device

        # Start blocked everywhere; diagonal self-attends
        mask = torch.ones(B, N, N, device=device, dtype=valid.dtype)
        eye = torch.eye(N, device=device, dtype=valid.dtype).unsqueeze(0)
        mask = mask * (1 - eye)

        if b2b_conn is not None and b2b_conn.numel() > 0:
            m = b2b_conn[..., 0] >= 0
            if m.any():
                e = b2b_conn[m]
                i = e[:, 0].long().clamp(0, N - 1)
                j = e[:, 1].long().clamp(0, N - 1)
                b_idx = torch.arange(B, device=device).unsqueeze(1)
                mask[b_idx, i, j] = 0.0
                mask[b_idx, j, i] = 0.0  # symmetric

        # mask padded rows/cols
        valid_f = valid.float()
        row = (1.0 - valid_f).unsqueeze(2)  # [B, N, 1]
        col = (1.0 - valid_f).unsqueeze(1)  # [B, 1, N]
        mask = mask + row + col
        mask = mask.clamp(0.0, 1.0)
        return mask

    def _build_clust_mask(
        self,
        constraints: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Mask that allows attention only between blocks in the same
        MIB group (col 2) or same cluster group (col 3).

        Returns a [B, N, N] mask with 1=block, 0=attend.
        """
        B, N = valid.shape
        device = valid.device

        mib = constraints[..., 2]  # [B, N]
        clust = constraints[..., 3]
        same_mib = (mib > 0).unsqueeze(2) & (mib > 0).unsqueeze(1) & (
            mib.unsqueeze(2) == mib.unsqueeze(1)
        )
        same_clust = (clust > 0).unsqueeze(2) & (clust > 0).unsqueeze(1) & (
            clust.unsqueeze(2) == clust.unsqueeze(1)
        )
        attend = (same_mib | same_clust).float()
        eye = torch.eye(N, device=device).unsqueeze(0)
        attend = torch.maximum(attend, eye)
        mask = 1.0 - attend  # 1 = block

        valid_f = valid.float()
        row = (1.0 - valid_f).unsqueeze(2)
        col = (1.0 - valid_f).unsqueeze(1)
        mask = mask + row + col
        return mask.clamp(0.0, 1.0)

    def _build_global_mask(self, valid: torch.Tensor) -> torch.Tensor:
        """Global attention mask: attend to everything (except padded)."""
        B, N = valid.shape
        device = valid.device
        mask = torch.zeros(B, N, N, device=device, dtype=valid.dtype)
        valid_f = valid.float()
        row = (1.0 - valid_f).unsqueeze(2)
        col = (1.0 - valid_f).unsqueeze(1)
        mask = mask + row + col
        return mask.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,            # [B, N, 4] noisy layout (normalized)
        t: torch.Tensor,            # [B] timesteps
        area_target: torch.Tensor,  # [B, N] (with -1 for padded)
        b2b_conn: torch.Tensor,     # [B, E, 3] (with -1 for padded)
        p2b_conn: torch.Tensor,     # [B, E, 3]
        pins_pos: torch.Tensor,     # [B, P, 2]
        constraints: torch.Tensor,  # [B, N, 5]
    ) -> torch.Tensor:
        """Returns predicted noise [B, N, 4] in normalized space."""
        B, N, _ = x.shape
        device = x.device
        valid = (area_target != -1).float()  # [B, N]
        # zero out padded positions (model is not asked to predict noise for them)
        x = x * valid.unsqueeze(-1)

        # ----- per-block conditioning features (12) -----
        # bounded to small ranges so they don't dominate the input proj.
        safe = torch.where(area_target > 0, area_target, torch.ones_like(area_target))
        log_area = torch.log(safe) * (area_target != -1).float()

        b2b_w = torch.zeros(B, N, device=device, dtype=x.dtype)
        b2b_d = torch.zeros(B, N, device=device, dtype=x.dtype)
        if b2b_conn is not None and b2b_conn.numel() > 0:
            m = b2b_conn[..., 0] >= 0
            i_idx = b2b_conn[..., 0].long().clamp(0, N - 1)
            j_idx = b2b_conn[..., 1].long().clamp(0, N - 1)
            w = b2b_conn[..., 2] * m.float()
            b2b_w.scatter_add_(1, i_idx, w)
            b2b_w.scatter_add_(1, j_idx, w)
            b2b_d.scatter_add_(1, i_idx, m.float())
            b2b_d.scatter_add_(1, j_idx, m.float())

        p2b_w = torch.zeros(B, N, device=device, dtype=x.dtype)
        p2b_px = torch.zeros(B, N, device=device, dtype=x.dtype)
        p2b_py = torch.zeros(B, N, device=device, dtype=x.dtype)
        if (
            p2b_conn is not None and p2b_conn.numel() > 0
            and pins_pos is not None and pins_pos.numel() > 0
        ):
            m = p2b_conn[..., 0] >= 0
            pin_idx = p2b_conn[..., 0].long().clamp(0)
            blk_idx = p2b_conn[..., 1].long().clamp(0, N - 1)
            wt = p2b_conn[..., 2] * m.float()
            P = pins_pos.shape[1]
            pin_idx_g = pin_idx.clamp(max=P - 1)
            px = pins_pos[..., 0].gather(1, pin_idx_g) * m.float()
            py = pins_pos[..., 1].gather(1, pin_idx_g) * m.float()
            p2b_w.scatter_add_(1, blk_idx, wt)
            p2b_px.scatter_add_(1, blk_idx, wt * px)
            p2b_py.scatter_add_(1, blk_idx, wt * py)

        is_hard = ((constraints[..., 0] > 0) | (constraints[..., 1] > 0)).float()
        boundary = constraints[..., 4].float() / 15.0
        mib_norm = (constraints[..., 2].float() / 16.0).clamp(0.0, 1.0)
        clust_norm = (constraints[..., 3].float() / 16.0).clamp(0.0, 1.0)
        p2b_w_log = torch.log1p(p2b_w)

        gf = torch.stack([
            log_area, b2b_w, b2b_d, p2b_w_log, p2b_px, p2b_py,
            is_hard, boundary, mib_norm, clust_norm,
            valid, valid,  # padding indicators
        ], dim=-1)  # [B, N, 12]

        # ----- embeddings -----
        tokens = self.input_proj(torch.cat([x, gf], dim=-1))
        tokens = tokens + self.cond_proj(gf)
        t_emb = self._time_embedding(t, self.dim)
        tokens = tokens + t_emb.unsqueeze(1)

        # positional encoding
        tokens = self.pos_encoder(tokens)

        # ----- 3-branch attention -----
        wire_mask = self._build_wire_mask(b2b_conn, valid)
        clust_mask = self._build_clust_mask(constraints, valid)
        glob_mask = self._build_global_mask(valid)

        for layer in self.layers:
            tokens = layer(tokens, wire_mask, clust_mask, glob_mask)
        tokens = self.final_norm(tokens)

        pred_noise = self.output_proj(tokens)
        return pred_noise * valid.unsqueeze(-1)
