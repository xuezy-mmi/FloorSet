# dit_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Positional Embedding
# ----------------------------------------------------------------------
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


# ----------------------------------------------------------------------
# Cross-Attention Block (DiT building block)
# ----------------------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, cond, mask=None, cond_mask=None):
        # self-attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), x, key_padding_mask=mask)[0]
        # cross-attention with condition
        x = x + self.cross_attn(self.norm2(x), cond, cond, key_padding_mask=cond_mask)[0]
        # MLP
        x = x + self.mlp(self.norm3(x))
        return x


# ----------------------------------------------------------------------
# Condition Encoder (graph + area + constraints)
# ----------------------------------------------------------------------
class ConditionEncoder(nn.Module):
    def __init__(self, feat_dim=64, hidden_dim=256, max_blocks=120):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # module features: log(area) + is_fixed? + is_preplaced? (3 dims)
        self.module_proj = nn.Linear(3, feat_dim)
        # connection weight projection (average edge weight per node)
        self.conn_proj = nn.Linear(1, feat_dim)
        # learnable positional encoding (max_blocks)
        self.pos_embed = nn.Embedding(max_blocks, feat_dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(feat_dim, nhead=8, batch_first=True),
            num_layers=4
        )

    def forward(self, area_targets, constraints, b2b_connectivity, target_positions=None):
        """
        Args:
            area_targets: torch.Tensor [B, K] (target area, -1 for padding)
            constraints: torch.Tensor [B, K, 5] (fixed, preplaced, mib, cluster, boundary)
            b2b_connectivity: torch.Tensor [B, E, 3] (i, j, weight)
            target_positions: optional [B, K, 4] (x,y,w,h) for fixed/preplaced
        Returns:
            cond: [B, K, hidden_dim]
            mask: [B, K] (True for valid blocks)
        """
        B, K = area_targets.shape
        device = area_targets.device

        # mask for valid blocks (area != -1)
        valid_mask = (area_targets != -1).float()  # [B,K]

        # Build module features: [log(area+1), is_fixed, is_preplaced]
        log_area = torch.log(area_targets + 1e-6).unsqueeze(-1)  # [B,K,1]
        is_fixed = (constraints[..., 0] != 0).float().unsqueeze(-1)  # [B,K,1]
        is_preplaced = (constraints[..., 1] != 0).float().unsqueeze(-1)  # [B,K,1]
        feat = torch.cat([log_area, is_fixed, is_preplaced], dim=-1)  # [B,K,3]
        feat = self.module_proj(feat)  # [B,K,feat_dim]

        # Add positional embedding
        positions = torch.arange(K, device=device).unsqueeze(0).expand(B, -1)
        feat = feat + self.pos_embed(positions)

        # Graph information: average connection weight per node
        # b2b_connectivity: [B, E, 3] -> compute node degree and weighted sum
        # Simplified: for each node, average weight of all incident edges
        conn_avg = torch.zeros(B, K, 1, device=device)
        for b in range(B):
            edges = b2b_connectivity[b]
            valid_edges = edges[edges[:, 0] != -1]
            if valid_edges.shape[0] > 0:
                i = valid_edges[:, 0].long()
                j = valid_edges[:, 1].long()
                w = valid_edges[:, 2]
                # accumulate weights
                sum_w = torch.zeros(K, device=device)
                cnt = torch.zeros(K, device=device)
                sum_w.index_add_(0, i, w)
                sum_w.index_add_(0, j, w)
                cnt.index_add_(0, i, torch.ones_like(i).float())
                cnt.index_add_(0, j, torch.ones_like(j).float())
                avg = sum_w / (cnt + 1e-6)
                conn_avg[b, :, 0] = avg
        graph_feat = self.conn_proj(conn_avg)  # [B,K,feat_dim]
        feat = feat + graph_feat

        # Transformer encoding (only valid positions)
        # We'll use a mask to ignore padding during attention
        key_padding_mask = (valid_mask == 0)  # True for padding
        # Convert to float and set padding to zero for attention stability
        feat = feat * valid_mask.unsqueeze(-1)
        cond = self.transformer(feat, src_key_padding_mask=key_padding_mask)  # [B,K,feat_dim]

        # Project to hidden_dim
        cond = nn.Linear(self.feat_dim, self.hidden_dim).to(device)(cond)
        return cond, valid_mask.bool()


# ----------------------------------------------------------------------
# Diffusion Transformer Model
# ----------------------------------------------------------------------
class DiffusionTransformer(nn.Module):
    """
    Predicts noise from noisy layout + timestep + condition.
    Input: x [B, K, 4], t [B], cond [B, K, cond_dim]
    Output: noise_pred [B, K, 4]
    """
    def __init__(self, dim=256, depth=12, num_heads=8, cond_dim=256):
        super().__init__()
        self.dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.input_proj = nn.Linear(4, dim)
        self.cond_proj = nn.Linear(cond_dim, dim)

        self.blocks = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads) for _ in range(depth)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )

    def forward(self, x, t, cond, mask=None):
        """
        x: [B, K, 4]
        t: [B]
        cond: [B, K, cond_dim]
        mask: [B, K] (True for valid blocks)
        """
        x = self.input_proj(x)          # [B,K,dim]
        t_emb = self.time_mlp(t).unsqueeze(1)   # [B,1,dim]
        x = x + t_emb

        cond = self.cond_proj(cond)      # [B,K,dim]

        # apply mask: if mask is provided, we set padded positions to zero
        if mask is not None:
            mask_pad = ~mask  # True for padding
            x = x.masked_fill(mask_pad.unsqueeze(-1), 0)
            # For attention, we need key_padding_mask: True for padded tokens
            key_padding_mask = mask_pad
        else:
            key_padding_mask = None

        for block in self.blocks:
            x = block(x, cond, mask=key_padding_mask)

        noise_pred = self.output_proj(x)
        return noise_pred