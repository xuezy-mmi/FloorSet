# import torch
# import torch.nn as nn
# import math

# class DiffusionTransformer(nn.Module):
#     def __init__(self, dim=512, depth=8, heads=8, cond_dim=128, n_steps=1000):
#         super().__init__()
#         self.dim = dim
#         # 时间步嵌入
#         self.time_embed = nn.Sequential(
#             nn.Linear(dim, dim*4),
#             nn.SiLU(),
#             nn.Linear(dim*4, dim)
#         )
#         # 条件投影（将面积、连接等编码为固定维向量）
#         self.cond_proj = nn.Linear(cond_dim, dim)
#         # 输入投影：每个模块的 (w,h,x,y) 4维 -> dim
#         self.input_proj = nn.Linear(4, dim)
#         # Transformer 编码器（处理序列）
#         encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
#         # 输出投影：dim -> 4
#         self.output_proj = nn.Linear(dim, 4)

#     def forward(self, x, t, area_target, b2b_conn, p2b_conn, pins_pos, constraints):
#         """
#         x: [B, N, 4] 带噪布局 (w,h,x,y)
#         t: [B] 时间步
#         其他条件张量见竞赛格式
#         """
#         B, N, _ = x.shape
#         # 1. 时间步嵌入
#         t_emb = self._time_embedding(t, self.dim)  # [B, dim]
#         t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, dim]

#         # 2. 条件编码（需要将多源条件融合为每个模块的 cond vector）
#         # 这里简化：使用 area_target 的对数作为每个模块的条件
#         cond = torch.log(area_target + 1).unsqueeze(-1)  # [B, N, 1]
#         # 扩展为 cond_dim
#         cond = cond.expand(-1, -1, self.cond_proj.in_features)  # 简单复制，实际应更复杂
#         cond_emb = self.cond_proj(cond)  # [B, N, dim]

#         # 3. 输入嵌入
#         x_emb = self.input_proj(x)  # [B, N, dim]

#         # 4. 相加并送入 Transformer
#         tokens = x_emb + cond_emb + t_emb
#         # Transformer 要求 (seq_len, batch, dim)，但 batch_first=True 已支持
#         out = self.transformer(tokens)  # [B, N, dim]

#         # 5. 预测噪声
#         pred_noise = self.output_proj(out)  # [B, N, 4]

#         return pred_noise

#     def _time_embedding(self, t, dim):
#         """ sinusoidal positional embedding for time steps """
#         half = dim // 2
#         freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
#         args = t.unsqueeze(-1).float() * freqs[None]
#         embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#         if dim % 2:
#             embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#         return embedding

import torch
import torch.nn as nn
import math


class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer model that predicts noise from noisy layout and conditions.
    """
    def __init__(
        self,
        dim: int = 512,
        depth: int = 8,
        heads: int = 8,
        cond_dim: int = 128,
        n_steps: int = 1000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.cond_dim = cond_dim
        self.n_steps = n_steps

        self.pos_embed = nn.Parameter(torch.randn(1, 1000, dim) * 0.02, requires_grad=True)

        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(dim, dim * 4),   # 注意：这里 dim 是 int，可以乘法
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

        # 条件投影（将每个模块的原始条件映射到 dim）
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        # 输入投影：每个模块的 (w, h, x, y) → dim
        self.input_proj = nn.Linear(4, dim)

        # Transformer 编码器（处理序列）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 输出投影：dim → 4 (预测噪声)
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 4)
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重，避免数值过大导致 nan"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)

    def forward(self, x, t, area_target, b2b_conn, p2b_conn, pins_pos, constraints):
        """
        Args:
            x: [B, N, 4] 带噪布局 (w, h, x, y)
            t: [B] 时间步索引
            area_target: [B, N] 模块目标面积 (未归一化)
            b2b_conn, p2b_conn, pins_pos, constraints: 条件信息
        Returns:
            pred_noise: [B, N, 4]
        """
        B, N, _ = x.shape
        device = x.device

        # 1. 时间步嵌入
        t_emb = self._time_embedding(t, self.dim)  # [B, dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, dim]

        # 2. 条件编码（简化版：只用面积作为条件）
        valid_mask = (area_target != -1).float()  # [B, N]
        safe_area = torch.where(area_target > 0, area_target, torch.ones_like(area_target))
        log_area = torch.log(safe_area) * valid_mask
        cond_feat = log_area.unsqueeze(-1)  # [B, N, 1]
        # 扩展为 cond_dim
        cond_feat = cond_feat.expand(-1, -1, self.cond_dim)
        cond_emb = self.cond_proj(cond_feat)

        # 3. 输入嵌入
        x_emb = self.input_proj(x)

        # 4. 相加并加位置编码
        tokens = x_emb + cond_emb + t_emb
        if not hasattr(self, 'pos_embed'):
            # 动态创建位置编码，确保与 tokens 同设备
            self.pos_embed = nn.Parameter(torch.randn(1, N, self.dim) * 0.02).to(device)
        else:
            # 如果预定义的 pos_embed 长度不足，扩展它
            if self.pos_embed.shape[1] < N:
                new_embed = torch.randn(1, N - self.pos_embed.shape[1], self.dim) * 0.02
                self.pos_embed = nn.Parameter(torch.cat([self.pos_embed, new_embed.to(device)], dim=1))
        tokens = tokens + self.pos_embed[:, :N, :]

        out = self.transformer(tokens)
        pred_noise = self.output_proj(out)
        return pred_noise

    def _time_embedding(self, t, dim):
        """正弦余弦时间嵌入"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t.unsqueeze(-1).float() * freqs[None]  # [B, half]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding