import torch
import math

class DiffusionScheduler:
    def __init__(self, n_steps, beta_start=1e-4, beta_end=0.02):
        self.n_steps = n_steps
        self.beta = torch.linspace(beta_start, beta_end, n_steps)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)

# def q_sample(x0, t, alpha_cumprod, noise=None):
#     """Forward diffusion: add noise to x0 at step t."""
#     if noise is None:
#         noise = torch.randn_like(x0)
#     sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod[t])[:, None, None]  # [B,1,1]
#     sqrt_one_minus = torch.sqrt(1 - alpha_cumprod[t])[:, None, None]
#     return sqrt_alpha_cumprod * x0 + sqrt_one_minus * noise, noise

# def q_sample(x0, t, alpha_cumprod, noise=None):
#     if noise is None:
#         noise = torch.randn_like(x0)
#     # 确保 alpha_cumprod 与 x0 同设备
#     if alpha_cumprod.device != t.device:
#         alpha_cumprod = alpha_cumprod.to(t.device)
#     alpha_cumprod_t = alpha_cumprod[t].view(-1, 1, 1)
#     sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
#     sqrt_one_minus = torch.sqrt(1 - alpha_cumprod_t)
#     return sqrt_alpha_cumprod * x0 + sqrt_one_minus * noise, noise


def q_sample(x0, t, alpha_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    if alpha_cumprod.device != t.device:
        alpha_cumprod = alpha_cumprod.to(t.device)
    alpha_cumprod_t = alpha_cumprod[t].view(-1, 1, 1)
    # 避免数值误差导致负数
    alpha_cumprod_t = torch.clamp(alpha_cumprod_t, min=0.0, max=1.0)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod_t)
    sqrt_one_minus = torch.sqrt(1 - alpha_cumprod_t)
    return sqrt_alpha_cumprod * x0 + sqrt_one_minus * noise, noise