#!/usr/bin/env python3
"""
train_dit_v2.py - DiT training v2 (items 1-3 from improvement plan)

Improvements over v1:
  1) Use a vectorized version of the contest cost as PRIMARY loss, with a
     small weight on diffusion noise loss.
  2) Per-channel z-score normalization on fp_sol (w, h, x, y) — fixes the
     scale mismatch (w,h are ~10-200, x,y are ~100-10000).
  3) Use x_0 parametrization during training (reconstruct x_0 in one step
     from the noise prediction) so we can drive the diff loss without
     running a full reverse loop. Inference uses DDIM (50 steps) for speed.

The vectorized diff loss mirrors iccad2026_evaluate.compute_training_loss_differentiable
but replaces the O(N^2) Python overlap loop with a [N,N] tensor op.

Saves to /home/xzy/eda/model/v2/diffusion_final.pth with a dict:
  {
    'model_state_dict': ...,
    'ema_state_dict':  ...,
    'norm_stats': {'mu': (4,), 'sigma': (4,)},
    'n_steps': 1000, 'cond_dim': 128,
    'model_kwargs': {...},
  }
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from iccad2026_evaluate import get_training_dataloader
from dit_model import DiffusionTransformer
from dit_utils import DiffusionScheduler, q_sample


SAVE_DIR = Path("/home/xzy/eda/model/v2")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_STEPS = 1000
N_EPOCHS = 12
BATCH_SIZE = 8
LR = 5e-5
NUM_SAMPLES = 2000
LAMBDA_NOISE = 0.05
GRAD_CLIP = 1.0
ALPHA, BETA = 0.5, 2.0


# -----------------------------------------------------------------------------
# Vectorized contest-cost proxy (mirrors compute_training_loss_differentiable)
# -----------------------------------------------------------------------------
def vectorized_diff_loss(
    positions: torch.Tensor,        # [N, 4] = (x, y, w, h)
    b2b_conn: torch.Tensor,
    p2b_conn: torch.Tensor,
    pins_pos: torch.Tensor,
    area_targets: torch.Tensor,
    baseline_metrics: torch.Tensor,
) -> torch.Tensor:
    N = positions.shape[0]
    x, y, w, h = positions[:, 0], positions[:, 1], positions[:, 2], positions[:, 3]
    cx, cy = x + w / 2, y + h / 2

    # HPWL b2b
    hpwl_b2b = torch.zeros((), device=positions.device, dtype=positions.dtype)
    if b2b_conn is not None and b2b_conn.numel() > 0:
        mask = b2b_conn[:, 0] >= 0
        if mask.any():
            e = b2b_conn[mask]
            i, j = e[:, 0].long(), e[:, 1].long()
            valid_ij = (i < N) & (j < N)
            i, j, wt = i[valid_ij], j[valid_ij], e[valid_ij, 2]
            if i.numel():
                hpwl_b2b = (wt * (torch.abs(cx[i] - cx[j]) + torch.abs(cy[i] - cy[j]))).sum()

    # HPWL p2b
    hpwl_p2b = torch.zeros((), device=positions.device, dtype=positions.dtype)
    if p2b_conn is not None and p2b_conn.numel() > 0:
        mask = p2b_conn[:, 0] >= 0
        if mask.any():
            e = p2b_conn[mask]
            pin_idx, blk_idx = e[:, 0].long(), e[:, 1].long()
            wt = e[:, 2]
            valid = (pin_idx < pins_pos.shape[0]) & (blk_idx < N)
            pin_idx, blk_idx, wt = pin_idx[valid], blk_idx[valid], wt[valid]
            if pin_idx.numel():
                px = pins_pos[pin_idx, 0]
                py = pins_pos[pin_idx, 1]
                hpwl_p2b = (wt * (torch.abs(cx[blk_idx] - px) + torch.abs(cy[blk_idx] - py))).sum()

    hpwl_total = hpwl_b2b + hpwl_p2b

    # BBox area
    bbox_area = (x.max() - x.min()).clamp(min=0) * (y.max() - y.min()).clamp(min=0)

    # Overlap (vectorized upper-triangle)
    xi, xj = x.unsqueeze(0), x.unsqueeze(1)
    yi, yj = y.unsqueeze(0), y.unsqueeze(1)
    wi, wj = w.unsqueeze(0), w.unsqueeze(1)
    hi, hj = h.unsqueeze(0), h.unsqueeze(1)
    ox = torch.relu(torch.min(xi + wi, xj + wj) - torch.max(xi, xj))
    oy = torch.relu(torch.min(yi + hi, yj + hj) - torch.max(yi, yj))
    tri = torch.triu(torch.ones(N, N, device=positions.device, dtype=positions.dtype), diagonal=1)
    overlap_area = ((ox * oy) * tri).sum()
    total_block_area = (w * h).sum().clamp(min=1e-6)
    overlap_v = overlap_area / total_block_area

    # Area tolerance
    valid_a = area_targets > 0
    if valid_a.any():
        err = torch.zeros_like(area_targets)
        err[valid_a] = torch.abs(w[valid_a] * h[valid_a] - area_targets[valid_a]) / area_targets[valid_a]
        area_v = torch.relu(err - 0.01).sum() / (valid_a.sum().float() + 1e-6)
    else:
        area_v = torch.zeros((), device=positions.device, dtype=positions.dtype)

    V_soft = torch.clamp(overlap_v + area_v, max=5.0)  # cap to keep training stable

    # Gaps
    baseline_area = baseline_metrics[0]
    baseline_hpwl = baseline_metrics[6] + baseline_metrics[7]
    hpwl_gap = torch.relu((hpwl_total - baseline_hpwl) / (baseline_hpwl + 1e-6))
    area_gap = torch.relu((bbox_area - baseline_area) / (baseline_area + 1e-6))

    cost = (1 + ALPHA * (hpwl_gap + area_gap)) * torch.exp(BETA * V_soft)
    return cost


# -----------------------------------------------------------------------------
# Per-channel (w, h, x, y) mean/std from a few batches
# -----------------------------------------------------------------------------
def compute_norm_stats(dataloader, max_batches: int = 64):
    ws, hs, xs, ys = [], [], [], []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        _, _, _, _, _, _, fp_sol, _ = batch
        fp = fp_sol.squeeze(0)
        valid = (fp != -1).all(dim=-1)
        fv = fp[valid]
        if fv.numel() == 0:
            continue
        ws.append(fv[:, 0])
        hs.append(fv[:, 1])
        xs.append(fv[:, 2])
        ys.append(fv[:, 3])
    w = torch.cat(ws).float()
    h = torch.cat(hs).float()
    x = torch.cat(xs).float()
    y = torch.cat(ys).float()
    mu = torch.stack([w.mean(), h.mean(), x.mean(), y.mean()])
    sigma = torch.stack([
        w.std().clamp(min=1.0),
        h.std().clamp(min=1.0),
        x.std().clamp(min=1.0),
        y.std().clamp(min=1.0),
    ])
    return mu, sigma


# -----------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print("Loading data ...")
    train_loader = get_training_dataloader(
        data_path="/home/xzy/eda/",
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES,
        shuffle=True,
    )
    print(f"  train batches/epoch: {len(train_loader)}")

    # z-score stats
    print("Computing z-score stats ...")
    stat_loader = get_training_dataloader(
        data_path="/home/xzy/eda/",
        batch_size=8,
        num_samples=512,
        shuffle=False,
    )
    mu, sigma = compute_norm_stats(stat_loader, max_batches=64)
    print(f"  mu    = {mu.tolist()}")
    print(f"  sigma = {sigma.tolist()}")
    mu, sigma = mu.to(DEVICE), sigma.to(DEVICE)

    # model + EMA
    model = DiffusionTransformer(
        dim=512, depth=8, heads=8, cond_dim=128, n_steps=N_STEPS
    ).to(DEVICE)
    ema = DiffusionTransformer(
        dim=512, depth=8, heads=8, cond_dim=128, n_steps=N_STEPS
    ).to(DEVICE)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = DiffusionScheduler(N_STEPS)
    alpha_cumprod = scheduler.alpha_cumprod.to(DEVICE)

    print("Begin training ...")
    for epoch in range(N_EPOCHS):
        total_loss, total_diff, total_noise, n_batches = 0.0, 0.0, 0.0, 0
        for batch_idx, batch in enumerate(train_loader):
            area_target, b2b, p2b, pins, constraints, tree_sol, fp_sol, metrics = batch
            area_target = area_target.to(DEVICE)
            b2b = b2b.to(DEVICE)
            p2b = p2b.to(DEVICE)
            pins = pins.to(DEVICE)
            constraints = constraints.to(DEVICE)
            fp_sol = fp_sol.to(DEVICE)
            metrics = metrics.to(DEVICE)

            B, N_max, _ = fp_sol.shape
            mask = (area_target != -1).unsqueeze(-1).expand_as(fp_sol).float()

            # (1) z-score normalize
            x0 = ((fp_sol - mu) / sigma) * mask

            t = torch.randint(0, N_STEPS, (B,), device=DEVICE)
            noise = torch.randn_like(x0)
            x_t, _ = q_sample(x0, t, alpha_cumprod, noise)

            pred_noise = model(x_t, t, area_target, b2b, p2b, pins, constraints)

            # noise loss
            loss_noise = F.mse_loss(pred_noise * mask, noise * mask, reduction='sum') / (mask.sum() + 1e-6)

            # x_0 reconstruction (x_0 parametrization)
            a_bar = alpha_cumprod[t].view(-1, 1, 1).clamp(min=1e-6)
            x0_pred = ((x_t - torch.sqrt(1 - a_bar) * pred_noise) / torch.sqrt(a_bar)) * mask
            x0_pred = x0_pred.clamp(-5.0, 5.0)  # also enforce at training; reduces sampler mismatch

            # unnormalize back to (w, h, x, y) real scale
            pos_real = x0_pred * sigma + mu
            pos_real = pos_real.clamp(min=0.0)

            # diff loss per sample
            loss_diff = torch.zeros((), device=DEVICE)
            count = 0
            for i in range(B):
                valid_i = area_target[i] != -1
                n_v = int(valid_i.sum().item())
                if n_v < 2:
                    continue
                # reorder: (x, y, w, h)
                pos_i = torch.stack([
                    pos_real[i, :n_v, 2],
                    pos_real[i, :n_v, 3],
                    pos_real[i, :n_v, 0],
                    pos_real[i, :n_v, 1],
                ], dim=-1)
                # guard: enforce hard constraints from ground truth during training
                if constraints is not None:
                    fixed = constraints[i, :n_v, 0] > 0
                    preplaced = constraints[i, :n_v, 1] > 0
                    if fixed.any():
                        pos_i[fixed, 2] = fp_sol[i, :n_v][fixed, 0]
                        pos_i[fixed, 3] = fp_sol[i, :n_v][fixed, 1]
                    if preplaced.any():
                        # fp_sol layout is (w, h, x, y); pos_i layout is (x, y, w, h)
                        pos_i[preplaced, 0] = fp_sol[i, :n_v][preplaced, 2]
                        pos_i[preplaced, 1] = fp_sol[i, :n_v][preplaced, 3]
                        pos_i[preplaced, 2] = fp_sol[i, :n_v][preplaced, 0]
                        pos_i[preplaced, 3] = fp_sol[i, :n_v][preplaced, 1]

                loss_diff = loss_diff + vectorized_diff_loss(
                    pos_i, b2b[i], p2b[i], pins[i],
                    area_target[i, :n_v], metrics[i],
                )
                count += 1
            loss_diff = loss_diff / max(count, 1)

            # explicit area loss to prevent w,h collapse
            with torch.no_grad():
                pass
            area_loss = torch.zeros((), device=DEVICE)
            for i in range(B):
                valid_i = area_target[i] != -1
                n_v = int(valid_i.sum().item())
                if n_v < 2:
                    continue
                wv = pos_real[i, :n_v, 0]
                hv = pos_real[i, :n_v, 1]
                av = area_target[i, :n_v]
                mask = av > 0
                if mask.any():
                    rel = (wv[mask] * hv[mask] - av[mask]).abs() / (av[mask] + 1e-6)
                    area_loss = area_loss + rel.mean()
            area_loss = area_loss / max(B, 1)
            area_loss = area_loss.clamp(max=2.0)

            loss = loss_diff + 0.5 * LAMBDA_NOISE * loss_noise + 1.0 * area_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ep{epoch} b{batch_idx} NaN/Inf, skip")
                optim.zero_grad()
                continue

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()

            # EMA
            with torch.no_grad():
                for p_em, p_m in zip(ema.parameters(), model.parameters()):
                    p_em.mul_(0.999).add_(p_m, alpha=0.001)

            total_loss += loss.item()
            total_diff += loss_diff.item()
            total_noise += loss_noise.item()
            n_batches += 1
            if batch_idx % 25 == 0:
                print(f"  ep{epoch} b{batch_idx:4d}  loss={loss.item():.4f}  "
                      f"diff={loss_diff.item():.4f}  noise={loss_noise.item():.4f}")

        print(f"Epoch {epoch}  avg_loss={total_loss/max(n_batches,1):.4f}  "
              f"avg_diff={total_diff/max(n_batches,1):.4f}  avg_noise={total_noise/max(n_batches,1):.4f}")

    save_path = SAVE_DIR / "diffusion_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'norm_stats': {'mu': mu.cpu(), 'sigma': sigma.cpu()},
        'n_steps': N_STEPS,
        'cond_dim': 128,
        'model_kwargs': {'dim': 512, 'depth': 8, 'heads': 8, 'cond_dim': 128, 'n_steps': N_STEPS},
    }, save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
