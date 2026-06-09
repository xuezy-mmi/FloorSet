#!/usr/bin/env python3
"""
train_dit_v2_regression.py - Regression model trained with diff loss

Direct regression Transformer. Input: graph-conditioned features.
Output: (x, y, w, h) in z-score space. Loss = vectorized diff cost +
auxiliary MSE on (x, y, w, h) to keep the model grounded.

Saves to /home/xzy/eda/model/v2_regression/diffusion_final.pth.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from iccad2026_evaluate import get_training_dataloader
from dit_optimizer_v2_regression import RegModel, aggregate_graph_features
from train_dit_v2 import vectorized_diff_loss, compute_norm_stats


SAVE_DIR = Path("/home/xzy/eda/model/v2_regression")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-4
NUM_SAMPLES = 4000
GRAD_CLIP = 1.0
LAMBDA_MSE = 0.1  # small auxiliary MSE on (x, y, w, h)
LAMBDA_AREA = 0.5  # explicit area loss


def main():
    print(f"Device: {DEVICE}")
    train_loader = get_training_dataloader(
        data_path="/home/xzy/eda/",
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES,
        shuffle=True,
    )
    print(f"  train batches/epoch: {len(train_loader)}")

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

    model = RegModel(dim=256, depth=6, heads=8, cond_in=8).to(DEVICE)
    ema = RegModel(dim=256, depth=6, heads=8, cond_in=8).to(DEVICE)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=N_EPOCHS)

    print("Begin training ...")
    for epoch in range(N_EPOCHS):
        total_loss, total_diff, total_area, total_mse, n_batches = 0.0, 0.0, 0.0, 0.0, 0
        for batch_idx, batch in enumerate(train_loader):
            area_target, b2b, p2b, pins, constraints, tree_sol, fp_sol, metrics = batch
            area_target = area_target.to(DEVICE)
            b2b = b2b.to(DEVICE)
            p2b = p2b.to(DEVICE)
            pins = pins.to(DEVICE)
            constraints = constraints.to(DEVICE)
            fp_sol = fp_sol.to(DEVICE)
            metrics = metrics.to(DEVICE)

            B = area_target.shape[0]
            mask = (area_target != -1).float()  # [B, N]

            # normalize target to z-score
            x0 = ((fp_sol - mu) / sigma)  # [B, N, 4] = (w, h, x, y)
            x0 = x0 * mask.unsqueeze(-1)

            pred = model(area_target, b2b, p2b, pins, constraints)  # [B, N, 4]
            pred = pred * mask.unsqueeze(-1)

            # direct MSE on z-score (x, y, w, h)
            loss_mse = F.mse_loss(pred, x0, reduction='sum') / (mask.sum() * 4 + 1e-6)

            # unnormalize pred to real scale and re-order to (x, y, w, h)
            pos_real = pred * sigma + mu  # [B, N, 4] = (w, h, x, y)
            pos_real = pos_real.clamp(min=0.0)

            loss_diff = torch.zeros((), device=DEVICE)
            area_loss = torch.zeros((), device=DEVICE)
            count = 0
            for i in range(B):
                valid_i = area_target[i] != -1
                n_v = int(valid_i.sum().item())
                if n_v < 2:
                    continue
                pos_i = torch.stack([
                    pos_real[i, :n_v, 2],  # x
                    pos_real[i, :n_v, 3],  # y
                    pos_real[i, :n_v, 0],  # w
                    pos_real[i, :n_v, 1],  # h
                ], dim=-1)
                if constraints is not None:
                    fixed = constraints[i, :n_v, 0] > 0
                    preplaced = constraints[i, :n_v, 1] > 0
                    if fixed.any():
                        pos_i[fixed, 2] = fp_sol[i, :n_v][fixed, 0]
                        pos_i[fixed, 3] = fp_sol[i, :n_v][fixed, 1]
                    if preplaced.any():
                        pos_i[preplaced, 0] = fp_sol[i, :n_v][preplaced, 2]
                        pos_i[preplaced, 1] = fp_sol[i, :n_v][preplaced, 3]
                        pos_i[preplaced, 2] = fp_sol[i, :n_v][preplaced, 0]
                        pos_i[preplaced, 3] = fp_sol[i, :n_v][preplaced, 1]
                # explicit area
                av = area_target[i, :n_v]
                m = av > 0
                if m.any():
                    rel = (pos_i[m, 2] * pos_i[m, 3] - av[m]).abs() / (av[m] + 1e-6)
                    area_loss = area_loss + rel.mean()
                loss_diff = loss_diff + vectorized_diff_loss(
                    pos_i, b2b[i], p2b[i], pins[i],
                    area_target[i, :n_v], metrics[i],
                )
                count += 1
            loss_diff = loss_diff / max(count, 1)
            area_loss = (area_loss / max(count, 1)).clamp(max=2.0)

            loss = loss_diff + LAMBDA_MSE * loss_mse + LAMBDA_AREA * area_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ep{epoch} b{batch_idx} NaN/Inf, skip")
                optim.zero_grad()
                continue

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()

            with torch.no_grad():
                for p_em, p_m in zip(ema.parameters(), model.parameters()):
                    p_em.mul_(0.995).add_(p_m, alpha=0.005)

            total_loss += loss.item()
            total_diff += loss_diff.item()
            total_area += area_loss.item()
            total_mse += loss_mse.item()
            n_batches += 1
            if batch_idx % 25 == 0:
                print(f"  ep{epoch} b{batch_idx:4d}  loss={loss.item():.3f}  "
                      f"diff={loss_diff.item():.3f}  area={area_loss.item():.3f}  mse={loss_mse.item():.3f}")

        sched.step()
        print(f"Epoch {epoch}  avg_loss={total_loss/max(n_batches,1):.3f}  "
              f"avg_diff={total_diff/max(n_batches,1):.3f}  avg_area={total_area/max(n_batches,1):.3f}  "
              f"avg_mse={total_mse/max(n_batches,1):.3f}")

    save_path = SAVE_DIR / "diffusion_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'norm_stats': {'mu': mu.cpu(), 'sigma': sigma.cpu()},
        'model_kwargs': {'dim': 256, 'depth': 6, 'heads': 8, 'cond_in': 8},
    }, save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
