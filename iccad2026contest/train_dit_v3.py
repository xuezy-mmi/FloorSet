"""
train_dit_v3.py - Train v3 DiT (pre-LN + edge-bias attention).

Loss = vectorized_diff_loss + 0.05 * noise_mse + 1.0 * hard_violation_loss

Trains using x_0 parametrization (one-step x_0 reconstruction from noise
prediction) and EMA(0.999). Saves to
    /home/xzy/eda/model/v3/diffusion_final.pth
with model/EMA weights, mu, sigma, and model_kwargs.
"""
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from iccad2026_evaluate import get_training_dataloader
from dit_model_v3 import DiffusionTransformer
from dit_utils_v3 import (
    CosineSchedule, q_sample_masked,
    compute_norm_stats, vectorized_diff_loss, hard_violation_loss,
)


SAVE_DIR = Path("/home/xzy/eda/model/v3")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_STEPS = 1000
N_EPOCHS = 20
BATCH_SIZE = 8
LR = 2e-4
NUM_SAMPLES = 4000
LAMBDA_NOISE = 0.05
LAMBDA_HARD = 1.0
GRAD_CLIP = 1.0


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

    model = DiffusionTransformer(
        dim=256, depth=6, heads=8, cond_in=8, n_steps=N_STEPS,
    ).to(DEVICE)
    ema = DiffusionTransformer(
        dim=256, depth=6, heads=8, cond_in=8, n_steps=N_STEPS,
    ).to(DEVICE)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = CosineSchedule(N_STEPS)
    alpha_cumprod = sched.alpha_cumprod.to(DEVICE)

    print("Begin training ...")
    t_start = time.time()
    for epoch in range(N_EPOCHS):
        total_loss, total_diff, total_noise, total_hard, n_batches = 0.0, 0.0, 0.0, 0.0, 0
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

            # z-score normalize target
            x0 = ((fp_sol - mu) / sigma) * mask

            t = torch.randint(0, N_STEPS, (B,), device=DEVICE)
            noise = torch.randn_like(x0)
            x_t, _ = q_sample_masked(x0, t, alpha_cumprod, mask, noise)

            pred_noise = model(x_t, t, area_target, b2b, p2b, pins, constraints)

            # noise loss
            loss_noise = F.mse_loss(pred_noise * mask, noise * mask,
                                    reduction='sum') / (mask.sum() + 1e-6)

            # x_0 reconstruction
            a_bar = alpha_cumprod[t].view(-1, 1, 1).clamp(min=1e-6)
            x0_pred = ((x_t - torch.sqrt(1.0 - a_bar) * pred_noise) / torch.sqrt(a_bar)) * mask
            x0_pred = x0_pred.clamp(-5.0, 5.0)

            # unnormalize to real scale (w, h, x, y)
            pos_real = x0_pred * sigma + mu
            pos_real = pos_real.clamp(min=0.0)

            # per-sample diff + hard loss
            loss_diff = torch.zeros((), device=DEVICE)
            loss_hard = torch.zeros((), device=DEVICE)
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
                if constraints is not None:
                    fixed = constraints[i, :n_v, 0] > 0
                    preplaced = constraints[i, :n_v, 1] > 0
                    # Build a target_pos-like tensor for the hard loss
                    tp = fp_sol[i, :n_v].clone()
                    if fixed.any():
                        pos_i[fixed, 2] = fp_sol[i, :n_v][fixed, 0]
                        pos_i[fixed, 3] = fp_sol[i, :n_v][fixed, 1]
                    if preplaced.any():
                        pos_i[preplaced, 0] = fp_sol[i, :n_v][preplaced, 2]
                        pos_i[preplaced, 1] = fp_sol[i, :n_v][preplaced, 3]
                        pos_i[preplaced, 2] = fp_sol[i, :n_v][preplaced, 0]
                        pos_i[preplaced, 3] = fp_sol[i, :n_v][preplaced, 1]

                    # Build target_pos in (x, y, w, h) format for hard loss
                    target_pos = torch.full_like(pos_i, -1.0)
                    if fixed.any():
                        target_pos[fixed, 2] = fp_sol[i, :n_v][fixed, 0]
                        target_pos[fixed, 3] = fp_sol[i, :n_v][fixed, 1]
                    if preplaced.any():
                        target_pos[preplaced, 0] = fp_sol[i, :n_v][preplaced, 2]
                        target_pos[preplaced, 1] = fp_sol[i, :n_v][preplaced, 3]
                        target_pos[preplaced, 2] = fp_sol[i, :n_v][preplaced, 0]
                        target_pos[preplaced, 3] = fp_sol[i, :n_v][preplaced, 1]
                    loss_hard = loss_hard + hard_violation_loss(
                        pos_i, target_pos, constraints[i, :n_v], area_target[i, :n_v]
                    )

                loss_diff = loss_diff + vectorized_diff_loss(
                    pos_i, b2b[i], p2b[i], pins[i],
                    area_target[i, :n_v], metrics[i],
                )
                count += 1
            loss_diff = loss_diff / max(count, 1)
            loss_hard = loss_hard / max(count, 1)

            loss = loss_diff + LAMBDA_NOISE * loss_noise + LAMBDA_HARD * loss_hard

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
                    p_em.mul_(0.999).add_(p_m, alpha=0.001)

            total_loss += loss.item()
            total_diff += loss_diff.item()
            total_noise += loss_noise.item()
            total_hard += loss_hard.item()
            n_batches += 1
            if batch_idx % 25 == 0:
                print(f"  ep{epoch} b{batch_idx:4d}  loss={loss.item():.3f}  "
                      f"diff={loss_diff.item():.3f}  noise={loss_noise.item():.3f}  "
                      f"hard={loss_hard.item():.3f}")

        avg = lambda x: x / max(n_batches, 1)
        print(f"Epoch {epoch}  avg_loss={avg(total_loss):.3f}  "
              f"avg_diff={avg(total_diff):.3f}  avg_noise={avg(total_noise):.3f}  "
              f"avg_hard={avg(total_hard):.3f}  "
              f"elapsed={(time.time()-t_start)/60:.1f}min")

    save_path = SAVE_DIR / "diffusion_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'norm_stats': {'mu': mu.cpu(), 'sigma': sigma.cpu()},
        'n_steps': N_STEPS,
        'cond_in': 8,
        'model_kwargs': {'dim': 256, 'depth': 6, 'heads': 8, 'cond_in': 8, 'n_steps': N_STEPS},
    }, save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
