#!/usr/bin/env python3
"""
train_dit_hd.py - HouseDiffusion-style training of the DiT floorplan model.

Differences from the original `train_dit.py`:
  - 3-branch masked attention backbone (`HouseDiffusionDiT`) instead of
    a plain TransformerEncoder.
  - Per-block 12-feature conditioning (area, b2b, p2b, is_hard,
    boundary, MIB / cluster, valid).
  - Mixed-precision training (autocast + GradScaler).
  - EMA (Exponential Moving Average) of model parameters, saved as
    `diffusion_ema.pt` and used by the optimizer for inference.
  - Mask-aware MSE loss on the predicted noise (HouseDiffusion's
    `mean_flat` with a padding mask).

Differences from the previous (broken) `train_dit_hd.py`:
  - Replaced z-score normalization with the original DiT's positive
    `norm_factor` (`x / 1000.0`). With z-score centering, the reverse
    process produced *negative* x/y positions after denormalization
    (the user's main complaint). With a plain positive divisor, all
    positions are guaranteed non-negative after `clamp(0, +inf)`.
  - Linear beta schedule (matches `dit_utils.DiffusionScheduler`) so
    the model is trained exactly like the working original DiT.

Just edit the ``CONFIG`` block and run:

    python train_dit_hd.py
"""

import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).parent))
from iccad2026_evaluate import get_training_dataloader
from dit_model_hd import HouseDiffusionDiT
from dit_utils_hd import (
    DiffusionScheduler,
    EMAModel,
    masked_mse,
    q_sample_masked,
)


# =============================================================================
# CONFIG  —  all hyperparameters & paths live here. Edit and run.
# =============================================================================
CONFIG = {
    # -------- I/O --------
    "data_path":        "/home/xzy/eda/",                 # FloorSet data root
    "save_dir":         "/home/xzy/eda/model/hd_v1",      # checkpoint output
    "resume":           "",                                # path to resume; "" = from scratch

    # -------- data sampling --------
    "num_train":        10000,                             # subset size; 0 = all 1M
    "batch_size":       32,                                # global batch size
    "microbatch":       -1,                                # -1 ⇒ same as batch_size

    # -------- training schedule --------
    "epochs":           20,
    "lr":               1e-4,
    "weight_decay":     0.0,
    "warmup_steps":     200,
    "use_fp16":         True,                              # mixed precision on CUDA
    "grad_clip":        1.0,

    # -------- diffusion --------
    "n_steps":          1000,                              # DDPM timesteps
    "norm_factor":      1000.0,                            # x / norm_factor for input
    "beta_start":       1e-4,                              # linear schedule
    "beta_end":         0.02,
    "hard_weight":      0.0,                               # 0 disables hard-constraint loss

    # -------- model architecture --------
    "dim":              256,
    "depth":            6,
    "heads":            4,
    "cond_in":          12,                                # per-block conditioning channels
    "max_blocks":       200,                               # max sequence length
    "dropout":          0.1,

    # -------- EMA --------
    "ema_decay":        0.9999,

    # -------- logging / checkpointing --------
    "log_interval":     50,                                # print every N batches
    "save_interval":    1,                                 # save every N epochs
}


# =============================================================================
# LR schedule (cosine with warmup, HouseDiffusion style)
# =============================================================================
def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(progress * math.pi))
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Training
# =============================================================================
def main(cfg=CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints to: {save_dir}")

    # --- data ---
    train_loader = get_training_dataloader(
        data_path=cfg["data_path"],
        batch_size=cfg["batch_size"],
        num_samples=cfg["num_train"] if cfg["num_train"] > 0 else None,
        shuffle=True,
    )
    print(f"Loaded {len(train_loader)} batches")

    # --- schedule ---
    schedule = DiffusionScheduler(
        n_steps=cfg["n_steps"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
    ).to(device)
    print(f"Schedule: linear [{cfg['beta_start']}, {cfg['beta_end']}], "
          f"n_steps={cfg['n_steps']}, norm_factor={cfg['norm_factor']}")

    # --- model ---
    model = HouseDiffusionDiT(
        dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"],
        cond_in=cfg["cond_in"], n_steps=cfg["n_steps"],
        max_blocks=cfg["max_blocks"], dropout=cfg["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    if cfg["resume"]:
        ck = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ck)
        print(f"Resumed model from {cfg['resume']}")

    # --- EMA ---
    ema = EMAModel(model, decay=cfg["ema_decay"])

    # --- optim ---
    optimizer = AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    total_steps = max(1, cfg["epochs"] * len(train_loader))
    scheduler = cosine_with_warmup(
        optimizer, warmup_steps=cfg["warmup_steps"], total_steps=total_steps,
    )
    scaler = GradScaler("cuda", enabled=cfg["use_fp16"])

    # --- training loop ---
    microbatch = cfg["microbatch"] if cfg["microbatch"] > 0 else cfg["batch_size"]
    norm_factor = cfg["norm_factor"]
    log = []
    for epoch in range(cfg["epochs"]):
        model.train()
        ep_loss, ep_hard, n_steps_done = 0.0, 0.0, 0
        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, fp_sol, metrics = batch
            area_target = area_target.to(device)
            b2b_conn = b2b_conn.to(device)
            p2b_conn = p2b_conn.to(device)
            pins_pos = pins_pos.to(device)
            constraints = constraints.to(device)
            fp_sol = fp_sol.to(device)

            B, N, _ = fp_sol.shape
            valid = (area_target != -1).float().unsqueeze(-1)  # [B, N, 1]
            mask = valid

            # normalize: positive divisor (no centering)
            x0 = (fp_sol / norm_factor) * mask

            t = torch.randint(0, cfg["n_steps"], (B,), device=device)
            x_t, noise = q_sample_masked(
                x0, t, schedule.alpha_cumprod, mask,
            )

            # micro-batch
            loss_total = 0.0
            hard_total = 0.0
            n_micro = 0
            for i in range(0, B, microbatch):
                end = min(i + microbatch, B)
                mslice = mask[i:end]
                with autocast("cuda", enabled=cfg["use_fp16"]):
                    pred_noise = model(
                        x_t[i:end], t[i:end], area_target[i:end],
                        b2b_conn[i:end], p2b_conn[i:end],
                        pins_pos[i:end], constraints[i:end],
                    )
                    loss = masked_mse(pred_noise, noise[i:end], mslice)

                    # Optional hard-constraint loss (area-tolerance penalty
                    # on the predicted x0 for free blocks).
                    if cfg["hard_weight"] > 0.0:
                        a_t = schedule.alpha_cumprod[t[i:end]].view(-1, 1, 1)
                        x0_pred = (
                            (x_t[i:end] - torch.sqrt(1.0 - a_t) * pred_noise)
                            / torch.sqrt(a_t.clamp(min=1e-8))
                        ) * mslice
                        # un-normalize
                        x0_unnorm = x0_pred * norm_factor

                        soft_mask = (constraints[i:end, :, 0] == 0) & (
                            constraints[i:end, :, 1] == 0
                        ) & (area_target[i:end] > 0)
                        if soft_mask.any():
                            block_w = x0_unnorm[..., 0]
                            block_h = x0_unnorm[..., 1]
                            actual = block_w * block_h
                            tgt = area_target[i:end]
                            rel = (actual - tgt).abs() / (tgt + 1e-6)
                            hard = torch.relu(rel - 0.01).mean()
                            hard_total = hard_total + hard.item()
                        else:
                            hard = torch.tensor(0.0, device=device)
                        loss = loss + cfg["hard_weight"] * hard
                loss_total = loss_total + loss.item()
                n_micro += 1

                scaler.scale(loss / max(1, B // microbatch)).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            ema.update(model)

            ep_loss += loss_total / max(1, n_micro)
            ep_hard += hard_total / max(1, n_micro)
            n_steps_done += 1

            if batch_idx % cfg["log_interval"] == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"ep {epoch} step {batch_idx}/{len(train_loader)} "
                    f"loss={loss_total / max(1, n_micro):.4f} "
                    f"hard={hard_total / max(1, n_micro):.4f} "
                    f"lr={lr_now:.2e} elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )
                if batch_idx == 0 and epoch == 0:
                    print(
                        f"  x0: min={x0.min().item():.4f}, max={x0.max().item():.4f}",
                        flush=True,
                    )
                    print(
                        f"  pred_noise: min={pred_noise.min().item():.4f}, max={pred_noise.max().item():.4f}",
                        flush=True,
                    )

        avg = ep_loss / max(1, n_steps_done)
        avg_hard = ep_hard / max(1, n_steps_done)
        print(
            f"=== epoch {epoch} done | avg loss={avg:.4f} "
            f"avg hard={avg_hard:.4f} | {time.time()-t0:.1f}s ===",
            flush=True,
        )
        log.append({"epoch": epoch, "loss": avg, "hard": avg_hard})

        # save checkpoint (model + EMA) at the end of each save_interval
        if (epoch + 1) % cfg["save_interval"] == 0 or epoch == cfg["epochs"] - 1:
            torch.save(model.state_dict(), save_dir / f"diffusion_ep{epoch}.pt")
            torch.save(ema.state_dict(), save_dir / f"diffusion_ema_ep{epoch}.pt")
            torch.save(model.state_dict(), save_dir / "diffusion_final.pt")
            torch.save(ema.state_dict(), save_dir / "diffusion_ema_final.pt")
            with open(save_dir / "train_log.json", "w") as f:
                json.dump(log, f, indent=2)

    # also save the schedule meta so the optimizer can reload exactly
    meta = {
        "n_steps":     cfg["n_steps"],
        "norm_factor": cfg["norm_factor"],
        "beta_start":  cfg["beta_start"],
        "beta_end":    cfg["beta_end"],
        "dim":         cfg["dim"],
        "depth":       cfg["depth"],
        "heads":       cfg["heads"],
        "cond_in":     cfg["cond_in"],
        "max_blocks":  cfg["max_blocks"],
    }
    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Training done. Final checkpoint + EMA at {save_dir}/", flush=True)


if __name__ == "__main__":
    main()
