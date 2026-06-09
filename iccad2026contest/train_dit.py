#!/usr/bin/env python3
"""
train_diffusion.py - Train Diffusion Transformer for Floorplanning
Usage: python train_diffusion.py
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent))
from iccad2026_evaluate import (
    get_training_dataloader,
    compute_training_loss_differentiable,
)
from dit_model import DiffusionTransformer
from dit_utils import DiffusionScheduler, q_sample

def main():

    batch_size = 64
    n_steps = 1000          # diffusion steps
    learning_rate = 1e-5
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device is", "GPU" if torch.cuda.is_available() else "CPU")

    train_loader = get_training_dataloader(
        data_path="/home/xzy/eda/",
        batch_size=batch_size,
        num_samples=10000,
        shuffle=True
    )

    print("Get DataSet Successfully.")
    

    model = DiffusionTransformer(
        dim=512,
        depth=8,
        heads=8,
        cond_dim=128,
        n_steps=n_steps
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    print("Build DiT Arch Successfully.")
    
    scheduler = DiffusionScheduler(n_steps, beta_start=1e-4, beta_end=0.02)
    alpha_cumprod = scheduler.alpha_cumprod.to(device)
    print("Build DiT-Scheduler Successfully.")
    print("Begin Training ...")

    # training loop
    # save_dir = Path("model")
    # save_dir.mkdir(exist_ok=True)
    save_dir = Path("/home/xzy/eda/model/v1")
    save_dir.mkdir(parents=True, exist_ok=True)
    norm_factor = 1000.0
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, fp_sol, metrics = batch
            # to device
            area_target = area_target.to(device)
            b2b_conn = b2b_conn.to(device)
            p2b_conn = p2b_conn.to(device)
            pins_pos = pins_pos.to(device)
            constraints = constraints.to(device)
            fp_sol = fp_sol.to(device)
            metrics = metrics.to(device)

            B, max_blocks, _ = fp_sol.shape

            valid_mask = (area_target != -1)
            mask = valid_mask.unsqueeze(-1).expand_as(fp_sol)  # [B, max_blocks, 4]

            x0 = fp_sol / norm_factor
            x0 = x0 * mask.float()

            t = torch.randint(0, n_steps, (B,), device=device)
            noise = torch.randn_like(x0)
            x_t, noise = q_sample(x0, t, alpha_cumprod, noise)

            pred_noise = model(x_t, t, area_target, b2b_conn, p2b_conn, pins_pos, constraints)

            # MSE loss only on valid positions
            loss_mse = nn.functional.mse_loss(pred_noise * mask, noise * mask, reduction='sum')
            loss_mse = loss_mse / (mask.sum() + 1e-6)
            loss = loss_mse

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}: loss={loss.item():.4f}")
            # ============================ test info ============================
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss at epoch {epoch} batch {batch_idx}, skipping")
                optimizer.zero_grad()
                continue

            if batch_idx == 0 and epoch == 0:
                print(f"x0: min={x0.min().item():.4f}, max={x0.max().item():.4f}, mean={x0.mean().item():.4f}")
                print(f"noise: min={noise.min().item():.4f}, max={noise.max().item():.4f}")
                print(f"x_t: min={x_t.min().item():.4f}, max={x_t.max().item():.4f}")
                print(f"pred_noise: min={pred_noise.min().item():.4f}, max={pred_noise.max().item():.4f}")
                print(f"loss_mse: {loss_mse.item():.6f}")
            # ============================ test info ============================

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} finished, avg loss={avg_loss:.4f}")
        # torch.save(model.state_dict(), f"model/diffusion_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_dir / f"diffusion_epoch_{epoch}.pth")

    # torch.save(model.state_dict(), "model/diffusion_final.pth")
    torch.save(model.state_dict(), save_dir / "diffusion_final.pth")
    print("Training done!")

if __name__ == "__main__":

    model_dir = Path("/home/xzy/eda/model/v1")
    if model_dir.exists():
        for file in model_dir.iterdir():
            if file.is_file():
                file.unlink()
        print(f"Cleaned old model files from {model_dir}")
    else:
        model_dir.mkdir(parents=True)
        print(f"Created {model_dir} directory")

    main()