# train_dit.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from dit_model import ConditionEncoder, DiffusionTransformer
from diffusion import GaussianDiffusion
from iccad2026_evaluate import get_training_dataloader, compute_training_loss_differentiable

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of training samples (None = all 1M)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # DataLoader
    dataloader = get_training_dataloader(
        data_path="../",  # adjust to your data path
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        shuffle=True
    )
    print(f"Training samples: {len(dataloader.dataset)}")

    # Models
    cond_encoder = ConditionEncoder(feat_dim=64, hidden_dim=256).to(device)
    diffusion_model = DiffusionTransformer(dim=256, depth=12, num_heads=8, cond_dim=256).to(device)
    diffusion = GaussianDiffusion(num_timesteps=1000, schedule='cosine').to(device)

    # Optimizer
    optimizer = optim.AdamW(
        list(cond_encoder.parameters()) + list(diffusion_model.parameters()),
        lr=args.lr, weight_decay=0.01
    )

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            # Unpack batch (8 tensors)
            area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, fp_sol, metrics = batch
            # Move to device
            area_target = area_target.to(device)
            b2b_conn = b2b_conn.to(device)
            p2b_conn = p2b_conn.to(device)
            pins_pos = pins_pos.to(device)
            constraints = constraints.to(device)
            metrics = metrics.to(device)
            fp_sol = fp_sol.to(device)   # [bsz, max_blocks, 4] (w,h,x,y)

            # Ground truth layout in [x,y,w,h] order
            # fp_sol: [bsz, max_blocks, 4] = (w, h, x, y)
            B, K, _ = fp_sol.shape
            gt_layout = torch.stack([fp_sol[...,2], fp_sol[...,3], fp_sol[...,0], fp_sol[...,1]], dim=-1)  # [B,K,4] (x,y,w,h)

            # Condition encoding
            cond, mask = cond_encoder(area_target, constraints, b2b_conn, target_positions=None)

            # Sample random timestep
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
            # Add noise to gt_layout
            x_t, noise = diffusion.q_sample(gt_layout, t)

            # Predict noise
            noise_pred = diffusion_model(x_t, t, cond, mask)

            # Diffusion loss (MSE on valid blocks)
            loss = F.mse_loss(noise_pred[mask], noise[mask])

            # Optional: Add differentiable contest loss as auxiliary
            # This can be computed using the denoised prediction at t=0 (approximated)
            # For simplicity we skip it here; you can add a weighted term.

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(cond_encoder.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            global_step += 1

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}")

        # Save checkpoint
        if (epoch+1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'cond_encoder': cond_encoder.state_dict(),
                'diffusion_model': diffusion_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pt'))

    # Save final model
    torch.save({
        'cond_encoder': cond_encoder.state_dict(),
        'diffusion_model': diffusion_model.state_dict(),
    }, os.path.join(args.save_dir, 'final_model.pt'))
    print("Training finished.")

if __name__ == '__main__':
    train()