#!/usr/bin/env python3
"""
train_dit.py - Train DiT for floorplan optimization.

Supports training on: lite, prime, or combined (lite + prime) datasets.

Loss = λ_hard * hard_violation_loss      # 优先级1: 硬约束（可行性）
   + λ_mse  * loss_mse_x0              # 优先级2: 直接监督GT布局
   + λ_diff * vectorized_diff_loss       # 优先级3: contest评分函数
   + λ_noise * noise_mse                # 优先级4: 扩散动态

Where:
  - hard_violation_loss: SSE on fixed/preplaced violations + area tolerance.
    PRIMARY loss for ensuring feasibility. = 0 when all hard constraints satisfied.
  - loss_mse_x0: MSE on predicted x̂₀ vs ground truth x₀ (z-score space).
    Directly supervises layout quality, independent of diffusion dynamics.
  - vectorized_diff_loss: contest cost formula (differentiable surrogate).
    HPWL_gap + Area_gap + exp(2×V_soft).
  - noise_mse: standard DDPM noise prediction loss.
  2. Soft constraints (wirelength, bbox area, grouping, boundary) are secondary
     and handled by vectorized_diff_loss.
  3. No post-processing is used — the model should learn to generate feasible
     layouts directly from the diffusion process.

Usage:
  python train_dit.py --dataset lite
  python train_dit.py --dataset prime
  python train_dit.py --dataset combined
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from iccad2026_evaluate import get_training_dataloader, FloorplanDatasetLite
from lite_dataset import floorplan_collate as lite_collate
from dit_model import DiffusionTransformer
from dit_utils import (
    CosineSchedule, q_sample_masked,
    compute_norm_stats, vectorized_diff_loss, hard_violation_loss,
)


# ---------------------------------------------------------------------------
# Loss weights (优先级从高到低)
# ---------------------------------------------------------------------------
LAMBDA_HARD = 10.0   # weight on hard constraint loss (FIXED/PREPLACED/area tolerance)
                       # PRIMARY: = 0 when all hard constraints satisfied, > 0 when violated.
LAMBDA_MSE = 0.5    # weight on direct MSE supervision: E[|x̂₀ - x₀|²]
                       # Directly supervises layout quality, accelerates convergence.
LAMBDA_DIFF = 1.0   # weight on vectorized_diff_loss (contest cost formula)
LAMBDA_NOISE = 0.05  # weight on noise MSE (standard DDPM loss)

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
SAVE_DIR = Path("model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_STEPS = 1000
N_EPOCHS = 20
BATCH_SIZE = 64          # 64是H100的安全起点，可逐步增至128/256
NUM_SAMPLES = 100000   # per dataset (for lite); prime uses all available
LR = 2e-4
GRAD_CLIP = 1.0
EMA_DECAY = 0.999

# ---------------------------------------------------------------------------
# GPU 利用率优化配置
# ---------------------------------------------------------------------------
NUM_WORKERS = 8          # DataLoader 并行读取（建议 8~16）
PIN_MEMORY = True        # 锁页内存，加速 CPU→GPU 传输
CUDA_GRAPH = True        # CUDA Graph 减少 kernel launch 开销
USE_AMP = True           # 混合精度 FP16（H100 原生支持 BF16）
AMP_DTYPE = torch.bfloat16  # H100 上 BF16 更高效；如遇问题改为 torch.float16
GRAD_ACCUM_STEPS = 1     # 显存不够时设为 2~4，相当于 batch_size × N


# ---------------------------------------------------------------------------
# Prime dataset collate (converts polygons -> bbox for fp_sol)
# ---------------------------------------------------------------------------
def prime_collate_fn(batch):
    """
    Collate for Prime dataset.
    Converts polygon ground-truth to (w, h, x, y) bounding boxes.
    """
    area_target = [item['input'][0] for item in batch]
    b2b_conn = [item['input'][1] for item in batch]
    p2b_conn = [item['input'][2] for item in batch]
    pins_pos = [item['input'][3] for item in batch]
    placement_constraints = [item['input'][4] for item in batch]

    # fp_sol for Prime is a list of polygons, each polygon is [V, 2]
    # Convert to (w, h, x, y) bounding boxes
    fp_sol_raw = [item['label'][0] for item in batch]
    fp_sol_bbox = []
    for polygons in fp_sol_raw:
        # polygons is a list of [V, 2] tensors, one per block
        bboxes = []
        for poly in polygons:
            if isinstance(poly, torch.Tensor):
                x_min = poly[:, 0].min().item()
                x_max = poly[:, 0].max().item()
                y_min = poly[:, 1].min().item()
                y_max = poly[:, 1].max().item()
            else:
                # numpy array
                x_min = poly[:, 0].min()
                x_max = poly[:, 0].max()
                y_min = poly[:, 1].min()
                y_max = poly[:, 1].max()
            w = max(x_max - x_min, 1e-6)
            h = max(y_max - y_min, 1e-6)
            bboxes.append([w, h, x_min, y_min])
        fp_sol_bbox.append(bboxes)

    metrics_sol = [item['label'][1] for item in batch]

    # Pad everything to variable lengths
    def pad_to_largest(tens_list):
        ndims = tens_list[0].ndim
        max_dims = [max(x.size(dim) for x in tens_list)
                    for dim in range(ndims)]
        padded = []
        for tens in tens_list:
            pad_tuple = tuple(x for d in range(ndims)
                             for x in (max_dims[d] - tens.size(d), 0))
            if tens.dtype == torch.bool:
                pad_val = False
            else:
                pad_val = -1.0
            padded.append(F.pad(tens, pad_tuple[::-1], mode='constant', value=pad_val))
        return torch.stack(padded)

    def pad_connectivity(conn_list):
        max_edges = max(c.size(0) for c in conn_list)
        padded = []
        for c in conn_list:
            if c.size(0) < max_edges:
                padded.append(F.pad(c, (0, 0, 0, max_edges - c.size(0)),
                                   mode='constant', value=-1.0))
            else:
                padded.append(c)
        return torch.stack(padded)

    def pad_pins(pins_list):
        max_pins = max(p.size(0) for p in pins_list)
        padded = []
        for p in pins_list:
            if p.size(0) < max_pins:
                padded.append(F.pad(p, (0, 0, 0, max_pins - p.size(0)),
                                   mode='constant', value=0.0))
            else:
                padded.append(p)
        return torch.stack(padded)

    # fp_sol_bbox: list of lists of [w,h,x,y] -> need special padding
    def pad_fp_sol(fp_list):
        """Pad list of [w,h,x,y] lists to same number of blocks."""
        max_blocks = max(len(fp) for fp in fp_list)
        result = []
        for fp in fp_list:
            if len(fp) < max_blocks:
                pad = [[-1.0, -1.0, -1.0, -1.0]] * (max_blocks - len(fp))
                fp = fp + pad
            result.append(torch.tensor(fp, dtype=torch.float32))
        return torch.stack(result)

    # tree_sol: Prime doesn't have it; use zeros
    def pad_tree_sol(sol_list):
        max_nodes = max(s.size(0) for s in sol_list) if sol_list else 1
        padded = []
        for s in sol_list:
            if s.size(0) < max_nodes:
                padded.append(F.pad(s, (0, 0, 0, max_nodes - s.size(0)),
                                   mode='constant', value=-1.0))
            else:
                padded.append(s)
        return torch.stack(padded)

    area_target = pad_to_largest(area_target)
    b2b_conn = pad_connectivity(b2b_conn)
    p2b_conn = pad_connectivity(p2b_conn)
    pins_pos = pad_pins(pins_pos)
    placement_constraints = pad_to_largest(placement_constraints)
    fp_sol = pad_fp_sol(fp_sol_bbox)
    tree_sol = pad_tree_sol([torch.zeros(1, 3) for _ in fp_sol_raw])
    metrics_sol = torch.stack(metrics_sol)

    return (area_target, b2b_conn, p2b_conn,
            pins_pos, placement_constraints, tree_sol, fp_sol, metrics_sol)


# ---------------------------------------------------------------------------
# Prime dataset loader
# ---------------------------------------------------------------------------
def get_prime_dataloader(data_path, batch_size, num_samples=None, shuffle=True):
    """Get DataLoader for Prime dataset."""
    sys.path.insert(0, str(Path(data_path).parent))
    from prime_dataset import FloorplanDatasetPrime
    dataset = FloorplanDatasetPrime(data_path)
    if num_samples is not None:
        indices = list(range(min(num_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=prime_collate_fn,
    )


# ---------------------------------------------------------------------------
# Combined dataloader (interleaves lite + prime)
# ---------------------------------------------------------------------------
def get_combined_dataloader(data_path, batch_size, num_samples=None, shuffle=True):
    """Get a combined DataLoader that interleaves lite and prime batches."""
    lite_loader = get_training_dataloader(
        data_path=data_path, batch_size=batch_size,
        num_samples=num_samples, shuffle=shuffle,
    )
    prime_loader = get_prime_dataloader(
        data_path=data_path, batch_size=batch_size,
        num_samples=num_samples, shuffle=shuffle,
    )
    return lite_loader, prime_loader


# ---------------------------------------------------------------------------
# Compute statistics for z-score normalization
# ---------------------------------------------------------------------------
def compute_dataset_stats(data_path, dataset_type, max_batches=64):
    """Compute mu/sigma from a dataset for z-score normalization."""
    if dataset_type == 'lite':
        loader = get_training_dataloader(data_path, batch_size=8,
                                         num_samples=512, shuffle=False)
    elif dataset_type == 'prime':
        loader = get_prime_dataloader(data_path, batch_size=8,
                                      num_samples=512, shuffle=False)
    else:  # combined — use lite as representative
        loader = get_training_dataloader(data_path, batch_size=8,
                                         num_samples=512, shuffle=False)
    return compute_norm_stats(loader, max_batches=max_batches)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------
def train_step(model, ema, optim, batch, alpha_cumprod, mu, sigma, device):
    """Run one training step. Returns loss components."""
    (area_target, b2b, p2b, pins, constraints,
     tree_sol, fp_sol, metrics) = batch
    area_target = area_target.to(device)
    b2b = b2b.to(device)
    p2b = p2b.to(device)
    pins = pins.to(device)
    constraints = constraints.to(device)
    fp_sol = fp_sol.to(device)
    metrics = metrics.to(device)

    B, N_max, _ = fp_sol.shape
    mask = (area_target != -1).unsqueeze(-1).expand_as(fp_sol).float()

    # z-score normalize ground truth
    x0 = ((fp_sol - mu) / sigma) * mask

    t = torch.randint(0, len(alpha_cumprod), (B,), device=device)
    noise = torch.randn_like(x0)
    x_t, _ = q_sample_masked(x0, t, alpha_cumprod, mask, noise)

    pred_noise = model(x_t, t, area_target, b2b, p2b, pins, constraints)

    # 1. Noise MSE loss
    loss_noise = F.mse_loss(pred_noise * mask, noise * mask,
                             reduction='sum') / (mask.sum() + 1e-6)

    # 2. x_0 reconstruction (z-score normalized space)
    a_bar = alpha_cumprod[t].view(-1, 1, 1).clamp(min=1e-6)
    x0_pred = ((x_t - torch.sqrt(1.0 - a_bar) * pred_noise) / torch.sqrt(a_bar)) * mask
    x0_pred = x0_pred.clamp(-5.0, 5.0)

    # 2b. Direct MSE supervision on x₀ (z-score space) vs ground truth
    # This directly penalizes deviation from correct layout, independent of diffusion dynamics
    loss_mse_x0 = F.mse_loss(x0_pred * mask, x0 * mask,
                               reduction='sum') / (mask.sum() + 1e-6)

    # unnormalize to real scale (w, h, x, y) for per-sample losses
    pos_real = x0_pred * sigma + mu
    pos_real = pos_real.clamp(min=0.0)

    # 3. Per-sample diff_loss + hard_loss
    loss_diff = torch.zeros((), device=device)
    loss_hard = torch.zeros((), device=device)
    count = 0
    for i in range(B):
        valid_i = area_target[i] != -1
        n_v = int(valid_i.sum().item())
        if n_v < 2:
            continue

        # Reorder fp_sol[i] from (w,h,x,y) to (x,y,w,h) for loss computation
        # fp_sol[i] shape: [N_max, 4] with -1 padding
        # positions need to be: [x, y, w, h]
        fp_i = fp_sol[i, :n_v]  # [n_v, 4] = (w, h, x, y)
        pos_i = torch.stack([
            fp_i[:, 2],  # x
            fp_i[:, 3],  # y
            fp_i[:, 0],  # w
            fp_i[:, 1],  # h
        ], dim=-1)  # [n_v, 4] = (x, y, w, h)

        # Build target_pos in (x, y, w, h) format for hard loss
        # Start from ground truth, override with fixed/preplaced targets
        target_pos = pos_i.clone()

        constr_i = constraints[i, :n_v]
        fixed = constr_i[:, 0] > 0
        preplaced = constr_i[:, 1] > 0

        # target_positions for fixed/preplaced come from fp_sol ground truth
        # (fp_sol IS the ground truth, so for fixed/preplaced, target == ground truth)
        if fixed.any() or preplaced.any():
            target_pos[fixed, 2] = fp_i[fixed, 0]   # w from ground truth
            target_pos[fixed, 3] = fp_i[fixed, 1]   # h from ground truth
            target_pos[preplaced, 0] = fp_i[preplaced, 2]  # x from ground truth
            target_pos[preplaced, 1] = fp_i[preplaced, 3]  # y from ground truth
            target_pos[preplaced, 2] = fp_i[preplaced, 0]  # w from ground truth
            target_pos[preplaced, 3] = fp_i[preplaced, 1]  # h from ground truth

        loss_hard = loss_hard + hard_violation_loss(
            pos_i, target_pos, constr_i, area_target[i, :n_v]
        )
        loss_diff = loss_diff + vectorized_diff_loss(
            pos_i, b2b[i], p2b[i], pins[i],
            area_target[i, :n_v], metrics[i],
        )
        count += 1

    loss_diff = loss_diff / max(count, 1)
    loss_hard = loss_hard / max(count, 1)

    # Combined loss: hard loss is PRIMARY (ensures feasibility)
    # loss = λ_hard×hard + λ_mse×mse_x0 + λ_diff×diff + λ_noise×noise
    loss = (LAMBDA_HARD * loss_hard
            + LAMBDA_MSE * loss_mse_x0
            + LAMBDA_DIFF * loss_diff
            + LAMBDA_NOISE * loss_noise)

    return loss, loss_diff, loss_mse_x0, loss_noise, loss_hard


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='lite',
                        choices=['lite', 'prime', 'combined'],
                        help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES)
    parser.add_argument('--data_path', type=str, default='/home/xzy/eda/')
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Loss weights: LAMBDA_HARD={LAMBDA_HARD}, LAMBDA_MSE={LAMBDA_MSE}, "
          f"LAMBDA_DIFF={LAMBDA_DIFF}, LAMBDA_NOISE={LAMBDA_NOISE}")

    # Load dataloaders
    print("Loading data ...")
    if args.dataset == 'lite':
        train_loader = get_training_dataloader(
            data_path=args.data_path, batch_size=args.batch_size,
            num_samples=args.num_samples, shuffle=True,
        )
        stat_loader = get_training_dataloader(
            data_path=args.data_path, batch_size=8,
            num_samples=512, shuffle=False,
        )
    elif args.dataset == 'prime':
        train_loader = get_prime_dataloader(
            data_path=args.data_path, batch_size=args.batch_size,
            num_samples=args.num_samples, shuffle=True,
        )
        stat_loader = get_prime_dataloader(
            data_path=args.data_path, batch_size=8,
            num_samples=512, shuffle=False,
        )
    else:  # combined
        lite_loader, prime_loader = get_combined_dataloader(
            data_path=args.data_path, batch_size=args.batch_size,
            num_samples=args.num_samples, shuffle=True,
        )
        # Use lite for stats computation (representative of both)
        stat_loader = get_training_dataloader(
            data_path=args.data_path, batch_size=8,
            num_samples=512, shuffle=False,
        )

    print(f"  train batches/epoch: {len(train_loader)}")

    # Compute z-score stats
    print("Computing z-score stats ...")
    mu, sigma = compute_norm_stats(stat_loader, max_batches=64)
    print(f"  mu    = {mu.tolist()}")
    print(f"  sigma = {sigma.tolist()}")
    mu, sigma = mu.to(DEVICE), sigma.to(DEVICE)

    # Build model + EMA
    model = DiffusionTransformer(
        dim=256, depth=6, heads=8, cond_in=8, n_steps=N_STEPS,
    ).to(DEVICE)
    ema = DiffusionTransformer(
        dim=256, depth=6, heads=8, cond_in=8, n_steps=N_STEPS,
    ).to(DEVICE)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = CosineSchedule(N_STEPS)
    alpha_cumprod = sched.alpha_cumprod.to(DEVICE)

    print("Build DiT arch + EMA + CosineSchedule successfully.")
    print("Begin training ...")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    for epoch in range(args.epochs):
        total_loss, total_diff, total_mse, total_noise, total_hard = 0.0, 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        # Handle combined dataset (interleave lite + prime)
        if args.dataset == 'combined':
            lite_iter = iter(lite_loader)
            prime_iter = iter(prime_loader)
            max_iter = max(len(lite_loader), len(prime_loader))
            batch_counter = 0
            for _ in range(max_iter):
                # Alternate lite / prime batches
                try:
                    batch = next(lite_iter)
                    loss, ld, lm, ln, lh = train_step(
                        model, ema, optim, batch, alpha_cumprod, mu, sigma, DEVICE)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_loss += loss.item()
                        total_diff += ld.item()
                        total_mse += lm.item()
                        total_noise += ln.item()
                        total_hard += lh.item()
                        n_batches += 1
                    batch_counter += 1
                    if batch_counter % 25 == 0:
                        print(f"  ep{epoch} b{batch_counter:4d}  "
                              f"loss={loss.item():.3f}  "
                              f"diff={ld:.3f}  mse={lm:.3f}  noise={ln:.3f}  hard={lh:.3f}")
                except StopIteration:
                    pass
                try:
                    batch = next(prime_iter)
                    loss, ld, lm, ln, lh = train_step(
                        model, ema, optim, batch, alpha_cumprod, mu, sigma, DEVICE)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_loss += loss.item()
                        total_diff += ld.item()
                        total_mse += lm.item()
                        total_noise += ln.item()
                        total_hard += lh.item()
                        n_batches += 1
                    batch_counter += 1
                    if batch_counter % 25 == 0:
                        print(f"  ep{epoch} b{batch_counter:4d}  "
                              f"loss={loss.item():.3f}  "
                              f"diff={ld:.3f}  mse={lm:.3f}  noise={ln:.3f}  hard={lh:.3f}")
                except StopIteration:
                    pass
        else:
            for batch_idx, batch in enumerate(train_loader):
                loss, loss_diff, loss_mse_x0, loss_noise, loss_hard = train_step(
                    model, ema, optim, batch, alpha_cumprod, mu, sigma, DEVICE)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  ep{epoch} b{batch_idx} NaN/Inf, skip")
                    optim.zero_grad()
                    continue

                total_loss += loss.item()
                total_diff += loss_diff.item()
                total_mse += loss_mse_x0.item()
                total_noise += loss_noise.item()
                total_hard += loss_hard.item()
                n_batches += 1

                if batch_idx % 25 == 0:
                    print(f"  ep{epoch} b{batch_idx:4d}  "
                          f"loss={loss.item():.3f}  "
                          f"diff={loss_diff.item():.3f}  "
                          f"mse={loss_mse_x0.item():.3f}  "
                          f"noise={loss_noise.item():.3f}  "
                          f"hard={loss_hard.item():.3f}")

        avg = lambda x: x / max(n_batches, 1)
        elapsed = (time.time() - t_start) / 60
        print(f"Epoch {epoch}  avg_loss={avg(total_loss):.3f}  "
              f"avg_diff={avg(total_diff):.3f}  avg_mse={avg(total_mse):.3f}  "
              f"avg_noise={avg(total_noise):.3f}  avg_hard={avg(total_hard):.3f}  "
              f"elapsed={elapsed:.1f}min")

        # Save checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema.state_dict(),
            'norm_stats': {'mu': mu.cpu(), 'sigma': sigma.cpu()},
            'n_steps': N_STEPS,
            'cond_in': 8,
            'model_kwargs': {'dim': 256, 'depth': 6, 'heads': 8, 'cond_in': 8, 'n_steps': N_STEPS},
        }, SAVE_DIR / f"diffusion_epoch_{epoch}.pth")

    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'norm_stats': {'mu': mu.cpu(), 'sigma': sigma.cpu()},
        'n_steps': N_STEPS,
        'cond_in': 8,
        'model_kwargs': {'dim': 256, 'depth': 6, 'heads': 8, 'cond_in': 8, 'n_steps': N_STEPS},
    }, SAVE_DIR / "diffusion_final.pth")
    print(f"Training done! Saved to {SAVE_DIR / 'diffusion_final.pth'}")


if __name__ == "__main__":
    SAVE_DIR = Path("model/v1")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    main()
