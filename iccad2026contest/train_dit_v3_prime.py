"""
train_dit_v3_prime.py - Train v3 DiT (pre-LN + edge-bias attention) on the
**Prime** dataset (polygonal floorplans), then save to
    /home/xzy/eda/model/v3_prime/diffusion_final.pth

This is the Prime-dataset counterpart of `train_dit_v3.py`. It re-uses
exactly the same model, optimizer, scheduler, loss formulation, and EMA
trick as the Lite version — only the data source and the rectangle
extraction step differ.

Key adaptions (vs. `train_dit_v3.py`):

  * The organiser-provided `iccad2026_evaluate.get_training_dataloader`
    only wraps the Lite dataset, so we cannot call it. Instead we
    import `FloorplanDatasetPrime` and `floorplan_collate` from the
    parent directory's `prime_dataset` module — this script does
    **not** modify any of the framework files.
  * `prime_dataset` returns `fp_sol` as a list of polygon vertex
    tensors. The v3 loss / norm-stats helper expect a uniform
    `[B, N, 4]` rectangle tensor in the `(w, h, x, y)` layout used
    by the Lite training set. We wrap the original collate with a
    small adapter (`floorplan_collate_prime_for_v3`) that turns each
    polygon into the bounding-box rectangle `(x_max-x_min,
    y_max-y_min, x_min, y_min)` and stacks everything into the
    8-tensor batch shape that `train_dit_v3.py` expects.
  * `compute_norm_stats` (in `dit_utils_v3.py`) is hard-coded to
    Lite's 8-tensor batch and uses `squeeze(0)`. We provide a
    batch-aware local copy (`compute_norm_stats_prime`) that walks
    the Prime-format batch instead.

Just edit the ``CONFIG`` block and run:

    python train_dit_v3_prime.py
"""
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

# Parent dir (where prime_dataset.py lives) is NOT inside the contest
# framework, so we explicitly add it to sys.path WITHOUT touching
# iccad2026_evaluate.py.
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from prime_dataset import FloorplanDatasetPrime  # noqa: E402
from torch.utils.data import DataLoader          # noqa: E402

from dit_model_v3 import DiffusionTransformer    # noqa: E402
from dit_utils_v3 import (                       # noqa: E402
    CosineSchedule, q_sample_masked,
    vectorized_diff_loss, hard_violation_loss,
)


# =============================================================================
# CONFIG  —  all hyperparameters & paths live here. Edit and run.
# =============================================================================
CONFIG = {
    # -------- I/O --------
    "data_path":        "/home/xzy/eda/",            # FloorSet data root (contains PrimeTensorData/)
    "save_dir":         "/home/xzy/eda/model/v3_prime",
    "log_interval":     25,                          # print every N batches

    # -------- data sampling --------
    "num_train":        4000,                        # subset of layouts; 0 = use all
    "batch_size":       8,                           # global batch size (per-step)
    "num_workers":      0,                           # DataLoader workers (Prime is in-RAM)

    # -------- training schedule --------
    "epochs":           20,
    "lr":               2e-4,
    "weight_decay":     0.0,
    "grad_clip":        1.0,
    "lambda_noise":     0.05,
    "lambda_hard":      1.0,

    # -------- diffusion --------
    "n_steps":          1000,                        # DDPM timesteps (unused for cosine, kept for arch)

    # -------- norm-stats pre-scan --------
    "norm_max_batches": 64,                          # how many batches to scan for z-score

    # -------- model architecture (must match v3 Lite) --------
    "dim":              256,
    "depth":            6,
    "heads":            8,
    "cond_in":          8,
    "dropout":          0.1,
}


# =============================================================================
# Polygon → rectangle conversion
# =============================================================================
def _polygon_to_rect(poly: torch.Tensor) -> torch.Tensor:
    """Convert a single polygon (variable-length `[V, 2]`) to `[4]` = (w, h, x, y).

    The Prime format pads each polygon to 14 vertices with -1 placeholders,
    so we first filter out the placeholders. Empty / fully-padded polygons
    return -1 (sentinel) so the caller can mask them.
    """
    if poly is None or poly.numel() == 0:
        return torch.tensor([-1.0, -1.0, -1.0, -1.0])
    valid = poly[poly[:, 0] != -1]
    if valid.numel() == 0:
        return torch.tensor([-1.0, -1.0, -1.0, -1.0])
    # `valid` is fp16 in some files; cast to float32 to avoid dtype issues.
    v = valid.float()
    x_min = v[:, 0].min()
    y_min = v[:, 1].min()
    x_max = v[:, 0].max()
    y_max = v[:, 1].max()
    w = (x_max - x_min).clamp(min=0.0)
    h = (y_max - y_min).clamp(min=0.0)
    return torch.stack([w, h, x_min, y_min])


def _polygons_to_rects(padded_polygons: torch.Tensor) -> torch.Tensor:
    """Vectorised-ish conversion of `[B, N_max, 14, 2]` to `[B, N_max, 4]`.

    Uses pure tensor ops (no Python loop) by:
      * treating each polygon as a point cloud
      * min/max over the 14 vertex axis
      * replacing -1 with `+inf` / `-inf` for x and y so they don't
        affect the min/max
    """
    B, N_max, V, _ = padded_polygons.shape
    poly = padded_polygons.float()  # avoid fp16 mean overflow
    # Build a per-polygon valid-vertex mask `[B, N_max, 14]`.
    valid_v = (poly[..., 0] != -1) & (poly[..., 1] != -1)
    # We need a mask broadcastable over (x, y) — valid_v [..., None] → [B,N,V,1]
    m = valid_v.float().unsqueeze(-1)
    safe = torch.where(m > 0, poly, torch.zeros_like(poly))

    # Compute sums/counts for masked min/max.
    cnt = m.sum(dim=2)  # [B, N_max, 1]
    cnt_safe = cnt.clamp(min=1.0)
    # Treat all-invalid polygons (cnt == 0) as fully-padded sentinels
    # (we'll fix those at the end).
    has_any = (cnt > 0)

    # For min, push invalid entries to +inf. For max, push invalid to -inf.
    inf = torch.tensor(float("inf"), dtype=poly.dtype, device=poly.device)
    neg_inf = torch.tensor(float("-inf"), dtype=poly.dtype, device=poly.device)
    safe_min = torch.where(m > 0, poly, inf.expand_as(poly))
    safe_max = torch.where(m > 0, poly, neg_inf.expand_as(poly))

    x_min = safe_min[..., 0].min(dim=2).values  # [B, N_max]
    y_min = safe_min[..., 1].min(dim=2).values
    x_max = safe_max[..., 0].max(dim=2).values
    y_max = safe_max[..., 1].max(dim=2).values

    w = (x_max - x_min).clamp(min=0.0)
    h = (y_max - y_min).clamp(min=0.0)

    rects = torch.stack([w, h, x_min, y_min], dim=-1)  # [B, N_max, 4]
    # Mark no-valid-vertex polygons with -1 so the training mask can drop them.
    rects = torch.where(has_any, rects, torch.full_like(rects, -1.0))
    return rects


# =============================================================================
# Custom collate:  prime_dataset batch  ->  Lite-style 8-tensor batch
# =============================================================================
def _prime_collate_for_v3(batch: List[dict]) -> Tuple[torch.Tensor, ...]:
    """Adapter that calls `prime_dataset.floorplan_collate` and reshapes
    its output into the 8-tensor batch that `train_dit_v3.py` expects:

        (area, b2b, p2b, pins, constraints, tree_sol, fp_sol, metrics)

    The 6th slot (`tree_sol`) is a Prime-specific dummy: Prime has no
    B*-tree, so we substitute a `[B, 1, 3]` zeros tensor. The v3
    training loop never uses `tree_sol`, so this is safe.

    The 7th slot (`fp_sol`) is the per-block rectangle tensor
    `[B, N_max, 4]` obtained by bounding-box-extracting the polygon
    list returned by `pad_polygons`. This is the data the v3 model
    actually trains on.

    All tensors are cast to **float32** to keep
    `dit_utils_v3.aggregate_graph_features` happy (it does
    `scatter_add_` which fails when target/src dtypes differ — the
    Prime files are stored as fp16 but the v3 helper expects fp32).
    """
    from prime_dataset import floorplan_collate as _prime_collate

    inputs_list, (padded_polygons, metrics) = _prime_collate(batch)
    area, b2b, p2b, pins, constraints = (t.float() for t in inputs_list)

    # padded_polygons: [B, N_max, 14, 2] (padded with -1)
    fp_sol = _polygons_to_rects(padded_polygons)  # [B, N_max, 4]
    fp_sol = fp_sol.float()

    B = area.shape[0]
    # Prime has no B*-tree — provide a minimal placeholder so the
    # 8-tensor shape matches the Lite training script.
    tree_sol = torch.zeros(B, 1, 3, dtype=torch.float32)
    metrics = metrics.float()

    return (area, b2b, p2b, pins, constraints, tree_sol, fp_sol, metrics)


# =============================================================================
# Batch-aware z-score stats for Prime (replaces dit_utils_v3.compute_norm_stats)
# =============================================================================
def _compute_norm_stats_prime(ds, max_batches: int = 64, batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std of (w, h, x, y) across the Prime dataset.

    Iterates the underlying `FloorplanDatasetPrime` (or a `Subset` of it)
    directly, builds mini-batches of raw item-dicts, and runs the
    custom collate so we get the same `[B, N, 4]` rectangle tensor
    the training loop sees.
    """
    ws, hs, xs, ys = [], [], [], []
    n_items = min(max_batches * batch_size, len(ds))
    for i in range(0, n_items, batch_size):
        items = [ds[j] for j in range(i, min(i + batch_size, n_items))]
        _, _, _, _, _, _, fp_sol, _ = _prime_collate_for_v3(items)
        # fp_sol: [B, N, 4] in (w, h, x, y) order, padded with -1.
        valid = (fp_sol != -1).all(dim=-1)  # [B, N]
        fv = fp_sol[valid]                   # [total_valid, 4]
        if fv.numel() == 0:
            continue
        ws.append(fv[:, 0])
        hs.append(fv[:, 1])
        xs.append(fv[:, 2])
        ys.append(fv[:, 3])
    if not ws:
        # Degenerate fallback — return unit-norm stats so the model
        # still trains (it will essentially denormalise to raw scale).
        z = torch.zeros(4)
        o = torch.ones(4)
        return z, o
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


# =============================================================================
# Training
# =============================================================================
def main(cfg=CONFIG):
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Saving checkpoints to: {save_dir}")

    n_steps    = int(cfg["n_steps"])
    n_epochs   = int(cfg["epochs"])
    batch_size = int(cfg["batch_size"])
    lr         = float(cfg["lr"])
    lambda_n   = float(cfg["lambda_noise"])
    lambda_h   = float(cfg["lambda_hard"])
    grad_clip  = float(cfg["grad_clip"])
    num_train  = int(cfg["num_train"])
    log_int    = int(cfg["log_interval"])

    # --- data ---
    print("Loading Prime dataset ...")
    ds = FloorplanDatasetPrime(cfg["data_path"])
    if num_train > 0:
        # Sub-sample by index — keeps Prime's file-cache efficient.
        # torch.utils.data.Subset would still call __getitem__ in order
        # which preserves locality. We instead construct a tiny wrapper.
        from torch.utils.data import Subset
        # Prime dataset is huge (~1000 configs × 10 files × 1000 layouts = 1M);
        # wrap the first `num_train` items only.
        keep = min(num_train, len(ds))
        ds = Subset(ds, list(range(keep)))
        print(f"  using first {keep} Prime layouts for training")
    else:
        print(f"  using all {len(ds)} Prime layouts for training")
    train_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=int(cfg["num_workers"]),
        collate_fn=_prime_collate_for_v3,
        drop_last=True,
    )
    print(f"  train batches/epoch: {len(train_loader)}")

    # --- z-score stats ---
    print("Computing z-score stats ...")
    from torch.utils.data import Subset as _Subset
    scan_keep = min(int(num_train) if num_train > 0 else len(ds), 512)
    scan_ds = FloorplanDatasetPrime(cfg["data_path"])
    scan_ds = _Subset(scan_ds, list(range(scan_keep)))
    mu, sigma = _compute_norm_stats_prime(
        scan_ds,
        max_batches=int(cfg["norm_max_batches"]),
        batch_size=8,
    )
    print(f"  mu    = {mu.tolist()}")
    print(f"  sigma = {sigma.tolist()}")
    mu, sigma = mu.to(device), sigma.to(device)

    # --- model + EMA ---
    dim, depth, heads, cond_in = (
        int(cfg["dim"]), int(cfg["depth"]), int(cfg["heads"]), int(cfg["cond_in"]),
    )
    model = DiffusionTransformer(
        dim=dim, depth=depth, heads=heads, cond_in=cond_in,
        n_steps=n_steps, dropout=float(cfg["dropout"]),
    ).to(device)
    ema = DiffusionTransformer(
        dim=dim, depth=depth, heads=heads, cond_in=cond_in,
        n_steps=n_steps, dropout=float(cfg["dropout"]),
    ).to(device)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=float(cfg["weight_decay"]),
    )
    sched = CosineSchedule(n_steps)
    alpha_cumprod = sched.alpha_cumprod.to(device)

    # --- training loop (mirrors train_dit_v3.py) ---
    print("Begin training ...")
    t_start = time.time()
    for epoch in range(n_epochs):
        total_loss, total_diff, total_noise, total_hard, n_batches = 0.0, 0.0, 0.0, 0.0, 0
        for batch_idx, batch in enumerate(train_loader):
            area_target, b2b, p2b, pins, constraints, _tree_sol, fp_sol, metrics = batch
            area_target = area_target.to(device)
            b2b         = b2b.to(device)
            p2b         = p2b.to(device)
            pins        = pins.to(device)
            constraints = constraints.to(device)
            fp_sol      = fp_sol.to(device)
            metrics     = metrics.to(device)

            B, N_max, _ = fp_sol.shape
            # The valid-block mask comes from area_target (col 0
            # padded with -1 in the Prime dataset) — but to be
            # robust we also intersect with the fp_sol sentinel
            # produced by `_polygons_to_rects`.
            valid_area = (area_target != -1)
            valid_rect = (fp_sol != -1).all(dim=-1)
            valid = (valid_area & valid_rect).unsqueeze(-1).expand_as(fp_sol).float()  # [B, N, 4]
            mask = valid  # alias used in v3 utils

            # z-score normalize target
            x0 = ((fp_sol - mu) / sigma) * mask

            t = torch.randint(0, n_steps, (B,), device=device)
            noise = torch.randn_like(x0)
            x_t, _ = q_sample_masked(x0, t, alpha_cumprod, mask, noise)

            pred_noise = model(x_t, t, area_target, b2b, p2b, pins, constraints)

            # noise loss
            loss_noise = F.mse_loss(
                pred_noise * mask, noise * mask,
                reduction='sum',
            ) / (mask.sum() + 1e-6)

            # x_0 reconstruction
            a_bar = alpha_cumprod[t].view(-1, 1, 1).clamp(min=1e-6)
            x0_pred = (
                (x_t - torch.sqrt(1.0 - a_bar) * pred_noise)
                / torch.sqrt(a_bar)
            ) * mask
            x0_pred = x0_pred.clamp(-5.0, 5.0)

            # unnormalize to real scale (w, h, x, y)
            pos_real = x0_pred * sigma + mu
            pos_real = pos_real.clamp(min=0.0)

            # per-sample diff + hard loss
            loss_diff = torch.zeros((), device=device)
            loss_hard = torch.zeros((), device=device)
            count = 0
            for i in range(B):
                n_v = int(valid_area[i].sum().item())
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
                    fixed     = constraints[i, :n_v, 0] > 0
                    preplaced = constraints[i, :n_v, 1] > 0

                    # Build target_pos in (x, y, w, h) format for hard loss.
                    target_pos = torch.full_like(pos_i, -1.0)
                    if fixed.any():
                        pos_i[fixed, 2] = fp_sol[i, :n_v][fixed, 0]
                        pos_i[fixed, 3] = fp_sol[i, :n_v][fixed, 1]
                        target_pos[fixed, 2] = fp_sol[i, :n_v][fixed, 0]
                        target_pos[fixed, 3] = fp_sol[i, :n_v][fixed, 1]
                    if preplaced.any():
                        pos_i[preplaced, 0] = fp_sol[i, :n_v][preplaced, 2]
                        pos_i[preplaced, 1] = fp_sol[i, :n_v][preplaced, 3]
                        pos_i[preplaced, 2] = fp_sol[i, :n_v][preplaced, 0]
                        pos_i[preplaced, 3] = fp_sol[i, :n_v][preplaced, 1]
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

            loss = loss_diff + lambda_n * loss_noise + lambda_h * loss_hard

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ep{epoch} b{batch_idx} NaN/Inf, skip")
                optim.zero_grad()
                continue

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()

            with torch.no_grad():
                for p_em, p_m in zip(ema.parameters(), model.parameters()):
                    p_em.mul_(0.999).add_(p_m, alpha=0.001)

            total_loss  += loss.item()
            total_diff  += loss_diff.item()
            total_noise += loss_noise.item()
            total_hard  += loss_hard.item()
            n_batches  += 1
            if batch_idx % log_int == 0:
                print(
                    f"  ep{epoch} b{batch_idx:4d}  loss={loss.item():.3f}  "
                    f"diff={loss_diff.item():.3f}  noise={loss_noise.item():.3f}  "
                    f"hard={loss_hard.item():.3f}",
                    flush=True,
                )

        avg = lambda x: x / max(n_batches, 1)
        print(
            f"Epoch {epoch}  avg_loss={avg(total_loss):.3f}  "
            f"avg_diff={avg(total_diff):.3f}  avg_noise={avg(total_noise):.3f}  "
            f"avg_hard={avg(total_hard):.3f}  "
            f"elapsed={(time.time() - t_start) / 60:.1f}min",
            flush=True,
        )

    # --- save checkpoint (same schema as train_dit_v3.py) ---
    save_path = save_dir / "diffusion_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'ema_state_dict':   ema.state_dict(),
        'norm_stats':       {'mu': mu.cpu(), 'sigma': sigma.cpu()},
        'n_steps':          n_steps,
        'cond_in':          cond_in,
        'model_kwargs': {
            'dim': dim, 'depth': depth, 'heads': heads,
            'cond_in': cond_in, 'n_steps': n_steps,
        },
        'dataset':          'prime',  # tag so a future optimizer can pick the right loader
    }, save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
