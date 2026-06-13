"""
train_dit_v3_combine.py - Train v3 DiT (pre-LN + edge-bias attention) on
**both** the Lite and Prime datasets in a single training run.

This combines the data sources of `train_dit_v3.py` (Lite) and
`train_dit_v3_prime.py` (Prime) into one model. Each epoch walks
through the Lite subset and the Prime subset back-to-back via
`itertools.chain`, so the model sees the joint distribution of both
datasets without doubling the per-step memory cost.

Key design points:

  * The contest framework (`iccad2026_evaluate.py`) is **not**
    modified. We use the existing `get_training_dataloader` for Lite
    and the existing `FloorplanDatasetPrime` from the parent
    directory for Prime.
  * The custom collate `_prime_collate_for_v3` from
    `train_dit_v3_prime.py` is imported to convert Prime polygons
    into the same `[B, N, 4]` rectangle layout that the v3 model
    expects. After this conversion, both Lite and Prime batches
    have an **identical** 8-tensor structure, so the training loop
    has a single uniform code path.
  * The z-score normalization statistics (`mu`, `sigma`) are computed
    over a **mix** of Lite and Prime samples so the model is
    normalized in a single, joint coordinate frame.
  * Model + optimizer + scheduler + loss + EMA + save schema are
    identical to `train_dit_v3.py`. The final checkpoint is saved
    to:
        /home/xzy/eda/model/v3_combine/diffusion_final.pth
    in the same schema as v3 Lite and v3 Prime (so the same
    optimizer file can load it).

Just edit the ``CONFIG`` block and run:

    python train_dit_v3_combine.py
"""
import itertools
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# ----------------------------------------------------------------------
# sys.path setup: parent dir for `prime_dataset`, current dir for
# `iccad2026_evaluate` and the v3 modules. The framework is untouched.
# ----------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_THIS_DIR))

from iccad2026_evaluate import get_training_dataloader        # noqa: E402
from prime_dataset import FloorplanDatasetPrime                # noqa: E402

# Reuse Prime -> v3-rectangle helpers from the dedicated Prime trainer.
from train_dit_v3_prime import (                              # noqa: E402
    _prime_collate_for_v3,
    _compute_norm_stats_prime,
)

from dit_model_v3 import DiffusionTransformer                  # noqa: E402
from dit_utils_v3 import (                                    # noqa: E402
    CosineSchedule, q_sample_masked,
    vectorized_diff_loss, hard_violation_loss,
)


# =============================================================================
# CONFIG  —  all hyperparameters & paths live here. Edit and run.
# =============================================================================
CONFIG = {
    # -------- I/O --------
    "data_path":        "/home/xzy/eda/",            # FloorSet data root
    "save_dir":         "/home/xzy/eda/model/v3_combine",
    "log_interval":     25,                          # print every N batches

    # -------- data sampling --------
    "num_train_lite":   4000,                        # Lite subset per epoch; 0 = all
    "num_train_prime":  4000,                        # Prime subset per epoch; 0 = all
    "batch_size":       8,                           # per-loader batch size
    "num_workers":      0,

    # -------- training schedule --------
    "epochs":           20,
    "lr":               2e-4,
    "weight_decay":     0.0,
    "grad_clip":        1.0,
    "lambda_noise":     0.05,
    "lambda_hard":      1.0,

    # -------- diffusion --------
    "n_steps":          1000,                        # DDPM timesteps (matches arch)

    # -------- norm-stats pre-scan --------
    "norm_max_batches": 32,                          # per-source batch budget

    # -------- model architecture (must match v3 Lite) --------
    "dim":              256,
    "depth":            6,
    "heads":            8,
    "cond_in":          8,
    "dropout":          0.1,
}


# =============================================================================
# Helpers
# =============================================================================
def _lite_subset_indices(total: int, num_train: int) -> List[int]:
    """Return the first `min(num_train, total)` indices as a list."""
    return list(range(min(num_train, total)))


def _compute_norm_stats_combine(
    lite_loader: DataLoader,
    prime_subset: Subset,
    max_batches_each: int = 32,
    lite_batch_size: int = 8,
    prime_batch_size: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel (w, h, x, y) mu/sigma over a mix of Lite
    and Prime samples so the model is normalized in a single, joint
    coordinate frame.

    Lite samples come from a pre-built DataLoader (which already
    yields the 8-tensor batch); Prime samples come from iterating
    `prime_subset` directly and running the custom collate.
    """
    ws: List[torch.Tensor] = []
    hs: List[torch.Tensor] = []
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []

    # ---- Lite ----
    for i, batch in enumerate(lite_loader):
        if i >= max_batches_each:
            break
        _, _, _, _, _, _, fp_sol, _ = batch
        # fp_sol: [B, N, 4] in (w, h, x, y) order, padded with -1.
        valid = (fp_sol != -1).all(dim=-1)
        fv = fp_sol[valid]
        if fv.numel() == 0:
            continue
        ws.append(fv[:, 0])
        hs.append(fv[:, 1])
        xs.append(fv[:, 2])
        ys.append(fv[:, 3])

    # ---- Prime ----
    n_items = min(max_batches_each * prime_batch_size, len(prime_subset))
    for i in range(0, n_items, prime_batch_size):
        items = [prime_subset[j] for j in range(
            i, min(i + prime_batch_size, n_items)
        )]
        _, _, _, _, _, _, fp_sol, _ = _prime_collate_for_v3(items)
        valid = (fp_sol != -1).all(dim=-1)
        fv = fp_sol[valid]
        if fv.numel() == 0:
            continue
        ws.append(fv[:, 0])
        hs.append(fv[:, 1])
        xs.append(fv[:, 2])
        ys.append(fv[:, 3])

    if not ws:
        # Degenerate fallback — return unit-norm stats so the model
        # still trains.
        return torch.zeros(4), torch.ones(4)

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

    n_steps        = int(cfg["n_steps"])
    n_epochs       = int(cfg["epochs"])
    batch_size     = int(cfg["batch_size"])
    lr             = float(cfg["lr"])
    lambda_n       = float(cfg["lambda_noise"])
    lambda_h       = float(cfg["lambda_hard"])
    grad_clip      = float(cfg["grad_clip"])
    num_lite       = int(cfg["num_train_lite"])
    num_prime      = int(cfg["num_train_prime"])
    num_workers    = int(cfg["num_workers"])
    log_int        = int(cfg["log_interval"])
    norm_batches   = int(cfg["norm_max_batches"])

    # ----------------------------------------------------------------------
    # Data: two DataLoaders, one per source. We rebuild them every
    # epoch so each loader's shuffle actually reshuffles. We chain
    # their iterations so each epoch walks all of Lite first, then
    # all of Prime (a single uniform code path in the training loop).
    # ----------------------------------------------------------------------
    print("Loading Lite dataset ...")
    lite_full = get_training_dataloader(
        data_path=cfg["data_path"],
        batch_size=batch_size,
        num_samples=None,   # peek to learn the dataset length
        shuffle=False,
    )
    n_lite_total = len(lite_full.dataset) if hasattr(lite_full, "dataset") else 0
    print(f"  Lite total samples on disk: {n_lite_total:,}")
    if num_lite > 0:
        lite_idx = _lite_subset_indices(n_lite_total, num_lite)
        print(f"  using first {len(lite_idx):,} Lite samples for training")
    else:
        lite_idx = None
        print(f"  using all {n_lite_total:,} Lite samples for training")
    del lite_full  # we'll rebuild below

    print("Loading Prime dataset ...")
    prime_full = FloorplanDatasetPrime(cfg["data_path"])
    n_prime_total = len(prime_full)
    print(f"  Prime total samples on disk: {n_prime_total:,}")
    if num_prime > 0:
        prime_idx = list(range(min(num_prime, n_prime_total)))
        print(f"  using first {len(prime_idx):,} Prime samples for training")
    else:
        prime_idx = list(range(n_prime_total))
        print(f"  using all {n_prime_total:,} Prime samples for training")
    prime_subset = Subset(prime_full, prime_idx)

    # ----------------------------------------------------------------------
    # z-score stats over a mix of both sources.
    # ----------------------------------------------------------------------
    print("Computing z-score stats over a mix of Lite + Prime ...")
    lite_scan = get_training_dataloader(
        data_path=cfg["data_path"],
        batch_size=8,
        num_samples=min(512, n_lite_total) if lite_idx is None
                      else min(512, len(lite_idx)),
        shuffle=False,
    )
    mu, sigma = _compute_norm_stats_combine(
        lite_loader=lite_scan,
        prime_subset=prime_subset,
        max_batches_each=norm_batches,
        lite_batch_size=8,
        prime_batch_size=8,
    )
    print(f"  mu    = {mu.tolist()}")
    print(f"  sigma = {sigma.tolist()}")
    mu, sigma = mu.to(device), sigma.to(device)
    del lite_scan  # free the scan loader

    # ----------------------------------------------------------------------
    # Model + EMA.
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # Training loop. Each epoch we rebuild the per-source loaders so
    # shuffling reshuffles; we then chain them via itertools.chain so
    # the body of the loop is a single uniform code path.
    # ----------------------------------------------------------------------
    print("Begin training ...")
    t_start = time.time()
    for epoch in range(n_epochs):
        # ---- rebuild loaders (fresh shuffles each epoch) ----
        lite_loader = get_training_dataloader(
            data_path=cfg["data_path"],
            batch_size=batch_size,
            num_samples=len(lite_idx) if lite_idx is not None else None,
            shuffle=True,
        )
        if lite_idx is not None:
            # `get_training_dataloader(num_samples=...)` already wraps
            # the first N samples internally, so we don't need to
            # manually subset. Just rebuild the loader with the
            # desired sample count.
            pass
        prime_loader = DataLoader(
            prime_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=_prime_collate_for_v3,
            drop_last=True,
        )
        # itertools.chain: walk all Lite batches, then all Prime batches.
        n_lite_batches = len(lite_loader)
        n_prime_batches = len(prime_loader)
        n_batches_total = n_lite_batches + n_prime_batches
        combined = itertools.chain(lite_loader, prime_loader)

        total_loss, total_diff, total_noise, total_hard, n_batches = 0.0, 0.0, 0.0, 0.0, 0
        lite_loss_sum, prime_loss_sum, lite_n, prime_n = 0.0, 0.0, 0, 0
        for batch_idx, batch in enumerate(combined):
            is_prime = batch_idx >= n_lite_batches
            area_target, b2b, p2b, pins, constraints, _tree_sol, fp_sol, metrics = batch
            # Cast to float32 — the v3 graph-feature helper does
            # `scatter_add_` which fails when target/src dtypes differ.
            area_target = area_target.float().to(device)
            b2b         = b2b.float().to(device)
            p2b         = p2b.float().to(device)
            pins        = pins.float().to(device)
            constraints = constraints.float().to(device)
            fp_sol      = fp_sol.float().to(device)
            metrics     = metrics.float().to(device)

            B, N_max, _ = fp_sol.shape
            # Valid mask: block has an area target AND a real (non-sentinel) rectangle.
            valid_area = (area_target != -1)
            valid_rect = (fp_sol != -1).all(dim=-1)
            valid = (valid_area & valid_rect).unsqueeze(-1).expand_as(fp_sol).float()
            mask = valid

            # z-score normalize target.
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
            if is_prime:
                prime_loss_sum += loss.item()
                prime_n += 1
            else:
                lite_loss_sum += loss.item()
                lite_n += 1

            if batch_idx % log_int == 0:
                src = "P" if is_prime else "L"
                print(
                    f"  ep{epoch} b{batch_idx:4d}/{n_batches_total} [{src}]  "
                    f"loss={loss.item():.3f}  diff={loss_diff.item():.3f}  "
                    f"noise={loss_noise.item():.3f}  hard={loss_hard.item():.3f}",
                    flush=True,
                )

        avg = lambda x: x / max(n_batches, 1)
        lite_avg = lite_loss_sum / max(lite_n, 1)
        prime_avg = prime_loss_sum / max(prime_n, 1)
        print(
            f"Epoch {epoch}  avg_loss={avg(total_loss):.3f}  "
            f"avg_diff={avg(total_diff):.3f}  avg_noise={avg(total_noise):.3f}  "
            f"avg_hard={avg(total_hard):.3f}  "
            f"lite={lite_avg:.3f} (n={lite_n})  prime={prime_avg:.3f} (n={prime_n})  "
            f"elapsed={(time.time() - t_start) / 60:.1f}min",
            flush=True,
        )

    # ----------------------------------------------------------------------
    # Save checkpoint (same schema as train_dit_v3.py / train_dit_v3_prime.py).
    # ----------------------------------------------------------------------
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
        # tag so a future optimizer can pick the right loader
        'dataset': 'lite+prime',
    }, save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
