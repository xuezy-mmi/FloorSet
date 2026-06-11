# HouseDiffusion-inspired floorplan optimizer

This is a port of the core ideas from **HouseDiffusion** (Shabani et al., 2022)
to the ICCAD 2026 FloorSet contest.

## Files

| File | Purpose |
|------|---------|
| `dit_model_hd.py`           | `HouseDiffusionDiT` — pre-LN DiT with **3-branch masked attention** (`wire_attn` / `clust_attn` / `glob_attn`) analogous to HouseDiffusion's `door_attn` / `self_attn` / `gen_attn` |
| `dit_utils_hd.py`           | Cosine schedule, DDPM/DDIM steps, EMA, mask-aware MSE, z-score stats, hard-constraint enforcement, overlap removal |
| `train_dit_hd.py`           | Training script: mixed precision, EMA, cosine LR, mask-aware MSE |
| `dit_optimizer_hd.py`       | Inference optimizer (DDPM and DDIM samplers). Falls back to SA if no checkpoint |
| `dit_optimizer_hd_sa.py`    | **Hybrid**: diffusion initialization + short SA refinement. Mirrors HouseDiffusion's post-processing step but with a proper SA loop on top |

## What was borrowed from HouseDiffusion

1. **3-branch masked attention** — three parallel multi-head attention
   modules with three different masks. In HouseDiffusion the masks
   restrict attention to (door-connected rooms) / (same room) / (all
   rooms). We adapt them to VLSI semantics:
     * `wire_attn` — only between blocks connected by a b2b edge
     * `clust_attn` — only between blocks in the same MIB or cluster group
     * `glob_attn` — full attention, used for global context
2. **Sinusoidal positional encoding** (HouseDiffusion's
   `PositionalEncoding` class).
3. **Per-block conditioning embeddings** combining constraint columns
   with graph features.
4. **Mask semantics** — `mask == 1` means *block* (do not attend), the
   opposite of PyTorch's `attn_mask`.
5. **Cosine beta schedule** (Nichol & Dhariwal 2021) used by
   HouseDiffusion.
6. **DDPM and DDIM samplers** in `dit_utils_hd.py` mirror
   `house_diffusion.gaussian_diffusion.p_sample` and
   `house_diffusion.gaussian_diffusion.ddim_sample`.
7. **EMA** of model parameters, same idea as
   `house_diffusion.train_util.TrainLoop._update_ema`.
8. **Post-processing** in the SA-hybrid optimizer mirrors
   `image_sample.py::save_samples` (overlap removal, dimension fix) but
   recast as a proper SA loop.

## What was changed for VLSI

* **No `expand_points` / no discrete bit head** — rectangular blocks
  only need continuous `(w, h, x, y)`.
* **Per-block conditioning** uses the 5-column FloorSet constraint
  tensor (`[fixed, preplaced, mib_id, cluster_id, boundary_code]`)
  instead of HouseDiffusion's per-corner conditioning.
* **Hard constraints** (fixed-shape and preplaced) are enforced
  *post-denoising* by overwriting the predicted positions, not via a
  loss term. This is the same pattern HouseDiffusion uses to snap
  doors to room edges.
* **z-score normalization** of the target layout (saved as
  `zscore_stats.pt`) is used in place of HouseDiffusion's `[-1, 1]`
  image-space normalization, because FloorSet coordinates are not in a
  fixed range.

## Quick start

Train (from `iccad2026contest/`):
```bash
python train_dit_hd.py \
    --data-path /home/xzy/eda/ \
    --batch-size 32 --num-train 10000 --epochs 20 \
    --save-dir /home/xzy/eda/model/hd_v1 \
    --dim 256 --depth 6 --heads 4 --use-fp16
```

Evaluate (DDPM):
```bash
python iccad2026_evaluate.py --evaluate dit_optimizer_hd.py
```

Evaluate (faster, fewer steps):
```bash
python iccad2026_evaluate.py --evaluate dit_optimizer_hd.py --test-id 0   # quick
```

Evaluate (hybrid diffusion + SA):
```bash
python iccad2026_evaluate.py --evaluate dit_optimizer_hd_sa.py
```

## Checkpoint layout

After `train_dit_hd.py`, `save-dir/` contains:
```
diffusion_final.pt         # model state_dict (last-iter)
diffusion_ema_final.pt     # EMA shadow state (used at inference)
diffusion_ep{N}.pt         # per-epoch model checkpoints
diffusion_ema_ep{N}.pt     # per-epoch EMA checkpoints
zscore_stats.pt            # {mu: [4], sigma: [4]}
meta.json                  # arch + diffusion config
train_log.json             # per-epoch losses
```

`dit_optimizer_hd.py` will look for these in this order:
1. `$save-dir/diffusion_ema_final.pt` (preferred — EMA generalizes better)
2. `$save-dir/diffusion_final.pt` (fallback)
3. `./diffusion_ema_final.pt`
4. `./diffusion_final.pt`

The default search paths are listed in `_CANDIDATE_DIRS` at the top of
`dit_optimizer_hd.py`.
