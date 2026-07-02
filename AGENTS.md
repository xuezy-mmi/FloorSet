# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## Project Overview

**FloorSet** is a VLSI Floorplanning Dataset with Design Constraints of Real-World SoCs, and the basis for the **ICCAD 2026 CAD Contest Problem C**. The dataset has 2M synthetic floorplan layouts in two variants:
- **FloorSet-Prime** (1M layouts, polygonal partitions) — files with `prime*` prefix
- **FloorSet-Lite** (1M layouts, rectangular-only partitions) — files with `lite*` prefix

Each floorplan has **21–120 blocks**. The contest uses **Lite only** (rectangular). Prime is the legacy/research variant.

## Install

```bash
pip install -r requirements.txt                    # Dataset loading + visualization
pip install -r iccad2026contest/requirements.txt   # Contest framework
```

Core deps: `torch>=2.0`, `numpy`, `matplotlib`, `Shapely>=2.0.5`, `tqdm`, `Requests`.

## Common Commands

All contest commands run from `iccad2026contest/`:

```bash
cd iccad2026contest

# Evaluate optimizer on full validation set (100 cases)
python iccad2026_evaluate.py --evaluate my_optimizer.py

# Evaluate single test case (0-99) for debugging
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0

# Format validation (no eval)
python iccad2026_evaluate.py --validate my_optimizer.py

# Save solutions and re-score without re-running
python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions
python iccad2026_evaluate.py --score my_optimizer_solutions.json

# Generate baselines, explore training data, visualize
python iccad2026_evaluate.py --baseline --output baselines.json
python iccad2026_evaluate.py --training
python iccad2026_evaluate.py --visualize --test-id 0

# DiT training variants (GPU recommended)
python train_dit.py                   # Original DiT
python train_dit_v3.py                # v3 DiT (pre-LN, edge-bias)
python train_dit_hd.py                # HD DiT (3-branch attention, EMA)

# Evaluate DiT optimizers
python iccad2026_evaluate.py --evaluate dit_optimizer.py         # Original DiT
python iccad2026_evaluate.py --evaluate dit_optimizer_hd.py      # HD DiT (DDPM)
python iccad2026_evaluate.py --evaluate dit_optimizer_hd_sa.py   # HD DiT + SA hybrid

# Differentiable training loss demo
python training_example.py

# Inspect raw dataset fields/shapes
python check_input.py --mode train --root /path/to/FloorSet --worker-idx 3 --layout-idx 2
python check_input.py --mode test  --root /path/to/FloorSet --config-idx 21 --input-idx 1
```

Legacy top-level loaders (not part of contest workflow):
```bash
python liteLoader.py      # Iterate FloorSet-Lite training data
python primeLoader.py     # Iterate FloorSet-Prime training data
python litetestLoader.py  # Iterate Lite validation data
```

There is no formal test suite. The only "test" is running the optimizer against `--evaluate`.

## Code Architecture

### Two Parallel Pipelines

The repo contains the broader FloorSet research framework AND the ICCAD 2026 contest framework. They share the Lite dataset classes but have separate evaluation/scoring code.

**Research / legacy pipeline (top-level `*.py`):**
- `lite_dataset.py` / `prime_dataset.py` — PyTorch `Dataset` classes for training data
- `lite_dataset_test.py` / `prime_dataset_test.py` — test set variants
- `cost.py` — wirelength (`calculate_weighted_b2b_wirelength`, `calculate_weighted_p2b_wirelength`) and `estimate_cost` for **Prime** polygon-based evaluation
- `utils.py` — Shapely-based constraint checks for the Prime polygonal flow (`check_fixed_const`, `check_preplaced_const`, `check_mib_const`, `check_boundary_const`, `check_clust_const`)
- `validate.py` — runs `estimate_cost` over the Prime dataset
- `visualize.py` — matplotlib helpers used by legacy `*Loader.py` scripts

**Contest pipeline (`iccad2026contest/`):**
- `iccad2026_evaluate.py` — central file implementing:
  - `FloorplanOptimizer` base class (the API contestants subclass)
  - `evaluate_solution()` — full contest scoring (hard + soft constraints)
  - `compute_cost()` — the scoring formula (see below)
  - `compute_total_score()` — `exp(n/12)`-weighted average across 100 test cases
  - `compute_training_loss_differentiable()` — differentiable proxy for backprop
  - `get_training_dataloader()` / `get_validation_dataloader()` — auto-downloading DataLoaders
  - `ContestEvaluator` — orchestrates evaluation, plus CLI via argparse
- `optimizer_template.py` — **canonical starting point** with `BStarTree` + SA baseline. Contestants subclass `FloorplanOptimizer` and rewrite `solve()`.
- `training_example.py` — end-to-end demo of differentiable loss and dataloader unpacking

**DiT (Diffusion Transformer) approach — three experimental variants in `iccad2026contest/`:**

| Variant | Model | Utils | Optimizer | Train | Notes |
|---------|-------|-------|-----------|-------|-------|
| **Original** | `dit_model.py` | `dit_utils.py` | `dit_optimizer.py` | `train_dit.py` | Simple TransformerEncoder (dim=512, heads=8), standard DDPM |
| **v3** | `dit_model_v3.py` | `dit_utils_v3.py` | `dit_optimizer_v3.py` | `train_dit_v3.py`, `train_dit_v3_combine.py`, `train_dit_v3_prime.py` | Pre-LN, edge-bias self-attention, z-score normalization |
| **HD (HouseDiffusion-inspired)** | `dit_model_hd.py` | `dit_utils_hd.py` | `dit_optimizer_hd.py`, `dit_optimizer_hd_sa.py` | `train_dit_hd.py` | 3-branch masked attention (wire/clust/glob), sinusoidal PE, cosine schedule, EMA |

All variants use `norm_factor=1000.0` (must match between training and inference), `n_steps=1000`, and the same `(w, h, x, y)` output convention. `check_input.py` is a CLI helper to inspect raw training/validation sample fields and shapes.

The **HD variant** is the most architecturally distinctive — see `iccad2026contest/HouseDiffusion_README.md` for details. It adapts HouseDiffusion's 3-branch masked attention to VLSI semantics:
- `wire_attn` (wire_mask): attention only between blocks sharing a b2b edge
- `clust_attn` (clust_mask): attention only between blocks in the same MIB or cluster group
- `glob_attn` (glob_mask): full attention for global context

Mask convention: `mask == 1` means *block* (do not attend) — opposite of PyTorch's `attn_mask`. This matches HouseDiffusion's original convention.

`dit_optimizer_hd_sa.py` is a **hybrid** that uses diffusion for an initial layout, then runs a short SA refinement pass on top (mirrors HouseDiffusion's post-processing approach).

### Optimizer Contract

Contestants subclass `FloorplanOptimizer` and implement:

```python
def solve(self, block_count, area_targets, b2b_connectivity,
          p2b_connectivity, pins_pos, constraints, target_positions=None) -> List[Tuple[float, float, float, float]]:
    # return list of (x, y, w, h) — one per block
```

`target_positions` is `[n, 4]` with `-1` for free blocks. Fixed-shape blocks have `(w, h)` set; preplaced blocks have all of `(x, y, w, h)` set. Both must be reproduced exactly (hard constraint, tolerance `1e-4`).

The evaluator dynamically imports the user's file, finds the first subclass of `FloorplanOptimizer`, and instantiates it with `verbose=True`.

### Scoring Formula

```
Cost     = (1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel) · max(0.7, RuntimeFactor^0.3)    [feasible]
         = 10.0                                                                              [infeasible]
Total    = Σ Cost[i] · exp(n_i/12) / Σ exp(n_j/12)         across 100 test cases
```

- `HPWL_gap` / `Area_gap` are clamped to ≥ 0 (beating baseline gives no bonus)
- Feasible cost capped at `M − 1e-6 = 9.999999` (any feasible beats any infeasible)
- Local evaluator sets `RuntimeFactor = 1.0` (neutral); runtime scored only on official leaderboard

### Constraint Model

`constraints` is `[n_blocks, 5]` = `[fixed, preplaced, mib_id, cluster_id, boundary_code]`:
- **Hard** (violation → cost = 10): `fixed` (col 0), `preplaced` (col 1), no overlaps, area within 1% of target
- **Soft** (penalized via `exp(2·V_rel)`): `boundary` (col 4, bitmask: 1=left, 2=right, 4=top, 8=bottom, corners are sums), `cluster_id` (col 3), `mib_id` (col 2)
- `V_rel = (V_boundary + V_grouping + V_mib) / N_soft` where `N_soft` excludes fixed/preplaced

### Lite Data Format (contest-relevant)

Training batch unpacking:
```python
(area_target,        # bsz x n_blocks
 b2b_connectivity,   # bsz x b2b_edges x 3  (block_i, block_j, weight)
 p2b_connectivity,   # bsz x p2b_edges x 3  (pin_i, block_j, weight)
 pins_pos,           # bsz x n_pins x 2
 placement_constraints,  # bsz x n_blocks x 5
 tree_sol,           # bsz x (n_blocks-1) x 3  (B*-tree representation)
 fp_sol,             # bsz x n_blocks x 4      (w, h, x, y)
 metrics_sol) = batch
```

Block count: `int((area_target != -1).sum().item())` after `squeeze(0)`.

### HouseDiffusion (`house_diffusion/`)

A separate research sub-project forked from OpenAI's guided-diffusion. Uses a custom Transformer with 3-branch masked attention (door/self/gen) for generating residential floorplans from the RPLAN dataset. Has its own `CLAUDE.md` with detailed architecture notes. Not part of the contest workflow.

The core ideas from HouseDiffusion were adapted into the contest's **HD DiT variant** (`dit_model_hd.py` etc. in `iccad2026contest/`) — see table above and `iccad2026contest/HouseDiffusion_README.md`.

## Key Conventions

- **Do not edit `iccad2026_evaluate.py` or `optimizer_template.py` directly** — copy `optimizer_template.py` to a new file for submissions.
- The contest specification PDF at `iccad2026contest/FloorplanningContest_ICCAD_2026_v10.pdf` is the source of truth for scoring ambiguities.
- `compute_total_score` uses `exp(n/12)` weighting (NOT `exp(n)`); large cases (n=116–120) carry ~34% of total.
- Use `shapely` for soft-constraint checks; the framework wraps it in `try/except` if missing.
- The top-level `README.md` documents the broader dataset; `iccad2026contest/README.md` is authoritative for the contest and has a detailed changelog.
- Data lives at `FloorSet/LiteTensorData/` (training, ~9.5GB) and `FloorSet/LiteTensorDataTest/` (validation, ~15MB), auto-downloaded from Hugging Face if missing.
