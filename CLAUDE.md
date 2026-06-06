# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FloorSet** is a VLSI Floorplanning Dataset with Design Constraints of Real-World SoCs. It is the basis for the **ICCAD 2026 CAD Contest Problem C: The FloorSet Challenge (Data-Driven SoC Floorplanning)**.

The dataset has 2M synthetic floorplan layouts split into two variants:
- **FloorSet-Prime** (1M layouts, polygonal partitions) — files using `prime*` prefix
- **FloorSet-Lite** (1M layouts, rectangular-only partitions) — files using `lite*` prefix

Each floorplan has **21–120 blocks**. The contest is **rectangular-only** (Lite). Prime is the legacy/research variant.

Datasets are downloaded from [Hugging Face IntelLabs/FloorSet](https://huggingface.co/datasets/IntelLabs/FloorSet) into:
- `FloorSet/LiteTensorData/` — training (1M samples, ~9.5GB expanded)
- `FloorSet/LiteTensorDataTest/` — validation (100 samples, ~15MB)
- Hidden test set (100 samples) is held by the contest organizers.

## Install

```bash
pip install -r requirements.txt                    # For dataset loading + visualization
pip install -r iccad2026contest/requirements.txt   # For contest framework
```

Python deps: `torch>=2.0`, `numpy`, `matplotlib`, `Shapely>=2.0.5`, `tqdm`, `Requests`.

## Common Commands

All contest commands run from `iccad2026contest/`:

```bash
cd iccad2026contest

# Quick start — copy the B*-tree SA baseline, then evaluate
cp optimizer_template.py my_optimizer.py
python iccad2026_evaluate.py --evaluate my_optimizer.py              # Full validation set
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0  # Single test case
python iccad2026_evaluate.py --validate my_optimizer.py              # Format check (no eval)
python iccad2026_evaluate.py --validate my_optimizer.py --quick      # Format check (skip run)

# Save + re-score (no re-run of optimizer)
python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions
python iccad2026_evaluate.py --score my_optimizer_solutions.json

# Baselines, training data exploration, visualization
python iccad2026_evaluate.py --baseline --output baselines.json
python iccad2026_evaluate.py --training
python iccad2026_evaluate.py --visualize --test-id 0

# Differentiable training demo
python training_example.py
```

Legacy top-level loaders (not part of contest workflow, used by `*.py` notebook-style scripts):
```bash
python liteLoader.py      # Iterate FloorSet-Lite training
python primeLoader.py     # Iterate FloorSet-Prime training
python litetestLoader.py  # Iterate Lite validation
```

There is no formal test suite. `lite_dataset_test.py` and `prime_dataset_test.py` are Jupyter-style smoke tests for the dataset classes; the only "test" the contest cares about is running the optimizer against `--evaluate`.

## Code Architecture

### Two parallel pipelines

The repo contains the broader FloorSet research framework AND the ICCAD 2026 contest framework. They share the Lite dataset classes but have separate evaluation/scoring code.

**Research / legacy pipeline (top-level `*.py`):**
- `lite_dataset.py` → `FloorplanDatasetLite`, `floorplan_collate` (1M Lite training)
- `prime_dataset.py` → `FloorplanDataset` (1M Prime training, polygon-based)
- `lite_dataset_test.py` / `prime_dataset_test.py` → test set variants
- `cost.py` → `calculate_weighted_b2b_wirelength`, `calculate_weighted_p2b_wirelength`, `estimate_cost` (Prime evaluation, polygon-based)
- `utils.py` → `unpad_tensor`, `check_fixed_const`, `check_preplaced_const`, `check_mib_const`, `check_boundary_const`, `check_clust_const` — Shapely-based constraint checks for the **Prime** polygonal flow
- `validate.py` → script that runs `estimate_cost` over the Prime dataset
- `visualize.py` → matplotlib helpers used by the legacy `*Loader.py` scripts
- `intel_testsuite.md` / `intel_testsuite_lite.md` → optimal metrics for the 100 static Intel test cases (research reference)

**Contest pipeline (`iccad2026contest/`):**
- `iccad2026_evaluate.py` → the central file. Implements:
  - `FloorplanOptimizer` base class (the API contestants subclass — see "Optimizer Contract" below)
  - `evaluate_solution(...)` → full contest scoring (hard + soft constraints, gap-vs-baseline)
  - `compute_cost(...)` → `Cost = (1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel) · max(0.7, RuntimeFactor^0.3)`, capped just below M=10
  - `compute_total_score(...)` → `exp(n/12)`-weighted average across the 100 test cases
  - `compute_training_loss_differentiable(...)` → same formula in pure tensor form (omits RuntimeFactor) for backprop
  - `get_training_dataloader()` / `get_validation_dataloader()` → auto-downloading PyTorch `DataLoader`s
  - `ContestEvaluator` → loads user optimizer, runs the 100 validation cases, prints/scores results
  - `validate_submission(...)`, `generate_baselines(...)`, `score_saved_solutions(...)`, `visualize_test_case(...)`
  - `main()` CLI (argparse, see Common Commands)
  - Baseline optimizers included for reference: `RandomOptimizer`, `SimulatedAnnealingOptimizer` (NOT the official baseline — see `optimizer_template.py` for the B*-tree SA)
- `optimizer_template.py` → the **canonical starting point**. Provides a working `BStarTree` + SA baseline as `MyOptimizer(FloorplanOptimizer)`. Contestants are expected to subclass `FloorplanOptimizer` and rewrite `solve()`.
- `training_example.py` → end-to-end demo of differentiable loss, gradient flow, and dataloader unpacking

### Optimizer Contract

Contestants subclass `FloorplanOptimizer` (imported from `iccad2026_evaluate.py`) and implement:

```python
def solve(self, block_count, area_targets, b2b_connectivity,
          p2b_connectivity, pins_pos, constraints, target_positions=None) -> List[Tuple[float, float, float, float]]:
    # return list of (x, y, w, h) — one per block
```

`target_positions` is `[n, 4]` with `-1` for free blocks. **Fixed-shape blocks have `(w, h)` set, preplaced blocks have all of `(x, y, w, h)` set.** Both must be reproduced exactly (hard constraint, tolerance `1e-4`).

The evaluator **dynamically imports** the user's file with `importlib`, finds the first class that subclasses `FloorplanOptimizer` (and isn't `FloorplanOptimizer` itself), and instantiates it with `verbose=True`. Class names `MyOptimizer` / `Optimizer` / `ContestOptimizer` also work as fallbacks.

### Constraint model

`constraints` is `[n_blocks, 5]` = `[fixed, preplaced, mib_id, cluster_id, boundary_code]`:
- `fixed` / `preplaced` (cols 0, 1) → hard constraints (any deviation → cost = M = 10)
- `boundary` (col 4) → bitmask: `1`=left, `2`=right, `4`=top, `8`=bottom; corners are sums (e.g. `5`=top-left). Soft.
- `mib_id` (col 2) / `cluster_id` (col 3) → group IDs (0 = unconstrained). Grouping requires abutment; MIB requires identical (w,h). Soft.

The soft-violation denominator `N_soft` excludes fixed/preplaced (those are now hard), so `V_rel = (V_boundary + V_grouping + V_mib) / N_soft ∈ [0, 1]`.

### Data formats (Lite, contest-relevant)

Per batch, unpacking pattern (training, `liteLoader.py`):
```python
(area_target,        # bsz x n_blocks
 b2b_connectivity,   # bsz x b2b_edges x 3  (block_i, block_j, weight)
 p2b_connectivity,   # bsz x p2b_edges x 3   (pin_i, block_j, weight)
 pins_pos,           # bsz x n_pins x 2
 placement_constraints,  # bsz x n_blocks x 5
 tree_sol,           # bsz x (n_blocks-1) x 3  (B*-tree representation)
 fp_sol,             # bsz x n_blocks x 4      (w, h, x, y)
 metrics_sol) = batch
```

Per validation sample (100 cases, `litetestLoader.py`): same input tuple plus `labels = (polygons, metrics)` for ground-truth baseline extraction. Block count is `int((area_target != -1).sum().item())` after `squeeze(0)`.

### Scoring at a glance

```
Cost     = (1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel) · max(0.7, RuntimeFactor^0.3)    [feasible]
         = 10.0                                                                              [infeasible]
Total    = Σ Cost[i] · exp(n_i/12) / Σ exp(n_j/12)         across 100 test cases
```
- `HPWL_gap` / `Area_gap` are clamped to ≥ 0 (beating the baseline gives no bonus).
- Feasible cost is capped at `M − 1e-6 = 9.999999` so any feasible solution beats any infeasible one.
- Local evaluator sets `RuntimeFactor = 1.0` (neutral) — runtime impact is only scored on the official leaderboard (per-test-case cross-submission median).

## Working in this repo

- **Authors/contestants should not edit `iccad2026_evaluate.py` or `optimizer_template.py` directly** — copy `optimizer_template.py` to a new file for the submission.
- The two READMEs cover complementary ground: the top-level `README.md` documents the broader dataset; `iccad2026contest/README.md` is authoritative for the contest. The contest `README.md` also has a detailed changelog (current is v10, May 20 2026) that documents scoring-formula, hard/soft-constraint, and runtime-normalization changes.
- The contest specification PDF is at `iccad2026contest/FloorplanningContest_ICCAD_2026_v10.pdf` — it is the source of truth for any scoring ambiguity.
- Use `shapely` for the soft-constraint checks in `evaluate_solution`; the framework already wraps it in `try/except` and prints a warning if missing.
- `compute_total_score` weights use `exp(n/12)` (NOT `exp(n)`); large cases (n=116–120) carry ~34% of the total.
