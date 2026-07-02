# ICCAD 2026 FloorSet Challenge

**Contest specification (PDF):** [FloorplanningContest_ICCAD_2026_v10.pdf](./FloorplanningContest_ICCAD_2026_v10.pdf)

## Changelog

### May 20, 2026
- **PDF**: Updated to v10 — corrected Total Score formula to use `exp(n/12)` weighting; updated Equation 2 to cap feasible cost at M−10⁻⁶; listed all 4 hard constraints in the Objective Function section; corrected "Total Score close to 0.1" to "close to 1.0"
- **iccad2026_evaluate.py**: Exponential weighting changed from `exp(n)` to `exp(n/12)`; feasible cost capped at M−10⁻⁶ to guarantee feasible always scores below infeasible; local RuntimeFactor set to 1.0 (neutral) for all test cases; `compute_cost` docstring updated to reflect all 4 hard constraints, gap clamping, and feasible cost cap
- **optimizer_template.py**: Added dimension immutability (fixed-shape/preplaced) to hard constraint list in docstring
- **training_example.py**: Corrected module docstring and inline comment — training loss is a proxy that omits RuntimeFactor and uses differentiable soft violations
- **README**: Updated PDF link to v10; added `--baseline` and `--visualize` to Commands Reference table; documented boundary constraint bitmask encoding; updated Scoring section with runtime normalization note, feasible cost cap, exponential weighting table, and all 4 hard constraints

### April 19, 2026
- **Hard constraints**: Fixed-shape and preplaced constraints are now **hard constraints** — any deviation from specified dimensions (or location for preplaced) renders the solution infeasible (cost = M = 10)
- **Soft constraints**: Removed fixed-shape and preplaced from the soft violation formula; only boundary, grouping, and MIB remain as soft constraints
- **Violation formula**: Updated `V_relative = (V_boundary + V_grouping + V_mib) / N_soft` and `N_soft = |B_boundary| + Σ(|G_p| - 1) + Σ(|M_q| - 1)` — fixed-shape and preplaced no longer contribute
- **Evaluation code**: `iccad2026_evaluate.py` updated to match; `total_soft_violations` and `n_soft` no longer include fixed/preplaced counts
- **PDF**: Updated to v9 with all hard/soft constraint changes and footnotes

### April 15, 2026
- **Scoring**: `compute_total_score` now uses exponential weighting by block count (was linear)
- **Soft constraints**: `evaluate_solution` now computes all five violation types (fixed, preplaced, grouping, boundary, MIB) and correct `N_soft` denominator; previously hardcoded to 0
- **Hard constraints**: Added dimension immutability check for fixed-shape and preplaced blocks; violations render solutions infeasible
- **Baseline extraction**: Fixed inconsistent logic between `_extract_baseline` and `generate_baselines` for HPWL metrics
- **Training helpers**: Fixed `compute_training_loss_batch` passing `constraints` as `baseline_metrics`; added explicit `metrics_batch` parameter
- **Validator**: Fixed `--validate` incorrectly picking `FloorplanOptimizer` base class instead of user's subclass
- **Dataloader docstrings**: Corrected usage examples for `get_training_dataloader` and `get_validation_dataloader`
- **Shapely warning**: Added warning message when `shapely` is not installed (required dependency)
- **Robustness**: Added `.flatten()` to `torch.nonzero()` calls for cross-version PyTorch compatibility
- **README**: Updated to match PDF specification (hard/soft constraints, scoring formula, exponential weighting, getting started instructions)

---

## Dataset Terminology

| Dataset | Samples | Purpose | Available |
|---------|---------|---------|-----------|
| **Training** | 1M | Train your models | Yes (LiteTensorData/) |
| **Validation** | 100 | Tune and evaluate locally | Yes (LiteTensorDataTest/) |
| **Test** | 100 | Final contest ranking | Hidden (same format as validation) |

All datasets contain floorplans with **21 to 120 blocks** (partitions).

## Constraint Relaxations

The following constraints from the original FloorSet dataset are **relaxed** for this contest:

| Constraint | Status | Notes |
|------------|--------|-------|
| **Aspect Ratio** | ✅ Relaxed | Any width/height ratio allowed |
| **Fixed Outline** | ✅ Removed | No fixed canvas size; compactness is implicitly optimized via bounding box area and pin-to-block HPWL in the cost function. Note: **boundary constraints** (blocks touching the solution's bounding-box edges/corners) still apply as soft constraints. |
| **Coordinates** | ✅ Floating-point allowed | Integer coordinates not required |

**Hard Constraints** (violation = infeasible, score 10.0):
- **No overlaps** between blocks
- **Area tolerance**: Soft-block area (w × h) must be within **1%** of target area
- **Dimension immutability**: Fixed-shape and preplaced blocks must have **exact** target dimensions (w, h). Preplaced blocks must also have exact target positions (x, y).

**Soft Constraints** (violations penalized via exp(β·V_rel) in cost function):
- **Grouping**: Blocks in a group must abut (share an edge), forming a single connected component
- **MIB** (Multi-Instantiation Blocks): Blocks in a group must have identical dimensions
- **Boundary**: Block must touch specified bounding-box edge(s) or corner

  The boundary requirement is encoded as an integer bitmask in `constraints[:, 4]`:

  | Bit | Value | Meaning |
  |-----|-------|---------|
  | 0 | 1 | Left edge of bounding box |
  | 1 | 2 | Right edge of bounding box |
  | 2 | 4 | Top edge of bounding box |
  | 3 | 8 | Bottom edge of bounding box |

  Edge examples: `1`=left, `2`=right, `4`=top, `8`=bottom.
  Corner examples: `5`=top-left (4+1), `10`=bottom-right (8+2), `9`=bottom-left (8+1), `6`=top-right (4+2).
  A block violates its constraint if any required bit is not satisfied.

**Quality Metrics** (in cost function quality factor):
- Block-to-block HPWL (minimize wirelength)
- Pin-to-block HPWL (encourages placement near fixed pins)
- Bounding box area (encourages compact placement)

## Dataset Downloads

- **Training data (1M samples):** [FloorSet-Lite on Hugging Face](https://huggingface.co/datasets/IntelLabs/FloorSet)
- **Validation data (100 samples):** [FloorSet-Lite-Test on Hugging Face](https://huggingface.co/datasets/IntelLabs/FloorSet)

Place datasets in the FloorSet root directory (parent of `iccad2026contest/`):
- `FloorSet/LiteTensorData/` (training)
- `FloorSet/LiteTensorDataTest/` (validation)

This matches the default `data_path="../"` used by the evaluation scripts.

## PyTorch DataLoaders (Auto-Download)

The contest framework provides convenience functions in `iccad2026_evaluate.py` that **automatically download** data from Hugging Face:

```python
from iccad2026_evaluate import get_training_dataloader, get_validation_dataloader

# Training data (1M samples) - auto-downloads ~15GB on first use
train_loader = get_training_dataloader(batch_size=1, num_samples=1000)

# Validation data (100 samples) - auto-downloads ~15MB on first use
val_loader = get_validation_dataloader(batch_size=1)
```

**Functions:**
| Function | Dataset | Samples | Purpose |
|----------|---------|---------|---------|
| `get_training_dataloader()` | Training | 1M | Train ML models |
| `get_validation_dataloader()` | Validation | 100 | Local evaluation |

Both return standard PyTorch `DataLoader` objects.

---

## Getting Started

```bash
# 1. Clone FloorSet repository
git clone https://github.com/IntelLabs/FloorSet.git
cd FloorSet

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: source venv/bin/activate.csh  # csh/tcsh

# 3. Install dependencies
pip install -r iccad2026contest/requirements.txt

# 4. Download datasets (see links above)
#    Place in: FloorSet/LiteTensorData/ (training, 1M samples)
#              FloorSet/LiteTensorDataTest/ (validation, 100 samples)

# 5. Enter the contest directory (all commands below run from here)
cd iccad2026contest

# 6. Copy the template optimizer
cp optimizer_template.py my_optimizer.py

# 7. Implement your algorithm in my_optimizer.py (edit the solve() method)

# 8. Evaluate on validation set
python iccad2026_evaluate.py --evaluate my_optimizer.py

# 9. Evaluate single validation case (for debugging, 0-99)
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0

# 10. Validate before submission
python iccad2026_evaluate.py --validate my_optimizer.py
```

---

## Your Task

Implement `solve()` in your optimizer file:

```python
def solve(self, block_count, area_targets, b2b_connectivity, 
          p2b_connectivity, pins_pos, constraints):
    """
    Place blocks to minimize wirelength and area.
    
    Returns: List of (x, y, width, height) tuples, one per block
             - Floating-point coordinates allowed
             - Any aspect ratio allowed (w/h not constrained)
             - Area (w*h) must be within 1% of area_targets[i]
    """
    positions = []
    for i in range(block_count):
        x, y = 0.0, 0.0       # Your placement algorithm
        w = h = math.sqrt(area_targets[i])  # Square is simplest valid shape
        positions.append((x, y, w, h))
    return positions
```

**Hard Constraints** (violation = infeasible, score 10.0):
- No overlapping blocks
- Soft-block area (w × h) must be within 1% of target
- Fixed-shape and preplaced blocks must have exact target dimensions

**Soft Constraints** (penalized in cost function):
- Grouping, MIB, boundary (see above for details)

**Relaxed Constraints** (not enforced):
- Aspect ratio: Any width/height ratio is valid
- Coordinate precision: Floating-point values allowed

---

## Using Training Data (1M samples)

```bash
# See full example (run from iccad2026contest/)
python training_example.py
```

```python
from iccad2026_evaluate import get_training_dataloader, compute_training_loss_differentiable

dataloader = get_training_dataloader(batch_size=1, num_samples=10000)

for batch in dataloader:
    area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, fp_sol, metrics = batch
    
    # Squeeze batch dimension
    area_target = area_target.squeeze(0)
    b2b_conn = b2b_conn.squeeze(0)
    p2b_conn = p2b_conn.squeeze(0)
    pins_pos = pins_pos.squeeze(0)
    metrics = metrics.squeeze(0)
    
    block_count = int((area_target != -1).sum().item())
    
    # Your NN predicts positions: [N, 4] tensor of (x, y, w, h)
    positions = my_model(area_target, b2b_conn, p2b_conn, pins_pos, constraints)
    
    # DIFFERENTIABLE contest cost function
    # Same formula: Cost = (1 + α·(HPWL_gap + Area_gap)) × exp(β·V_soft)
    loss = compute_training_loss_differentiable(
        positions, b2b_conn, p2b_conn, pins_pos, 
        area_target[:block_count], metrics
    )
    loss.backward()  # Gradients flow!
```

**Differentiable loss includes:**
- HPWL gap (vs ground truth baseline)
- Area gap (vs ground truth baseline)
- Overlap violation (soft, differentiable)
- Area tolerance violation (soft, differentiable)

**Important assumptions:**
- **No model provided** - You must implement your own neural network
- **Placement constraints NOT included** - Fixed, preplaced, MIB, cluster, boundary constraints are not in the differentiable loss (but ARE checked in final evaluation)
- **Training proxy** - The differentiable loss approximates the contest score; final evaluation uses exact (non-differentiable) scoring
- **Ground truth as baseline** - Training uses `metrics` from training data; validation/test evaluation uses validation/test baselines

---

## Final Evaluation

Your submission will be evaluated on:
1. **Validation set (100 samples)** - Provided for local development (LiteTensorDataTest/)
2. **Hidden test set (100 samples)** - Same format, same block range (21-120), used for final ranking

---

## Saving and Re-scoring Solutions

```bash
# Run optimizer and save solutions to JSON (run from iccad2026contest/)
python iccad2026_evaluate.py --evaluate my_optimizer.py --save-solutions
# Output: my_optimizer_solutions.json

# Re-score saved solutions (without re-running optimizer)
python iccad2026_evaluate.py --score my_optimizer_solutions.json
```

This is useful for:
- Comparing scores after bug fixes
- Analyzing results without re-running slow optimizers

---

## Scoring

```
Cost = (1 + 0.5×(HPWL_gap + Area_gap)) × exp(2×V_rel) × max(0.7, RuntimeFactor^0.3)
     = 10.0 if infeasible (any hard constraint violated)
```

**Hard constraints** (any violation → infeasible, Cost = 10.0):
1. No block overlaps
2. Soft-block area within 1% of target
3. Fixed-shape block dimensions exactly match input (w, h)
4. Preplaced block position and dimensions exactly match input (x, y, w, h)

Where:
- **HPWL_gap, Area_gap**: Relative gaps vs. baseline, **clamped to zero from below** — solutions that beat the baseline receive no additional score bonus (quality_factor ≥ 1 always)
- **V_rel** ∈ [0, 1]: Normalized soft constraint violations (grouping, boundary, MIB only; fixed/preplaced are hard constraints and excluded)
- **RuntimeFactor**: Your runtime / median runtime of all submissions (per test case) — see Runtime Normalization below
- **max(0.7, ...)**: Speed benefit capped at 30%; slowness penalty is **uncapped**

**Lower score = better.** Final ranking uses **exponentially weighted** average across all 100 tests:

```
Total Score = Σ Cost[i] · e^{n_i/12} / Σ e^{n_j/12}
```

where n_i is the block count (21–120). The /12 scaling ensures every size from 21 to 120 carries non-zero weight while still strongly favouring larger instances:

| Weighting | n=120 alone | n=116–120 bucket | n=21–25 bucket |
|-----------|-------------|------------------|----------------|
| exp(n) (old) | 63% | 99% | 0% |
| exp(n/4) | 22% | 71% | 0% |
| **exp(n/12) (current)** | **34%** | **34%** | **0.01%** |

**Expected score range:**
- Perfect quality, no violations, **median runtime** → Cost = **1.00** → Total Score ≈ **1.0**
- Perfect quality, no violations, **≥3× faster** (capped) → Cost = **0.70** → Total Score ≈ **0.70**
- Infeasible solution → Cost = **10.0**

> **Note:** A perfect solution achieving baseline metrics on all test cases with median runtime has a Total Score of **1.0**. The theoretical minimum (maximum speed bonus, capped) is **0.70**. These values follow directly from substituting into the formula above.

### Runtime Normalization

**Official final evaluation** uses a **per-test-case cross-submission median**: for each of the 100 hidden test cases, the median runtime across *all* contestant submissions is computed independently, and every contestant's RuntimeFactor for that test case is relative to that per-case median.

**Local evaluator** cannot replicate this. The 100 validation cases span block counts from 21 to 120 — runtimes vary enormously by size. Using a mixed-size median as the denominator would give small cases an artificial speed bonus and large cases an artificial slowness penalty, regardless of how your solver actually compares on each size. Therefore the local evaluator sets **RuntimeFactor = 1.0** (neutral — assumes you are at the median) for every test case. Local costs reflect solution quality and soft-constraint violations only; runtime impact is assessed on the official leaderboard.

### Infeasibility Penalty (M = 10) and Feasible Cost Ceiling

Feasible cost is **capped at M − ε (just below 10.0)**, guaranteeing that any feasible solution — regardless of how poor the quality — always scores strictly better than an infeasible one:

```
feasible cost  = min(quality × violation × runtime,  9.999999)
infeasible cost = 10.0  (exactly)
```

This means the statement *"any feasible solution scores better than an infeasible one"* holds unconditionally.

### Exponential Weighting

The Total Score uses `exp(n_i/12)` weights, normalized. With block counts from 21 to 120:

| Bucket | Weight |
|--------|--------|
| n = 116–120 | 34% |
| n = 111–115 | 22% |
| n = 106–110 | 15% |
| n = 101–105 | 10% |
| n = 96–100 | 6% |
| n = 91–95 | 4% |
| n = 86–90 | 3% |
| n = 81–85 | 2% |
| n ≤ 80 | 4% |

The /12 scaling was chosen as the smallest divisor where every bucket from 21–120 carries non-zero weight, while still strongly rewarding scalability to large instances. The top bucket (116–120) accounts for ~34% of the total score.

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `--evaluate FILE` | Run optimizer on 100 test cases, compute score |
| `--score FILE` | Re-score saved solutions (without re-running optimizer) |
| `--validate FILE` | Check submission format before submitting |
| `--baseline` | Generate baseline metrics for the validation set |
| `--training` | Explore training data statistics |
| `--visualize` | Visualize a validation case (requires `--test-id N`) |
| `--test-id N` | Run on single test case (for debugging) |
| `--save-solutions` | Export positions to JSON (use with `--evaluate`) |
| `--info` | Show scoring formula |
