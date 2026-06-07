# DiT Floorplanning Workflow

## 1. Problem Analysis

### 1.1 Contest Task

Given floorplan instance inputs (block areas, connectivity, pin positions, constraints), predict block placements `(x, y, w, h)` that minimize:

```
Cost = (1 + 0.5 × (HPWL_gap + Area_gap)) × exp(2 × V_rel) × max(0.7, RuntimeFactor^0.3)
     = 10.0 if infeasible (overlap or dimension violation)
```

- **HPWL_gap**: (predicted HPWL - baseline HPWL) / baseline HPWL
- **Area_gap**: (predicted bbox area - baseline area) / baseline area
- **V_rel**: normalized soft constraint violations ∈ [0, 1]
- **RuntimeFactor**: T_submission / T_median (median over all submissions per test case)
- **Infeasible** → Cost = 10.0

### 1.2 Input Features

| Tensor | Shape | Description |
|--------|-------|-------------|
| `area_target` | `[n_blocks]` | Target area per block |
| `b2b_connectivity` | `[b2b_edges, 3]` | (block_i, block_j, weight) |
| `p2b_connectivity` | `[p2b_edges, 3]` | (pin_idx, block_idx, weight) |
| `pins_pos` | `[n_pins, 2]` | Pin (x, y) positions |
| `constraints` | `[n_blocks, 5]` | [fixed, preplaced, mib, cluster, boundary] |
| `target_positions` | `[n_blocks, 4]` | Target (x, y, w, h); -1 means "free" |

Block count ranges from **21 to 120**.

### 1.3 Constraint Encoding

```
constraints[:, 0] fixed       — 0/1 binary: block must have exact target (w, h)
constraints[:, 1] preplaced  — 0/1 binary: block must have exact target (x, y, w, h)
constraints[:, 2] mib_id      — 0 = no constraint, otherwise group ID (same-shape group)
constraints[:, 3] cluster_id  — 0 = no constraint, otherwise group ID (must form connected component)
constraints[:, 4] boundary   — bitmask: 1=left, 2=right, 4=TOP, 8=BOTTOM
                               5=TOP-LEFT(4+1), 6=TOP-RIGHT(4+2), 9=BOTTOM-LEFT(8+1), 10=BOTTOM-RIGHT(8+2)
```

### 1.4 target_positions 语义

| Block Type | target_positions[i] 内容 | solve() 要求 |
|-----------|--------------------------|-------------|
| Soft (free) | all -1 | output any valid (x,y,w,h) satisfying area tolerance |
| Fixed-shape | w, h = target, x=y=-1 | output (x,y,w,h) where w,h **exactly** match target |
| Preplaced | x, y, w, h all set | output **exactly** (x,y,w,h) matching all four values |

**Dimension immutability (hard constraint)**: For fixed-shape and preplaced blocks, any deviation from target dimensions (or location for preplaced) makes the solution infeasible (cost = 10.0).

### 1.5 V_rel 计算公式

```
V_rel = (V_boundary + V_grouping + V_mib) / N_soft
N_soft = |B_boundary| + Σ_p (|G_p| - 1) + Σ_q (|M_q| - 1)
```

- **Fixed-shape 和 preplaced 不参与 V_rel 计算**（它们是 hard constraint，违反就直接 infeasible）
- **V_boundary**: 不在边界上的 block 数量
- **V_grouping**: Σ(连通分量数 - 1) per cluster group（完全分离=violation最大，完全连接=0）
- **V_mib**: Σ(不同 (w,h) 形状数 - 1) per MIB group（全部相同=0，有差异=violation）

### 1.6 Soft vs Hard Constraint 区分

| Constraint Type | Category | If Violated |
|----------------|----------|-------------|
| Overlap | **Hard** | Infeasible (cost=10) |
| Area tolerance (soft blocks) | **Hard** | Infeasible (cost=10) |
| Dimension immutability (fixed/preplaced) | **Hard** | Infeasible (cost=10) |
| Boundary | Soft | exp(2×V_rel) penalty |
| Grouping (cluster) | Soft | exp(2×V_rel) penalty |
| MIB (same-shape) | Soft | exp(2×V_rel) penalty |

## 2. Model Architecture

### 2.1 Overall Design

Use a **Diffusion Transformer (DiT)** that takes the floorplan instance as conditioning and generates block positions.

**Model input**: `[n_blocks, D]` feature vector per block
**Conditioning**: global context (connectivity adjacency, pin locations, canvas stats) + target_positions for fixed/preplaced
**Output**: `[n_blocks, 4]` tensor of `(x, y, w, h)` — denoised from noisy initial guess

### 2.2 Feature Encoding

```python
# Per-block features [n_blocks, 12]
block_feat = {
    'log_area':        log(area_target + 1e-6),              # [N]
    'fixed':           constraints[:,0],                      # [N] binary
    'preplaced':       constraints[:,1],                        # [N] binary
    'mib_id':          constraints[:,2],                        # [N] (0 = none)
    'cluster_id':      constraints[:,3],                       # [N] (0 = none)
    'boundary_code':   constraints[:,4],                        # [N]
    'w_target':        target_positions[:,2].clip(0),          # [N] (0 if free)
    'h_target':        target_positions[:,3].clip(0),           # [N]
    'x_target':        target_positions[:,0].clip(0),           # [N]
    'y_target':        target_positions[:,1].clip(0),            # [N]
    'is_fixed_or_preplaced': (constraints[:,0] + constraints[:,1] > 0),  # [N]
    'active':          (area_target > 0).float(),              # [N] padding mask
}
```

**Key design**: Fixed/preplaced blocks have their target dimensions embedded in the feature vector. The model should learn to "reproduce" these dimensions faithfully (dimension immutability is enforced by post-processing snapping, not soft loss).

**Global features**:
- `n_blocks`: scalar
- `total_area`: sum of area_target
- `n_b2b_edges`, `n_p2b_edges`, `n_pins`: scalar stats
- `b2b_conn`: edge list for graph attention
- `p2b_conn`: edge list for pin attachment
- `pins_pos`: [n_pins, 2] pin coordinates

### 2.3 DiT Architecture

```
Input: [B, N, D] block features
        ↓
Project to [B, N, d_model] with linear
        ↓
Add: block_idx positional encoding + target_position embedding
        ↓
Process through DiT blocks:
  - Self-attention over blocks (N×N attention)
  - Cross-attention to global conditioning (optional)
  - Feed-forward MLP
  - adaLN conditioning (scale+shift from timestep and global features)
        ↓
Output head: linear → [B, N, 4] → (x, y, w, h)
```

**Key modifications from standard DiT**:
- Global conditioning via **conditional normalization** (adaLN) — timestep embedding AND global features modulate the attention/FFN
- Fixed/preplaced blocks: target (w,h,x,y) encoded as part of block features; model should learn to preserve these
- Variable sequence length (different N per sample) — pad to `max_N=120`
- Output head: linear, no activation (unbounded continuous values)

### 2.4 Denoising Diffusion Formulation

- **Forward process**: `x_t = α_bar_t × x_0 + (1-α_bar_t) × ε`, where x_0 is ground truth placement
- **Reverse process**: DiT predicts noise ε_θ(x_t, t, cond) or directly predicts x_0
- **Training loss**: `||ε_θ(x_t, t, cond) - ε||²` (noise prediction) or direct MSE on x_0
- **Sampler**: DDIM or DPM-Solver for fast inference (20–50 steps)

### 2.5 Handling Variable Block Count

Use a **固定最大长度 padding 方案**:
- Pad all block features to `max_N = 120`
- Mark padded positions with `area_target = 0` / `active = 0`
- Attention masks ensure padded blocks don't contribute to loss or output
- Loss computed only on `active == 1` blocks

## 3. Training Pipeline

### 3.1 Data Loading

```python
from iccad2026_evaluate import get_training_dataloader
train_loader = get_training_dataloader(batch_size=32, num_samples=100000, shuffle=False)
# Returns 8 tensors: area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, fp_sol, metrics
```

Each batch: 8 tensors, all padded to the largest tensor in the batch.

### 3.2 Ground Truth

- `fp_sol`: `[n_blocks, 4]` = **(w, h, x, y)** in ground truth order → reorder to `(x, y, w, h)` for loss computation
- `tree_sol`: B*Tree representation — not directly usable as regression target, but can be converted to (x,y,w,h) for supervision
- `metrics`: `[area, num_pins, ..., b2b_wl, p2b_wl]` — for differentiable loss baseline

### 3.3 Differentiable Contest Cost

```python
from iccad2026_evaluate import compute_training_loss_differentiable
loss = compute_training_loss_differentiable(
    positions,      # [N, 4] predicted (x,y,w,h)
    b2b_conn, p2b_conn, pins_pos, area_target, metrics
)
loss.backward()
```

This implements the contest cost formula in differentiable form for end-to-end training.

### 3.4 Training Loop

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

for step, batch in enumerate(train_loader):
    (area_target, b2b_conn, p2b_conn, pins_pos,
     constraints, tree_sol, fp_sol, metrics) = batch

    # Ground truth: fp_sol [N, 4] = (w, h, x, y) → [N, 4] = (x, y, w, h)
    gt = torch.stack([fp_sol[:,2], fp_sol[:,3], fp_sol[:,0], fp_sol[:,1]], dim=1)

    # Active mask
    active = (area_target[:, :, 0] != -1).float()  # [B, N]

    # Sample timestep t ~ Uniform({1,...,T-1})
    t = torch.randint(1, T, (B,), device=device)

    # Add noise
    noise = torch.randn_like(gt)
    alpha_bar_t = cosine_beta_schedule(t / T).view(B, 1, 1)
    noisy_gt = alpha_bar_t * gt + (1 - alpha_bar_t) * noise

    # Prepare features
    block_feat = prepare_block_features(area_target, constraints, target_positions)
    global_feat = prepare_global_features(area_target, b2b_conn, p2b_conn, pins_pos)

    # Forward pass
    pred = model(noisy_gt, t, block_feat, global_feat)  # predicts noise or x0

    # Loss on active blocks only
    loss = ((pred - gt) ** 2 * active.unsqueeze(-1)).sum() / active.sum()
    # OR: use differentiable contest cost
    # loss = compute_training_loss_differentiable(pred, b2b_conn, p2b_conn, pins_pos, area_target, metrics)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
```

### 3.5 Cosine Beta Schedule

```python
def cosine_beta_schedule(t, T=1000):
    """Cosine schedule as in improved DDPM."""
    alpha_bar_t = torch.cos(t / T * math.pi / 2) ** 2
    return alpha_bar_t
```

## 4. Inference

### 4.1 Optimizer Wrapper

```python
from iccad2026_evaluate import FloorplanOptimizer

class DiTOptimizer(FloorplanOptimizer):
    def __init__(self, model_path, device='cuda'):
        super().__init__(verbose=False)
        self.device = device
        self.model = load_model(model_path).to(device)
        self.model.eval()

    def solve(self, block_count, area_targets, b2b_conn, p2b_conn,
              pins_pos, constraints, target_positions):
        # Prepare input features
        x = self._prepare_input(block_count, area_targets, ...)
        # Denoise (DDIM sampling, 30-50 steps)
        with torch.no_grad():
            positions = self.model.sample(x, steps=50)
        # Post-processing (hard constraint enforcement)
        positions = self.post_process(positions, constraints, target_positions, area_targets)
        return positions
```

### 4.2 Post-Processing (Hard Constraint Enforcement)

**CRITICAL**: DiT output must be post-processed to strictly satisfy hard constraints:

```python
def post_process(positions, constraints, target_positions, area_targets):
    """
    positions: List of (x, y, w, h) from DiT
    Returns: positions where hard constraints are satisfied
    """
    n = len(positions)

    # Step 1: Snap fixed/preplaced blocks to exact target dimensions/positions
    for i in range(n):
        tp = target_positions[i]
        if tp[2] != -1 and tp[3] != -1:  # has target w, h
            x, y, w, h = positions[i]
            # Fixed-shape: preserve position, snap dimensions
            if constraints[i, 1] == 0:  # not preplaced
                positions[i] = (x, y, float(tp[2]), float(tp[3]))
            # Preplaced: snap everything
            if constraints[i, 1] != 0:  # preplaced
                positions[i] = (float(tp[0]), float(tp[1]), float(tp[2]), float(tp[3]))

    # Step 2: Rescale soft blocks to satisfy 1% area tolerance
    for i in range(n):
        tp = target_positions[i]
        if tp[2] == -1:  # soft block (no fixed dimensions)
            x, y, w, h = positions[i]
            target_area = float(area_targets[i])
            actual_area = w * h
            rel_error = abs(actual_area - target_area) / target_area
            if rel_error > 0.01:
                # Rescale to match target area, preserve aspect ratio
                scale = math.sqrt(target_area / actual_area)
                positions[i] = (x, y, w * scale, h * scale)

    # Step 3: Remove overlaps via constrained optimization
    positions = remove_overlaps_conservative(positions, constraints, target_positions)

    return positions

def remove_overlaps_conservative(positions, constraints, target_positions):
    """
    Remove overlaps WITHOUT disturbing fixed/preplaced blocks or changing their dimensions.
    Only adjusts soft blocks.
    """
    n = len(positions)
    fixed_or_preplaced = set()
    for i in range(n):
        if constraints[i, 0] != 0 or constraints[i, 1] != 0:
            fixed_or_preplaced.add(i)

    # Iterative force-directed relaxation (only for non-fixed blocks)
    for iteration in range(200):
        overlaps_found = False
        for i in range(n):
            for j in range(i + 1, n):
                if i in fixed_or_preplaced or j in fixed_or_preplaced:
                    continue
                x1, y1, w1, h1 = positions[i]
                x2, y2, w2, h2 = positions[j]
                # Check overlap
                ox = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                oy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
                if ox > 1e-6 and oy > 1e-6:
                    overlaps_found = True
                    # Push blocks apart (along connecting line)
                    cx1, cy1 = x1 + w1/2, y1 + h1/2
                    cx2, cy2 = x2 + w2/2, y2 + h2/2
                    dx, dy = cx2 - cx1, cy2 - cy1
                    dist = math.sqrt(dx*dx + dy*dy) + 1e-6
                    push = (min(w1, h1, w2, h2) * 0.1) + 1.0
                    if cx1 < cx2:
                        positions[i] = (positions[i][0] - push, positions[i][1], w1, h1)
                        positions[j] = (positions[j][0] + push, positions[j][1], w2, h2)
                    else:
                        positions[i] = (positions[i][0] + push, positions[i][1], w1, h1)
                        positions[j] = (positions[j][0] - push, positions[j][1], w2, h2)
        if not overlaps_found:
            break

    return positions
```

### 4.3 Evaluation

```bash
cd iccad2026contest
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0   # Single case
python iccad2026_evaluate.py --evaluate my_optimizer.py                # Full eval (100 cases)
python iccad2026_evaluate.py --validate my_optimizer.py                 # Format check
```

## 5. Key Design Decisions

### 5.1 Why DiT for Floorplanning

- **Continuous output**: Block positions are continuous — well-suited for diffusion
- **Conditional generation**: Block areas, connectivity, and fixed/preplaced target dimensions are natural conditioning signals
- **Handles variable N**: Padding + attention masking handles 21–120 blocks
- **Differentiable loss available**: `compute_training_loss_differentiable` enables end-to-end training with the exact contest cost formula

### 5.2 Two-Stage vs End-to-End

| Approach | Pros | Cons |
|----------|------|------|
| **Two-stage (learn + post-process)** | Hard constraints guaranteed; model only learns soft objectives | Post-processing may degrade soft constraint satisfaction |
| **End-to-end (hard constraints in loss)** | Unified optimization | Complex hard constraints (overlap) not easily differentiable; infeasible solutions possible |

**Recommendation**: Two-stage approach with conservative post-processing (only modifies soft blocks for overlap removal, respects fixed/preplaced exactly).

### 5.3 Handling Fixed/Preplaced Blocks

1. **During training**: Encode target dimensions in block features; model learns to reproduce these (via supervision on fp_sol which has correct dimensions for all blocks)
2. **During inference**: Post-processing snaps fixed/preplaced to exact target values — this is a hard constraint, not a soft one

### 5.4 Boundary Constraint Implementation

```python
def check_boundary_violation(positions, constraints, block_idx):
    """Check if block touches its required boundary."""
    code = int(constraints[block_idx, 4])
    if code == 0:
        return False
    x, y, w, h = positions[block_idx]
    # Compute bounding box
    x_min = min(p[0] for p in positions)
    y_min = min(p[1] for p in positions)
    x_max = max(p[0] + p[2] for p in positions)
    y_max = max(p[1] + p[3] for p in positions)
    eps = 1e-6
    touches = {
        1: abs(x - x_min) < eps,           # LEFT edge
        2: abs(x + w - x_max) < eps,       # RIGHT edge
        4: abs(y + h - y_max) < eps,       # TOP edge
        8: abs(y - y_min) < eps,           # BOTTOM edge
    }
    return not all(touches[bit] for bit in (1, 2, 4, 8) if code & bit)
```

### 5.5 Grouping (Cluster) Constraint Implementation

```python
from shapely.ops import unary_union
from shapely.geometry import box

def check_grouping_violation(positions, constraints, cluster_id):
    """Check if blocks in a cluster form a single connected component."""
    group_indices = [i for i in range(len(constraints))
                     if constraints[i, 3] == cluster_id]
    group_polys = [box(*pos) for pos in group_indices]
    union = unary_union(group_polys)
    if union.geom_type == 'MultiPolygon':
        return len(union.geoms) - 1  # violation = num_components - 1
    return 0
```

### 5.6 MIB (Multi-Instantiation Block) Constraint Implementation

```python
def check_mib_violation(positions, constraints, mib_id):
    """Check if blocks in an MIB group have identical dimensions."""
    group_indices = [i for i in range(len(constraints))
                     if constraints[i, 2] == mib_id]
    shapes = set()
    for i in group_indices:
        w, h = round(positions[i][2], 4), round(positions[i][3], 4)
        shapes.add((w, h))
    return len(shapes) - 1  # violation = distinct_shapes - 1
```

## 6. Files to Create

```
iccad2026contest/
├── dit_model.py          # DiT model definition
├── dit_train.py          # Training script
├── dit_optimizer.py      # Inference wrapper (FloorplanOptimizer subclass)
└── workflow.md          # This document
```

## 7. Quick Start

```bash
# 1. Install dependencies
pip install -r iccad2026contest/requirements.txt

# 2. Download data (auto-downloads on first use)
python iccad2026_evaluate.py --training

# 3. Train model
cd iccad2026contest
python dit_train.py

# 4. Create optimizer
cp optimizer_template.py my_optimizer.py
# Implement DiTOptimizer in my_optimizer.py

# 5. Evaluate
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0
python iccad2026_evaluate.py --evaluate my_optimizer.py
```

## 8. Important Notes

- **Ground truth order**: `fp_sol` is `[w, h, x, y]`, NOT `(x, y, w, h)`. Always reorder when using as training supervision.
- **Fixed/preplaced are hard**: Any dimension mismatch → infeasible (cost=10). Never treat them as soft.
- **Runtime factor**: Speedup benefit capped at 30% (`max(0.7, ...)`), but slowness penalty is uncapped.
- **Contest cost = 10 for infeasible**: Overlap, area tolerance violation (>1% for soft blocks), or dimension mismatch (fixed/preplaced).
- **Soft constraints are soft**: Boundary, grouping, MIB violations only affect `exp(2×V_rel)`, not hard infeasibility.