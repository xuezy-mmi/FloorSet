# DiT 模型技术路线报告

基于 ICCAD 2026 FloorSet 竞赛的 DiT（Diffusion Transformer）

---

## 第一部分：DiT 模型架构

### 1.1 问题背景与建模

芯片版图规划（Floorplanning）的目标是为数十到上百个矩形模块在二维平面上分配合适的位置和尺寸，使得：

1. **线长（Wirelength）最小化**：块间连接和引脚引线总长度最短
2. **包围盒面积（Bounding Box Area）最小化**：所有模块整体占据面积最小
3. **硬约束满足**：
   - **固定块（Fixed-shape）**：尺寸 `(w, h)` 必须与目标完全一致
   - **预置块（Preplaced）**：位置 `(x, y)` 和尺寸 `(w, h)` 必须与目标完全一致
   - **软块（Soft blocks）**：面积 `|w×h - target| / target ≤ 1%`
4. **软约束尽量满足**（影响评分指数因子，不满足不导致 infeasible）：
   - **MIB**（Multi-Instance Block）：同组块形状一致
   - **Cluster**（簇）：同簇块必须两两相接
   - **Boundary**（边界）：必须贴指定边 / 角

我们将每个模块的布局建模为 4 维张量 **`[w, h, x, y]`**，并将其视为去噪扩散概率模型（DDPM）扩散过程中的"干净图像"，通过去噪网络学习从噪声中恢复合法布局。

由于每个 IC block 都处于一个图中（节点 = block，边 = b2b 连线 / p2b 引脚），
模型在 forward 时同时输入当前 noisy 布局、时间步 `t` 和该图的拓扑特征，
称为**条件扩散**（Conditional Diffusion）。

### 1.2 扩散模型基础

#### 1.2.1 DDPM（去噪扩散概率模型）

DDPM 通过两步过程工作：

**前向过程（Forward Process）**：向真实布局 `x₀` 中逐步添加高斯噪声，经过 T 步得到 `x_T`：

```
x_t = √(ᾱ_t) · x₀ + √(1 - ᾱ_t) · ε,   ε ~ N(0, I)
```

**反向过程（Reverse Process）**：训练一个神经网络 `ε_θ(x_t, t, c)` 预测噪声，从而恢复干净布局：

```
x₀_hat = (x_t - √(1 - ᾱ_t) · ε_θ) / √ᾱ_t
```

#### 1.2.2 DDIM（确定性采样加速）

完整 DDPM 需要 1000 步采样，竞赛实时评测无法接受。DDIM 将采样步数压缩到 50 步：

```
x_{t-1} = √α_{t-1} · x₀_hat + √(1 - α_{t-1}) · ε_θ(x_t, t, c)
```

在保持生成质量的同时将推理速度提升约 20 倍。

#### 1.2.3 Cosine Schedule（余弦噪声调度）

采用 Nichol & Dhariwal (2021) 提出的余弦调度，相比线性调度在细粒度结构（模块位置/尺寸）上提供更平滑的噪声衰减：

```
β_t = 1 - ᾱ_t / ᾱ_{t-1},   ᾱ_t = cos²((t + s)/(1 + s) · π/2)
```

其中 `s = 0.008` 为平滑超参数。`dit_utils_v3.CosineSchedule` 实现了完整的余弦调度（`alpha_cumprod` 用于 forward 采样与 reverse DDIM 步进）。

### 1.3 模型架构详解（`dit_model_v3.DiffusionTransformer`）

#### 1.3.1 整体结构

```
输入: x_t [B, N, 4] (z-score 标准化布局) + 条件信息
  ↓
aggregate_graph_features: 将条件聚合成 [B, N, cond_in] 每块特征（默认 8 通道 / 扩展 12 通道）
  ↓
输入投影: Linear(4 + cond_in → dim) + LayerNorm → tokens [B, N, dim]
  ↓
残差条件: tokens += cond_proj(gf)   (Linear → SiLU → Linear)
  ↓
时间嵌入: tokens += time_mlp(sinusoid(t))   (dim → 4*dim → dim)
  ↓
Pre-LN Transformer × depth 层（默认 depth=6, dim=512, heads=8）
   每层含 edge-bias 自注意力 + FFN
  ↓
输出投影: Linear(dim → dim) + GELU + LayerNorm + Linear(dim → 4)
  ↓
输出: pred_noise [B, N, 4] (z-score 空间, 已被 valid mask 屏蔽)
```

**Masking**：`aggregate_graph_features` 对填充的 -1 area 会把 feature 清零；模型
在 `forward()` 内还用 `(area_target != -1)` 重新 mask 一次，保证填充位置的
logits 不参与注意力（用 `0 *`，再在 attn 输出加上）。

#### 1.3.2 条件特征通道

`dit_utils_v3.aggregate_graph_features` 提供两种配置：

**基础版 (8 通道)**：用于原始 v3 Lite 训练

| 通道 | 含义 | 作用 |
|------|------|------|
| 0 | `log(area)` | 面积大小，基础尺度信息 |
| 1 | `b2b_w` | 块间连线权重之和（度中心性） |
| 2 | `b2b_d` | 块间连线数量（度数） |
| 3 | `p2b_w` | 引脚连线权重之和 |
| 4 | `p2b_px` | 引脚 x 坐标加权平均 |
| 5 | `p2b_py` | 引脚 y 坐标加权平均 |
| 6 | `is_hard` | 是否硬约束（fixed/preplaced） |
| 7 | `boundary` | 边界约束位掩码（归一化到 [0,1]） |

**扩展版 (12 通道)**：`train_dit_v3_combine.py` 用 `cond_in=12, extended_features=True`
打开。除上述 8 通道外，再叠加 4 个软约束指示特征：

| 通道 | 含义 |
|------|------|
| 8 | `has_mib` — 该块是否属于 MIB 组（mib_id > 0） |
| 9 | `has_cluster` — 该块是否属于 Cluster 组（cluster_id > 0） |
| 10 | `mib_size_norm` — MIB 组大小归一化 |
| 11 | `cluster_size_norm` — Cluster 组大小归一化 |

扩展特征仅在 `aggregate_graph_features_ext` 里生效，对 `dit_optimizer_v3.MyOptimizer`
加载 checkpoint 时，optimizer 自动检测 `'dataset' in ('lite+prime', 'prime')` 并切
到 12 通道推理。

#### 1.3.3 Edge-Bias 自注意力（`dit_model_v3._attn_with_edge_bias`）

**创新点**：把 b2b 连线权重视为注意力偏置，让模型直接看到拓扑耦合度。

实现方式：

1. 从 `b2b_conn [B, E, 3]` 构建 `[B, N, N]` 边权重矩阵 `edge_w`
2. 通过 `Linear(1, 1, bias=False)` 投影为单标量偏置；权重初始为 0，所以训练初期
   不引入扰动
3. 将偏置广播到所有头并 reshape 为 `(B*H, N, N)`，作为 `attn_mask` 传给
   `self_attn(...)`
4. 在 `_attn_with_edge_bias` 里逐层手动执行 Pre-LN 的 norm + 自注意 + FFN，而不是
   直接调用整层 `TransformerEncoder`，从而能插入上面的 mask

**效果**：在注意力 logit 上，连线紧密的块之间有更高的相互响应，模型能直接学到
"这两个块应该离得近"的拓扑偏好。

#### 1.3.4 Pre-LN Transformer

标准 Transformer 的结构是 `Attention → LayerNorm → Residual`，而 Pre-LN
（Pre-LayerNorm）为：

```
src  = LayerNorm(x)
attn_out = Attention(src, src, src)
x = x + attn_out
src2 = LayerNorm(x)
ff_out = FFN(src2)
x = x + ff_out
```

优势：

- 训练更稳定，适合 6 层以上的深度网络
- 梯度在各层间更均匀地流动

代码中通过 `nn.TransformerEncoderLayer(..., norm_first=True)` 配合手动逐层
forward 实现。

#### 1.3.5 创新总结

| 创新点 | 原始 DiT | 本 DiT v3 |
|--------|----------|-----------|
| 注意力机制 | 标准自注意力 | **Edge-bias 自注意力**（融入 b2b 连线权重） |
| 条件编码 | 单通道 log(area) | **8 通道图特征**（含硬约束、边界、拓扑） |
| 扩展条件（Lite+Prime 训练） | — | **+4 通道软约束指示**（has_mib / has_cluster / 组大小） |
| Transformer 归一化 | Post-LN | **Pre-LN**（更稳定） |
| 输入/输出投影 | 单一 Linear | **Linear + LayerNorm** |
| 噪声调度 | 线性 β | **Cosine Schedule** |
| 推理加速 | DDPM 1000 步 | **DDIM 50 步**（~20× 提速） |
| 竞赛感知 | 无 | **Z-score + 有效掩码 + 硬约束覆盖 + 可微损失** |
| 多数据集训练 | 单数据集 | **Lite / Prime / Combined**（combine 加载外部 `_prime_collate_for_v3`） |

---

## 第二部分：训练技术路线

### 2.1 数据集支持

#### 2.1.1 Lite 数据集（FloorSet-Lite，矩形布局）

- **路径**：`/home/xzy/eda/FloorSet/LiteTensorData/`
- **规模**：1M 训练样本，100 个 worker 分片（`worker_X/layouts_Y.th`）
- **格式**：`fp_sol` 为 `[w, h, x, y]` 张量（直接可用）
- **加载**：通过 `iccad2026_evaluate.get_training_dataloader()`，
  Lite `__getitem__` 返回 7 张量，8-tensor batch（`(area, b2b, p2b, pins, constraints, tree_sol, fp_sol, metrics)`）

### 2.2 训练稳定性技术

#### 2.2.1 EMA（指数移动平均）

```python
with torch.no_grad():
    for p_em, p_m in zip(ema.parameters(), model.parameters()):
        p_em.mul_(0.999).add_(p_m, alpha=0.001)
```

- 衰减率 0.999（`momentum = 1 - 0.999 = 0.001`）
- 推理时使用 EMA 权重，平滑训练末期的参数振荡，提升稳定性

#### 2.2.2 梯度裁剪

`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)`
（默认 `grad_clip=0.5`）防止梯度爆炸。

#### 2.2.3 NaN / Inf 检测

每个 batch 检测 loss 是否为 NaN/Inf；若是，则跳过该步的 optimizer 更新，
避免污染 EMA 与梯度直方图。

#### 2.2.4 Mixed Precision（可选）

GPU 环境下可启用 `torch.amp.autocast('cuda', dtype=torch.bfloat16)` 与
`torch.amp.GradScaler('cuda', ...)`。已知陷阱（v3 中已修复）：

- `masked_fill(..., -1e9)` 在 fp16 下会溢出；统一改用 `torch.finfo(scores.dtype).min`
- `scatter_add_` 在 fp16 源张量（Prime 文件以 fp16 存储）上会因 dtype 不匹配报错，
  因此联合训练 collate 把所有张量 cast 到 float32

### 2.3 Loss 定义详解（四分量结构，混合扩散 + 可微评分代理）

完整损失：

```python
loss = lambda_noise * MSE(pred_noise, ε)         # 扩散动态
     + lambda_diff  * vectorized_diff_loss(...)  # 竞赛评分可微代理
     + lambda_hard  * hard_violation_loss(...)   # 硬约束保证
     + lambda_soft  * compute_soft_loss(...)     # 软约束可微代理
```

#### 2.3.1 优先级 1：`λ_hard × hard_violation_loss`（默认 λ=0.1）

**作用**：保证**可行性**——使模型不产生 infeasible 解。

`hard_violation_loss(positions, target_positions, constraints, area_targets)`:

```python
# Fixed blocks: (w, h) 偏离目标 → MSE
fixed = constraints[:, 0] > 0
loss += ((w[fixed]   - target[fixed, 2]) ** 2 +
         (h[fixed]   - target[fixed, 3]) ** 2).mean()

# Preplaced blocks: (x, y, w, h) 全偏离 → MSE
pp = constraints[:, 1] > 0
loss += ((x[pp] - target[pp, 0]) ** 2 +
         (y[pp] - target[pp, 1]) ** 2).mean()
loss += ((w[pp] - target[pp, 2]) ** 2 +
         (h[pp] - target[pp, 3]) ** 2).mean()

# Free blocks area tolerance: relu(|w*h - a|/a - 0.01)
free = (~fixed) & (~pp) & (area > 0)
rel = (w[free]*h[free] - area[free]).abs() / (area[free] + 1e-6)
loss += 10 * relu(rel - 0.01).mean()
```

**Loss原理**：当全部硬约束满足时此 loss 为 0；任一违反都产生正梯度，把对应变量
拉回到合法值。

#### 2.3.2 优先级 2：`λ_diff × vectorized_diff_loss`（默认 λ=1.0）

**作用**：与竞赛评分公式完全对齐的可微代理。

`vectorized_diff_loss(positions, b2b, p2b, pins, area_targets, baseline_metrics)`:

```python
# HPWL 块间 + 引脚
hpwl_b2b = Σ w_ij * (|cx_i - cx_j| + |cy_i - cy_j|)
hpwl_p2b = Σ w_k  * (|cx_bk - px_k| + |cy_bk - py_k|)
hpwl_total = hpwl_b2b + hpwl_p2b

# BBox
bbox_area = (x.max() - x.min()) * (y.max() - y.min())

# Overlap (upper-triangle pairwise)
overlap_area = Σ_{i<j} relu(ox)*relu(oy)
overlap_v    = overlap_area / Σ(w*h)

# Area tolerance
area_v = Σ relu(|w*h - a|/a - 0.01) / N_free

V_soft = clamp(overlap_v + area_v, max=5.0)

hpwl_gap = relu((hpwl_total - baseline_hpwl) / baseline_hpwl)   ≥ 0
area_gap = relu((bbox_area - baseline_area) / baseline_area)   ≥ 0

return (1 + 0.5 * (hpwl_gap + area_gap)) * exp(2 * V_soft)
```

所有不可微操作都改用 ReLU / clamp / 上三角掩码近似，保证整条链路对 `positions`
可微。

#### 2.3.3 优先级 3：`λ_soft × compute_soft_loss`（默认 λ=0.1，含可微边界/MIB/分组）

**作用**：把竞赛的 boundary / grouping / MIB 三类软约束变成可微代理，
从而让梯度能往这些方向走（不能用 Shapely 求连通分量，所以用代理实现）：

```python
boundary_soft_loss(positions, constraints)
    # 对每条边界位 (1=L, 2=R, 4=T, 8=B)，把块的对应 edge 拉到 chip bbox 的对应边
    # penalty = (xi - 0)^2 (L)  +  (xi - (x_max - wi))^2 (R) + ...
    # normalize by chip size  → scale invariant

mib_soft_loss(positions, constraints)
    # 对每个 mib_id > 0 的组，惩罚组内 (w, h) 的离散度：
    # pen = Σ_g  std(w_g) / mean(w_g)  +  Σ_g std(h_g) / mean(h_g)

grouping_soft_loss(positions, constraints)
    # 对每个 cluster_id > 0 的组，惩罚组内两两块的 x_gap + y_gap
    # （离散 CC 的可微替代：gap=0 ⇔ 相接）
```

`compute_soft_loss` 是三者的加权和（`lambda_boundary / lambda_mib / lambda_grouping`
各 1.0）。

#### 2.3.4 优先级 4：`λ_noise × loss_noise`（默认 λ=1.0）

**作用**：标准 DDPM 噪声预测损失；维持扩散动力学正确性。

```python
loss_noise = MSE(pred_noise, true_noise)
```

#### 2.3.5 完整 Loss 公式

```python
loss = 1.0  * loss_noise                              # 扩散动态
     + 1.0  * vectorized_diff_loss(...)               # 评分代理
     + 0.1  * hard_violation_loss(...)                # 可行性
     + 0.1  * compute_soft_loss(...)                  # 软约束
```

### 2.4 模型收敛的三个阶段

#### 阶段 1：可行性优先（训练早期）

`hard_violation_loss` 梯度最强，模型首先学会：

- Fixed 块的 `(w, h)` 匹配目标
- Preplaced 块的 `(x, y, w, h)` 完全匹配目标
- Soft 块的面积在 1% 容限内

此阶段可能布局质量极差，但**保证不产 infeasible 解**。

#### 阶段 2：评分收敛（训练中期）

`vectorized_diff_loss` 主导，模型学习：

- 块位置靠近 GT、降低 HPWL
- 缩小包围盒面积
- 减少重叠

#### 阶段 3：软约束 + 细节精修（训练后期）

`compute_soft_loss` 与低噪声步的 `loss_noise` 帮助模型细化：
- MIB 同组形状收敛
- Cluster 内块贴近
- Boundary 块对齐边缘

### 2.5 训练流程总览

```
数据加载 (lite / prime / combined)
   ↓
Z-Score 标准化 (μ, σ from 64 batches)
   ↓
前向扩散: x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
   ↓
模型预测: ε_θ(x_t, t, c) → pred_noise (z-score space)
   ↓
四分量 Loss 计算
   ↓
反向传播 + 梯度裁剪
   ↓
参数更新 + EMA 更新 (decay=0.999)
   ↓
每 log_interval 打印 loss_noise / diff / hard / soft
   ↓
每 epoch 末保存: model_state_dict, ema_state_dict, norm_stats
   ↓
最终模型 →  diffusion_final.pth
```

### 2.6 关键技术特点总结

| 技术 | 作用 | 实现位置 |
|------|------|---------|
| **Cosine Schedule** | 更平滑的噪声衰减 | `dit_utils_v3.CosineSchedule` |
| **EMA (0.999)** | 推理稳定性 | trainer per-step `p_em = 0.999·p_em + 0.001·p_m` |
| **Z-Score 标准化** | 数值稳定性 | `x_norm = (x - μ) / σ`；σ 下限 1.0 |
| **有效掩码** | 处理变长序列 | `mask = (area != -1)`，输入/输出都乘 |
| **Edge-Bias 自注意力** | 注入 b2b 拓扑 | `diff.v3._attn_with_edge_bias` |
| **8 / 12 通道条件** | 块间图特征 | `aggregate_graph_features{, _ext}` |
| **DDIM 50 步推理** | 实时性 | `dit_optimizer_v3._ddim_sample` |
| **硬约束可微惩罚** | 训练期就保证可行性 | `hard_violation_loss` |
| **评分可微代理** | 与竞赛公式对齐 | `vectorized_diff_loss` |
| **软约束可微代理** | boundary / MIB / grouping | `compute_soft_loss` |
| **多数据集训练** | Lite+Prime 联合 | `train_dit_v3_combine.py` |

### 2.7 与原始 `compute_training_loss_differentiable` 的区别

| 维度 | ICCAD 官方函数 | 本训练方案 |
|------|---------------|-----------|
| 硬约束惩罚 | ❌ 无 | ✅ `hard_violation_loss` λ=0.1 |
| 评分公式代理 | ✅ | ✅ `vectorized_diff_loss` |
| 边界/MIB/分组可微 | ❌ 无 | ✅ `compute_soft_loss` |
| DDPM 动态 | ❌ 无 | ✅ `loss_noise` λ=1.0 |
| EMA | ❌ 无 | ✅ 0.999 衰减 |
| Cosine Schedule | ❌ 线性 | ✅ 余弦 |
| 多数据集 | ❌ 仅 lite | ✅ Lite + Prime + Combined |
| 扩展 12 通道条件 | ❌ | ✅ `aggregate_graph_features_ext` |

**核心区别**：ICCAD 官方函数只覆盖评分公式的可微版本，缺乏**硬约束保证**、
**边界/MIB/分组可微代理**、**DDPM 扩散动态**。本方案在此基础上构建完整监督体系：
"先学会不犯法（hard），再学会做好（diff），再学会做对（soft），最后学会去噪（noise）"。

---

## 第三部分：推理 + 可微后处理

### 3.1 推理管线（`dit_optimizer_v3.MyOptimizer`）

```
1. 加载 DiT EMA checkpoint（v3 / v3_combine / v3_prime 之一）
2. DDIM 50 步采样得到 (w_dit, h_dit) 初始值
3. 识别 locked 块（preplaced + fixed-shape）
4. 初始化宽度/高度
   - locked (preplaced + fixed)：使用 target (w, h)
   - free 块：取 √area（避免 DiT 输出导致负坐标，几何上更稳健）
5. B*-tree SA（locked-block aware，1.5s 预算）优化连线长度
6. post-processing（11 步强制可行性）→ 返回 (x, y, w, h) 列表
```

### 3.2 强制可行的 11 步 post-processing

```
a. Translate so preplaced lands at target (x, y)
b. Override (w, h) for fixed blocks
c. Bbox normalize（min 坐标 → 0）
d. Explicit (x, y, w, h) override for preplaced；collect locked_all
e. MIB propagation：用同组中 locked 成员的 (w, h) 作为 canonical 复制给 free
f. De-overlap pass 1：以锁定块为锚，把 free 块沿 ox<oy 的轴推出
   (f1) 同块 only-against-locked 的 greedy（fits 校验，最深 50 轮）
g. De-overlap pass 2：候选点搜索
   (g1) 对仍与 locked 重叠的 i 取 8 个候选位置（j 的四个角）+ (0,0)
   (g2) 选 best (按 y 最小、x 次最小)
h. Bbox re-normalize
i. Re-override preplaced（h 已平移，重新对齐 target 绝对坐标）
j. Soft area fix（free 块 rel_err>0.005 → 按 √area 重缩放）
k. Boundary soft fix（bitmask 1/2/4/8 → 贴 L/R/T/B，且 fits 校验避免引入 overlap）
```

最后再做一次 bbox re-normalize + preplaced re-override 并 return。

### 3.3 Best-of-N 重试

外部 `dit_optimizer_v3.MyOptimizer.solve(...)` 包裹一层 best-of-N：
每跑完一次 `_solve_once()`，检查 `overlap_count`，取当前最佳；若有 0 overlap
立即返回；否则换种子再试，最多重试 6 次。这是处理 SA 偶尔陷入局部最小
（1-2 对顽固 overlap）的兜底机制。

### 3.4 关键工程坑

1. **Z-score 与负坐标**：若 `x_zscore = -3, σ = 33.94, μ = 50.1`，则
   `x_raw = -3 * 33.94 + 50.1 = -51.7`。修复：把 `w_dit / h_dit / x_dit / y_dit`
   整体 `clamp(min=0)`，并且只用 √area 初始化 free 块的边长，避免直接用 DiT 输出做
   边长（负边长在 B*-tree 里直接 NaN）。

2. **preplaced 绝对坐标 vs. free 块归一化坐标**：preplaced 在 4d/4i 两次 override
   都使用 target 绝对坐标，而 bbox_normalize 把 free 块坐标改成相对原点；因此最后
   的 "bbox_normalize + preplaced re-override" 循环以及 best-of-N 重试是必要的。

3. **`scatter_add_` dtype**：联合训练中 Prime 文件是 fp16，collate 必须 cast 到 fp32，
   否则 `aggregate_graph_features` 会因 dtype 不匹配崩溃。

4. **`compute_soft_loss` 中的 `bitwise_and`**：必须 `bc = constraints[:, 4].long()`
   cast 到 int64，否则 fp32 上 `bitwise_and_cuda` 会报 "not implemented for Float"。

---

## 第四部分：核心代码职责对照

| 文件 | 角色 | 关键 API |
|------|------|---------|
| `dit_model_v3.py` | DiT 主干 | `DiffusionTransformer(dim, depth, heads, cond_in, n_steps)` |
| `dit_utils_v3.py` | 调度、采样、损失、特征聚合 | `CosineSchedule`, `q_sample_masked`, `aggregate_graph_features{,_ext}`, `vectorized_diff_loss`, `hard_violation_loss`, `compute_soft_loss`, `compute_norm_stats` |
| `dit_optimizer_v3.py` | 推理 + 后处理 | `class MyOptimizer(FloorplanOptimizer): solve(...)`, `BStarTree`, `_postprocess`, `_deoverlap` |
| `dit_optimizer.py` | 推理 | |

---

## 第五部分：调用与使用说明（Usage）

下面所有命令均假设在 `iccad2026contest/` 目录下运行：

```bash
cd iccad2026contest
```

### 5.1 安装依赖

```bash
pip install -r requirements.txt   # torch >= 2.0, numpy, matplotlib, shapely, tqdm, requests
```

GPU 训练推荐 ≥ 8GB 显存（如 RTX 3060+）。

### 5.2 下载数据集（如果还没有）

Lite 训练集由 `iccad2026_evaluate.get_training_dataloader()` 在首次调用时自动
从 Hugging Face `IntelLabs/FloorSet` 下载到 `/home/xzy/eda/FloorSet/LiteTensorData/`
（约 9.5 GB 展开）。

### 5.3 训练：从零开始

#### 5.3.1 Lite-only 训练

```bash
python train_dit_v3.py
```

输出 checkpoint：
`/home/xzy/eda/model/v3/diffusion_final.pth`

（若想写到竞赛目录下的子路径，可修改训练脚本顶部 `CONFIG["save_dir"]`。）


**可调超参**（在 `CONFIG` 内编辑）：

| 字段 | 默认值 | 含义 |
|------|------|------|
| `epochs` | 10 | 训练轮数 |
| `batch_size` | 64 | DataLoader 批大小 |
| `lr` | 1e-4 | AdamW 学习率 |
| `weight_decay` | 0.01 | AdamW 权重衰减 |
| `grad_clip` | 0.5 | 梯度裁剪上限 |
| `n_steps` | 1000 | DDPM 时间步 |
| `dim` / `depth` / `heads` | 512 / 6 / 8 | Transformer 主干配置 |
| `cond_in` / `extended_features` | 12 / True | 条件通道数与扩展开关 |
| `lambda_noise` | 1.0 | 噪声预测 loss 权重 |
| `lambda_diff` | 1.0 | 评分代理 loss 权重 |
| `lambda_hard` | 0.1 | 硬约束 loss 权重 |
| `lambda_soft` | 0.1 | 软约束代理 loss 权重 |
| `lambda_boundary / lambda_mib / lambda_grouping` | 1.0 / 1.0 / 1.0 | 软约束内部分项权重 |
| `num_train_lite` | 100000 | Lite 每 epoch 采样子集；0 = 全量 |
| `num_train_prime` | 100000 | Prime 每 epoch 采样子集；0 = 全量 |

训练恢复：将 `model_state_dict` / `ema_state_dict` / `optimizer_state_dict` 加载后
从指定 epoch 继续（脚本默认从 0 开始，可在 main loop 处 `load_state_dict`）。

### 5.4 评测

#### 5.4.1 跑完整 100 道验证集

```bash
python iccad2026_evaluate.py --evaluate dit_optimizer.py
python iccad2026_evaluate.py --evaluate dit_optimizer_v3.py
```

输出：

- 控制台：每 case 的 `is_feasible / cost / overlap_v / area_v / dim_v`
- `dit_optimizer_results.json`：完整结果（`total_score`, `summary`, `test_results[]`）

#### 5.4.2 只跑一道（debug 用）

```bash
python iccad2026_evaluate.py --evaluate dit_optimizer.py --test-id 0
```

#### 5.4.3 只校验格式（不跑 SA）

```bash
python iccad2026_evaluate.py --validate dit_optimizer.py
```

#### 5.4.4 把布局存为 JSON（用于离线可视化或重打分）

```bash
python iccad2026_evaluate.py --evaluate dit_optimizer.py --save-solutions
```

输出 `dit_optimizer_solutions.json`，包含每个测试 ID 的
`{test_id, block_count, positions}`。

#### 5.4.5 不重跑 SA，直接重新打分 JSON

```bash
python iccad2026_evaluate.py --score dit_optimizer_solutions.json
```

### 5.5 单元级 / 集成测试

- 输入字段快速检查：

  ```bash
  python check_input.py --mode train --root /home/xzy/eda/FloorSet --worker-idx 3 --layout-idx 2
  python check_input.py --mode test  --root /home/xzy/eda/FloorSet --config-idx 21 --input-idx 1
  ```

- 可微 loss 通路冒烟：

  ```bash
  python training_example.py
  ```

- GPU 上整个训练 / 推理通路冒烟：直接运行 §5.3 中的训练脚本，若无报错即通过。

### 5.6 文件与目录一览

```
iccad2026contest/
├── iccad2026_evaluate.py        # 竞赛主框架（不要修改）
├── optimizer_template.py        # B*-tree SA baseline 参考（不要修改）
├── dit_model_v3.py              # DiT 主干
├── dit_utils_v3.py              # 调度 / 采样 / 损失 / 特征聚合
├── dit_optimizer.py             # 推理
├── dit_optimizer_v3.py          # 推理 + 后处理
├── train_dit_v3.py              # Lite-only 训练
├── visualize_optimizer.py       # 一键评测 + 生成 GT vs solution 对比图
├── DiT_Technical_Report.md      # 本报告
└── requirements.txt

```
