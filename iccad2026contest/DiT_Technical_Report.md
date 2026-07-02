# DiT 模型技术路线报告

> 基于 ICCAD 2026 FloorSet 竞赛的 DiT（Diffusion Transformer）模型设计

---

## 第一部分：DiT 模型架构

### 1.1 问题背景与建模

芯片版图规划（Floorplanning）的目标是为数十到上百个矩形模块在二维平面上分配合适的位置和尺寸，使得：

1. **线长（Wirelength）最小化**：块间连接和引脚引线总长度最短
2. **包围盒面积（Bounding Box Area）最小化**：所有模块整体占据面积最小
3. **硬约束满足**：
   - **固定块（Fixed-shape）**：尺寸 (w, h) 必须与目标完全一致
   - **预置块（Preplaced）**：位置 (x, y) 和尺寸 (w, h) 必须与目标完全一致
   - **软块（Soft blocks）**：面积 |w×h - target| / target ≤ 1%
4. **软约束尽量满足**（影响评分指数因子，不满足不导致 infeasible）：
   - MIB 同组块形状一致
   - Cluster 簇内块聚合
   - Boundary 边界贴边

我们将每个模块的布局建模为 4 维向量 **[w, h, x, y]**，然后将其表示为 DDPM 扩散过程中的"干净图像"，通过去噪网络学习从噪声中恢复合法布局。

### 1.2 扩散模型基础

#### 1.2.1 DDPM（去噪扩散概率模型）

DDPM 通过两步过程工作：

**前向过程（Forward Process）**：向真实布局 `x₀` 中逐步添加高斯噪声，经过 T 步得到 `x_T`：
```
x_t = √(ᾱ_t) · x₀ + √(1 - ᾱ_t) · ε,   ε ~ N(0, I)
```

**反向过程（Reverse Process）**：训练一个神经网络 `ε_θ(x_t, t, c)` 预测噪声，从而恢复干净布局：
```
x₀_hat = (x_t - √(1 - ᾱ_t) · ε_θ) / √ᾱ_t
```

#### 1.2.2 DDIM（确定性采样加速）

完整 DDPM 需要 1000 步采样，竞赛实时评测无法接受。DDIM 将采样步数压缩到 50 步，利用：
```
x_{t-1} = √α_{t-1} · x₀_hat + √(1 - α_{t-1}) · ε_θ(x_t, t, c)
```
在保持生成质量的同时将推理速度提升 20 倍。

#### 1.2.3 Cosine Schedule（余弦噪声调度）

采用 Nichol & Dhariwal (2021) 提出的余弦调度，相比线性调度在细粒度结构（模块位置/尺寸）上提供更平滑的噪声衰减：
```
β_t = 1 - ᾱ_t / ᾱ_{t-1},   ᾱ_t = cos²((t + s)/(1 + s) · π/2)
```
` s = 0.008` 为平滑超参数。

### 1.3 模型架构详解

#### 1.3.1 整体结构

```
输入: x_t [B, N, 4] (带噪布局, z-score 标准化) + 条件信息
  ↓
aggregate_graph_features: 将所有条件聚合成 [B, N, 8] 每块特征
  ↓
输入投影: Linear(4+8 → dim) + LayerNorm → tokens [B, N, dim]
  ↓
条件投影: Linear(8 → dim) + SiLU + Linear(dim → dim) → cond_emb
  ↓
tokens = tokens + cond_emb + time_emb
  ↓
Pre-LN Transformer × 6层 (每层含 edge-bias 注意力 + FFN)
  ↓
输出投影: Linear(dim → dim) + GELU + LayerNorm + Linear(dim → 4)
  ↓
输出: pred_noise [B, N, 4] (z-score 空间)
```

#### 1.3.2 8 通道图特征（aggregate_graph_features）

每块提取 8 维条件特征，比原始 DiT 的单通道 log(area) 丰富得多：

| 通道 | 含义 | 作用 |
|------|------|------|
| 0 | log(area) | 面积大小，基础尺度信息 |
| 1 | b2b_w | 块间连线权重之和（度中心性） |
| 2 | b2b_d | 块间连线数量（度数） |
| 3 | p2b_w | 引脚连线权重之和 |
| 4 | p2b_px | 引脚 x 坐标加权平均 |
| 5 | p2b_py | 引脚 y 坐标加权平均 |
| 6 | is_hard | 是否硬约束（fixed/preplaced） |
| 7 | boundary | 边界约束位掩码（归一化到 [0,1]） |

#### 1.3.3 Edge-Bias 自注意力

**创新点**：将块间连线权重（b2b_conn）引入注意力机制。

实现方式：
1. 从 b2b_conn [B, E, 3] 构建 [B, N, N] 边权重矩阵 `edge_w`
2. 通过 `Linear(1, 1)` 投影为单标量偏置
3. 将偏置加到注意力 logits 上：`attn_logits = QK^T / √d + edge_bias`
4. 偏置在所有注意力头之间广播

这样，**连线紧密的块在注意力中有更高的相互响应**，让模型学习布局中的拓扑结构。

#### 1.3.4 Pre-LN Transformer

标准 Transformer 的结构是 `LayerNorm → Attention → Residual`，而 Pre-LN（Pre-LayerNorm）为：
```
src = LayerNorm(x)
attn_out = Attention(src, src, src)
x = x + attn_out
src2 = LayerNorm(x)
ff_out = FFN(src2)
x = x + ff_out
```

优势：
- 训练更稳定，适合 6 层以上的深度网络
- 梯度在各层间更均匀地流动

### 1.4 针对竞赛的技术支持

#### 1.4.1 Z-Score 标准化

训练时对 ground truth 布局做通道级 Z-Score 标准化：
```
x₀_normalized = (fp_sol - μ) / σ
```
每个通道（w, h, x, y）独立统计均值和标准差，确保所有维度在相同数值范围内训练。

#### 1.4.2 有效块掩码（Valid Block Mask）

数据集使用 -1 填充变长序列，模型通过 `mask = (area_target != -1)` 区分有效块和填充块，在所有计算中自动排除填充位置的影响。

#### 1.4.3 硬约束后处理（零后处理方案）

当前 `dit_optimizer.py` **完全不做后处理**（无 SA、无去重叠、无面积修正），仅在推理后对 fixed/preplaced 块做硬覆盖：
- Preplaced 块：强制替换为 `(x, y, w, h) = target`
- Fixed 块：强制替换为 `(w, h) = target`

这意味着模型**必须自己学会生成合法布局**，后处理仅用于纠正极少数硬约束违规。

### 1.5 模型创新总结

| 创新点 | 原始 DiT | 本 DiT |
|--------|----------|--------|
| 注意力机制 | 标准自注意力 | Edge-bias 自注意力（融入 b2b 连线权重） |
| 条件编码 | 单通道 log(area) | 8 通道图特征（含硬约束、边界、拓扑信息） |
| Transformer 归一化 | Post-LN | Pre-LN（更稳定） |
| 输入/输出投影 | 单一 Linear | Linear + LayerNorm（更稳定） |
| 噪声调度 | 线性 β | Cosine Schedule（更平滑） |
| 推理加速 | DDPM 1000步 | DDIM 50 步（20x 加速） |
| 竞赛感知 | 无 | 硬约束覆盖 + Z-score + 有效掩码 |

---

## 第二部分：训练技术路线

### 2.1 数据集支持

#### 2.1.1 Lite 数据集（FloorSet-Lite）

- **来源**：`/home/xzy/eda/floorset_lite/`
- **规模**：100 万样本，100 个 worker 分片
- **格式**：`fp_sol [w, h, x, y]` 矩形布局（直接可用）
- **加载**：通过 `iccad2026_evaluate.get_training_dataloader()` 自动下载和管理

#### 2.1.2 Prime 数据集（FloorSet-Prime）

- **来源**：`/home/xzy/eda/PrimeTensorData/config_*/primedata_*.pth`
- **规模**：10 万样本（100 configs × 1000 samples/config）
- **格式**：`fp_sol` 为多边形顶点列表（每块 [V_i, 2]），需转换为包围盒 `[x_min, y_min, x_max, y_max]`
- **加载**：通过 `prime_dataset.FloorplanDatasetPrime` + 自定义 `prime_collate_fn`

#### 2.1.3 联合训练模式

`--dataset combined` 时交替加载 lite 和 prime 批次，让模型同时学习两种数据分布。

### 2.2 训练稳定性技术

#### 2.2.1 EMA（指数移动平均）

```python
with torch.no_grad():
    for p_em, p_m in zip(ema.parameters(), model.parameters()):
        p_em.mul_(0.999).add_(p_m, alpha=0.001)
```
- 衰减率 0.999，推理使用 EMA 权重
- 平滑训练噪声，提升推理稳定性

#### 2.2.2 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP=1.0)
```
防止梯度爆炸，确保训练稳定。

#### 2.2.3 NaN/Inf 检测

每个 batch 检测 loss 是否为 NaN/Inf，跳过该步更新避免梯度污染。

### 2.3 Loss 定义详解（四层结构）

#### 2.3.1 优先级 1：`λ_hard × hard_violation_loss` (λ=10)

**核心**：保证**可行性**——使模型不产生 infeasible 解。

```python
hard_violation_loss = SSE(fixed块偏离目标w,h)
                   + SSE(preplaced块偏离目标x,y,w,h)
                   + Σ relu(|w×h - a|/a - 0.01) / n_free_blocks
```

**设计哲学**：
- **= 0** 当且仅当所有硬约束满足（fixed/preplaced 精确匹配，软块面积在 1% 内）
- **> 0** 当硬约束违反，梯度强制拉回
- λ=10 是最高权重，确保模型首先学会"不犯错"

**与竞赛评测的关系**：竞赛中 infeasible 解直接得 **cost = 10.0**，而此 loss 的存在使模型在训练阶段就学会避免这种情况。

#### 2.3.2 优先级 2：`λ_mse × loss_mse_x0` (λ=0.5)

**核心**：直接监督 ground truth 布局位置。

```python
loss_mse_x0 = MSE(x̂₀, x₀)   # 在 z-score 标准化空间计算
```

**设计哲学**：
- 之前版本的缺失项——只通过扩散动态隐式学习 GT
- 新增后直接告诉模型"正确布局应该在哪里"
- 在 z-score 空间计算保证各通道同等权重
- 与 hard_violation_loss 互补（后者惩罚违反，前者奖励正确）

#### 2.3.3 优先级 3：`λ_diff × vectorized_diff_loss` (λ=1.0)

**核心**：比赛评分公式的可微分代理。

```python
vectorized_diff_loss = (1 + 0.5 × (hpwl_gap + area_gap)) × exp(2 × V_soft)

其中：
  hpwl_gap = relu((hpwl - hpwl_baseline) / hpwl_baseline)  ≥ 0
  area_gap = relu((bbox_area - area_baseline) / area_baseline) ≥ 0
  V_soft = overlap_ratio + area_tolerance_excess
         = (Σ overlap_area / Σ block_area) + Σ relu(|w×h-a|/a - 0.01) / n
```

**设计哲学**：
- 与竞赛真实评分公式完全一致的操作可微版本
- 所有不可微操作（min, if, count）均用 relu 等可微近似替代
- 驱动模型降低线长、缩小面积、减少重叠
- 注意：这里用的是 baseline（数据集中的 GT 指标）而非绝对值

#### 2.3.4 优先级 4：`λ_noise × loss_noise` (λ=0.05)

**核心**：标准 DDPM 噪声预测损失。

```python
loss_noise = MSE(pred_noise, true_noise)
```

**设计哲学**：
- 维持扩散动态正确性的基础信号
- 权重最低但不可或缺
- 独立于具体问题，为去噪提供通用能力

### 2.4 完整 Loss 公式

```python
loss = 10.0  × hard_violation_loss   # 可行性
     + 0.5   × MSE(x̂₀, x₀)          # GT 直接监督
     + 1.0   × vectorized_diff_loss   # 竞赛评分代理
     + 0.05  × MSE(pred_noise, ε)    # 扩散动态
```

### 2.5 模型收敛方向

#### 阶段 1：可行性优先（训练早期）

`hard_violation_loss` 梯度最强，模型首先学会：
- Fixed 块的 (w, h) 尺寸匹配目标
- Preplaced 块的 (x, y, w, h) 完全匹配目标
- Soft 块的面积在目标 1% 容限内

此阶段模型可能布局质量很差（线长、面积大），但**必须保证不产生 infeasible 解**。

#### 阶段 2：布局质量提升（训练中期）

`loss_mse_x0` 和 `vectorized_diff_loss` 主导，模型学习：
- 块位置靠近 GT 位置
- 线长（HPWL）降低
- 包围盒面积缩小
- 重叠减少

`hard_violation_loss` 保持低值但可能有小幅波动。

#### 阶段 3：扩散动态精细化（训练后期）

`loss_noise` 在低噪声步（t → 0）时信号更清晰，帮助模型精细调整块的位置和尺寸精度。

#### 最终收敛状态

```
hard_violation_loss ≈ 0     ← 所有硬约束满足
loss_mse_x0 → 最小化        ← 布局趋近 GT
vectorized_diff_loss → 趋近 baseline 或更优  ← 质量指标良好
loss_noise → 趋近 0          ← 扩散动态正确
```

### 2.6 训练流程总览

```
数据集加载 (lite / prime / combined)
  ↓
Z-Score 标准化 (μ, σ from 64 batches)
  ↓
前向扩散: x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
  ↓
模型预测: ε_θ(x_t, t, c) → pred_noise
  ↓
四分量 Loss 计算
  ↓
反向传播 + 梯度裁剪
  ↓
参数更新 + EMA 更新
  ↓
每 25 batch 打印: loss, diff, mse, noise, hard
  ↓
每 epoch 保存: model_state_dict, ema_state_dict, norm_stats
```

### 2.7 关键技术特点总结

| 技术 | 作用 | 实现 |
|------|------|------|
| **Cosine Schedule** | 更平滑的噪声衰减 | `dit_utils.CosineSchedule` |
| **EMA (0.999)** | 训练稳定性 + 推理质量 | 每步 `p_em = 0.999·p_em + 0.001·p_m` |
| **Z-Score 标准化** | 数值稳定性 | `x_norm = (x - μ) / σ` |
| **有效掩码** | 处理变长序列 | `mask = (area != -1)` |
| **4 层 Loss 结构** | 可行性 + 准确性 + 评分 + 动态 | λ=(10, 0.5, 1.0, 0.05) |
| **DDIM 50 步推理** | 实时性（vs 1000 步 DDPM） | `dit_optimizer._ddim_sample` |
| **Hard Override** | 保证硬约束 | 仅在 fixed/preplaced 上做后处理 |

### 2.8 与原始 `compute_training_loss_differentiable` 的区别

| 维度 | ICCAD 官方函数 | 本训练方案 |
|------|---------------|-----------|
| 硬约束惩罚 | ❌ 无 | ✅ `hard_violation_loss` λ=10 |
| GT 直接监督 | ❌ 无 | ✅ `loss_mse_x0` λ=0.5 |
| 竞赛评分公式 | ✅ | ✅ `vectorized_diff_loss` |
| DDPM 动态 | ❌ 无 | ✅ `loss_noise` λ=0.05 |
| EMA | ❌ 无 | ✅ 0.999 衰减 |
| Cosine Schedule | ❌ 线性 | ✅ 余弦调度 |
| 多数据集 | ❌ 仅 lite | ✅ lite + prime + combined |

**核心区别**：ICCAC 官方函数只提供了竞赛评分公式的可微版本，缺乏硬约束保证和直接监督。本方案在此基础上构建了完整的四层监督体系，确保模型在训练过程中首先学会"不犯法"（可行性），其次学会"做好"（质量），最终学会"做对"（逼近 GT）。
