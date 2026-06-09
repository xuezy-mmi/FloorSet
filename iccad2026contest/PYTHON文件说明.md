# DiT 路线 Python 文件说明

> 编写时间：2026-06-09
> 目标：让用户清楚每个 .py 文件是什么、彼此关系，以及如何运行。

---

## 1. 文件总览表

按"用途"分四类。

### 1.1 模型定义（PyTorch nn.Module）

| 文件 | 模型类 | 作用 |
|---|---|---|
| `dit_model.py` | `DiffusionTransformer` | **v1** 模型。自带 `pos_embed`（破坏排列等变性），只用 `log(area)` 作条件，dim=512、depth=8 |
| `dit_model_v3.py` | `DiffusionTransformer` | **v3** 模型。**去掉** `pos_embed`、用 `aggregate_graph_features` 把 b2b / p2b / pin / 约束压成 8 维条件向量，dim=256、depth=6 |

### 1.2 训练脚本（入口：直接 `python xxx.py`）

| 文件 | 训练目标 | 产物路径 | 备注 |
|---|---|---|---|
| `train_dit.py` | v1 DiT，纯噪声 MSE 损失 | `/home/xzy/eda/model/diffusion_final.pth` | 最早版本，已被 v2/v3 替代 |
| `train_dit_v2.py` | v1 架构 DiT，**向量化比赛 cost + 噪声 MSE + 显式面积 loss**，z-score 归一化，x₀ 参数化 | `/home/xzy/eda/model/v2/diffusion_final.pth` | 12 epochs, batch=8, lr=5e-5 |
| `train_dit_v2_regression.py` | **直接回归** Transformer（不跑 diffusion），输出 (x, y, w, h) | `/home/xzy/eda/model/v2_regression/diffusion_final.pth` | 30 epochs, batch=16, lr=1e-4, Cosine LR |
| `train_dit_v3.py` | v3 架构 + v2 训练技巧 | `/home/xzy/eda/model/v3/diffusion_final.pth` | 8 epochs, batch=8, lr=2e-4 |

### 1.3 推理 / 提交用优化器（入口：`python iccad2026_evaluate.py --evaluate <file>`）

| 文件 | 推理方式 | 模型来源 | 后处理 |
|---|---|---|---|
| `dit_optimizer.py` | DDPM 1000 步全量采样（速度很慢），clamp 归一化 | v1 DiT (`/home/xzy/eda/model/diffusion_final.pth`) | 50 次外循环简单推 + 面积微调 |
| `dit_optimizer_v2.py` | **DDIM 100 步**采样，z-score 反归一化，优先 EMA 权重 | v1 DiT (v2 ckpt) | 20 次外循环 deoverlap（hard 块不推） |
| `dit_optimizer_v2_regression.py` | **直接前向**（无 diffusion） | v2_regression ckpt | 20 次外循环 deoverlap |
| `dit_optimizer_v3.py` | **DDIM 50 步**采样，z-score 反归一化 | v3 DiT (v3 ckpt) | 20 次外循环 deoverlap |
| `dit_optimizer_v2_sa.py` | **不用 DiT**：B\*-树 + 模拟退火 + locked block 保护 + 多阶段后处理 | 无（纯经典方法） | 详见 §3 |

### 1.4 工具脚本

| 文件 | 作用 |
|---|---|
| `dit_utils.py` | `DiffusionScheduler`（DDPM β 调度）、`q_sample`（前向加噪） |
| `check_input.py` | 工具脚本，**不参与训练/推理**。用于检查训练/测试样本的原始字段和形状 |

### 1.5 组委会原始文件（**不要修改**）

| 文件 | 作用 |
|---|---|
| `iccad2026_evaluate.py` | 评分、基线、评估器、训练 dataloader |
| `optimizer_template.py` | B\*-树 SA 官方模板（建议作为参考） |
| `training_example.py` | 可微 loss + dataloader 演示 |

### 1.6 备份 / 临时文件

| 文件 | 状态 |
|---|---|
| `dit_optimizer_v2_sa.copy.py` | 旧版备份，内容与 `dit_optimizer_v2_sa.py` 几乎一样，**可以删除** |

---

## 2. 文件之间的依赖关系

```
                                  训练                                  推理 / 评测
                                ──────                                 ───────────

   dit_utils.py ◄────────────┐
   (Scheduler, q_sample)     │
                             │
   dit_model.py ─────────────┤  train_dit.py ─────────────► /home/xzy/eda/model/diffusion_final.pth
                             │  train_dit_v2.py ──────────► /home/xzy/eda/model/v2/diffusion_final.pth
                             │  train_dit_v2_regression.py ► /home/xzy/eda/model/v2_regression/diffusion_final.pth
   dit_model_v3.py ──────────┤  train_dit_v3.py ──────────► /home/xzy/eda/model/v3/diffusion_final.pth
                             │
                             └─ train_dit_v2.py 暴露的公共函数：
                                vectorized_diff_loss, compute_norm_stats
                                会被 v2_regression / v3 训练脚本 import 复用

                                  ┌── dit_optimizer.py          (吃 v1 ckpt)
   训练产物 ─────────────────────┤
                                  ├── dit_optimizer_v2.py       (吃 v2 ckpt)
                                  ├── dit_optimizer_v2_regression.py (吃 v2_regression ckpt)
                                  ├── dit_optimizer_v3.py       (吃 v3 ckpt)
                                  └── dit_optimizer_v2_sa.py    (不需要 ckpt)
```

---

## 3. 各文件详细说明

### 3.1 `dit_model.py`（v1 模型）
- **内容**：最早期 `DiffusionTransformer`。8 层 TransformerEncoder，dim=512，8 头 head。**带可学习 `pos_embed[1, 1000, dim]`**，对 block 顺序敏感。前向用 `x_emb + cond_emb + t_emb`，条件只用 `log(area_target)`。
- **缺点**：
  - `pos_embed` 让模型对 block 顺序有偏好，但推理时 block 顺序由输入决定 → 训练/推理不一致。
  - 条件信息被压成单维 `log(area)`，b2b / p2b / 约束完全没进模型。
- **使用方式**：被 `train_dit.py`、`train_dit_v2.py`、`dit_optimizer.py`、`dit_optimizer_v2.py` 引用。

### 3.2 `dit_model_v3.py`（v3 模型）
- **内容**：重新设计的 `DiffusionTransformer`。dim=256、depth=6，**无 pos_embed**。把 b2b 权重和度、p2b 权重和加权 pin 位置、是否硬约束、boundary code 拼成 8 维条件特征，再投影到 dim。
- **优点**：
  - 排列等变（pos_embed 没了，self-attention 本身是置换等变）。
  - b2b / p2b / pin 位置真的进了模型，HPWL 应当能学得更好。
- **使用方式**：被 `train_dit_v3.py`、`dit_optimizer_v3.py` 引用。**`aggregate_graph_features` 这个函数也被 `dit_optimizer_v2_regression.py` 复用**。

### 3.3 `dit_utils.py`
- **内容**：
  - `DiffusionScheduler(n_steps, beta_start, beta_end)`：维护 `alpha` 和 `alpha_cumprod` 张量
  - `q_sample(x0, t, alpha_cumprod, noise)`：前向加噪 `x_t = sqrt(α̅_t)·x0 + sqrt(1−α̅_t)·ε`
- **使用方式**：被 `train_dit.py`、`train_dit_v2.py`、`train_dit_v3.py`、`dit_optimizer.py` 引用。

### 3.4 `dit_optimizer.py`（v1 推理）
- **类**：`MyOptimizer(FloorplanOptimizer)`
- **核心流程**：
  1. 加载 `/home/xzy/eda/model/diffusion_final.pth`（v1 ckpt）
  2. DDPM 1000 步反向采样
  3. clamp(−1, 1) × 1000 反归一化
  4. **最后用 `target_positions` 覆盖 fixed / preplaced 的 (x, y, w, h)**
  5. 50 次外循环 deoverlap（推右下）
  6. 软块面积微调到 1% 误差内
- **运行**：
  ```bash
  cd iccad2026contest
  python iccad2026_evaluate.py --evaluate dit_optimizer.py
  python iccad2026_evaluate.py --evaluate dit_optimizer.py --test-id 0
  ```
- **已知问题**：1000 步 DDPM 太慢；deoverlap 会把 hard 块也推走（**这是导致 10.0 的原因之一**）。

### 3.5 `dit_optimizer_v2.py`（v2 DiT 推理）
- **类**：`MyOptimizer(FloorplanOptimizer)`
- **核心流程**：
  1. 加载 `/home/xzy/eda/model/v2/diffusion_final.pth`，**优先用 EMA 权重**
  2. 恢复 `norm_stats` (mu, sigma) 和 `n_steps`
  3. DDIM 100 步采样（`ts = linspace(T-1, 0, 101)`）
  4. 反归一化 `x0 = z · sigma + mu`，clamp(0)
  5. hard-constraint 覆盖
  6. 20 次 deoverlap（**hard 块跳过**）
- **运行**：
  ```bash
  python train_dit_v2.py                                # 训练
  python iccad2026_evaluate.py --evaluate dit_optimizer_v2.py
  ```

### 3.6 `dit_optimizer_v2_regression.py`（直接回归推理）
- **类**：`MyOptimizer(FloorplanOptimizer)`，内含 `RegModel` (Transformer 回归器)
- **核心流程**：
  1. 加载 `/home/xzy/eda/model/v2_regression/diffusion_final.pth`
  2. **不采样**，直接前向：`out = model(area, b2b, p2b, pins, constraints)`
  3. 顺序是 `(x, y, w, h)`，**注意与 v2/v3 的 `(w, h, x, y)` 相反**
  4. hard-constraint 覆盖
  5. 20 次 deoverlap
- **运行**：
  ```bash
  python train_dit_v2_regression.py                    # 训练
  python iccad2026_evaluate.py --evaluate dit_optimizer_v2_regression.py
  ```

### 3.7 `dit_optimizer_v3.py`（v3 DiT 推理）
- **类**：`MyOptimizer(FloorplanOptimizer)`
- **核心流程**：与 v2 相同，但加载 `/home/xzy/eda/model/v3/diffusion_final.pth`，DDIM **50 步**（比 v2 的 100 步更快）。
- **运行**：
  ```bash
  python train_dit_v3.py
  python iccad2026_evaluate.py --evaluate dit_optimizer_v3.py
  ```

### 3.8 `dit_optimizer_v2_sa.py`（**当前推荐**，纯经典方法）
- **类**：`MyOptimizer(FloorplanOptimizer)`，内含 `BStarTree`
- **核心流程**：
  1. 把 fixed / preplaced 块加入 `locked` 集合
  2. 以 preplaced 为根建 B\*-树，free block 用 `sqrt(area)` 初始化
  3. **模拟退火**（`initial_temp=60, cooling=0.9, moves_per_temp=8, time_budget=1.5s`），move 包括 `move_rotate` 和 `move_delete_insert`（**locked / root 全部跳过**）
  4. **后处理流水线**：
     - 平移使 preplaced 落到目标 (x, y)
     - 覆盖 fixed 的 (w, h)
     - bbox 归零
     - 再覆盖 preplaced 的 (x, y, w, h)
     - **MIB 后处理**：locked 成员为标准，传播 (w, h)
     - 50 次外循环 deoverlap（**仅推 free 块避开 locked**）
     - 贪心候选搜索兜底（按 (y, x) 取最小合法位置）
     - bbox 再次归零 + 再次覆盖 preplaced
- **运行**（**不需要训练**）：
  ```bash
  python iccad2026_evaluate.py --evaluate dit_optimizer_v2_sa.py
  python iccad2026_evaluate.py --evaluate dit_optimizer_v2_sa.py --test-id 0
  ```
- **当前结果**：总分 ≈ 9.9991（100 case 中 72 可行，28 不可行被 exp(n/12) 加权拉到 cap）。

### 3.9 `train_dit.py`（v1 训练）
- **内容**：
  - 64 batch, 20 epochs, lr=1e-5, 1000 diffusion steps
  - 损失 = 噪声 MSE（在 valid mask 上 sum 然后除以 mask 总和）
  - 训练前**先清空 `/home/xzy/eda/model/`**
- **运行**：
  ```bash
  cd iccad2026contest
  python train_dit.py
  ```
- **重要**：开训前会 `unlink` 整个 `/home/xzy/eda/model/` 下的所有文件，**会清掉其他版本（v2、v3）的 ckpt**。

### 3.10 `train_dit_v2.py`（v2 训练）
- **内容**：
  - 12 epochs, batch=8, lr=5e-5, NUM_SAMPLES=2000
  - 主损失 = `vectorized_diff_loss`（向量化 HPWL + bbox area + overlap + 面积误差，全部按比赛 cost 公式加权）
  - 辅助损失 = `0.05 × noise_mse` + `1.0 × area_loss`（防 w, h 坍缩）
  - z-score 归一化 (mu, sigma)
  - x₀ 参数化（一并反传）
  - EMA (decay=0.999)
  - 训练时把 ground truth 的 fixed/preplaced 硬性写回 `pos_i` 防梯度骗过
- **导出**：`/home/xzy/eda/model/v2/diffusion_final.pth`（含 model / ema / norm_stats / n_steps / model_kwargs）
- **运行**：
  ```bash
  cd iccad2026contest
  python train_dit_v2.py
  ```
- **会被其他脚本 import**：`vectorized_diff_loss` 和 `compute_norm_stats` 被 `train_dit_v2_regression.py`、`train_dit_v3.py` 复用。

### 3.11 `train_dit_v2_regression.py`（回归训练）
- **内容**：
  - 30 epochs, batch=16, lr=1e-4 + Cosine LR
  - 损失 = `1.0 × diff_loss + 0.1 × mse_loss + 0.5 × area_loss`
  - 直接回归 (x, y, w, h) 在 z-score 空间
  - EMA (decay=0.995)
- **导出**：`/home/xzy/eda/model/v2_regression/diffusion_final.pth`
- **运行**：
  ```bash
  cd iccad2026contest
  python train_dit_v2_regression.py
  ```

### 3.12 `train_dit_v3.py`（v3 训练）
- **内容**：与 v2 类似，但用 `dit_model_v3.DiffusionTransformer`（8 维条件、dim=256、depth=6）。
- **导出**：`/home/xzy/eda/model/v3/diffusion_final.pth`
- **运行**：
  ```bash
  cd iccad2026contest
  python train_dit_v3.py
  ```

### 3.13 `check_input.py`（工具）
- 检查单个训练 / 测试样本的字段和形状。**与训练流程无关**，可单独运行：
  ```bash
  python check_input.py --mode train --root /home/xzy/eda/FloorSet --worker-idx 3 --layout-idx 2
  python check_input.py --mode test  --root /home/xzy/eda/FloorSet --config-idx 21 --input-idx 1
  ```

### 3.14 `dit_optimizer_v2_sa.copy.py`（备份）
- 与 `dit_optimizer_v2_sa.py` 内容几乎相同的旧版本，**不再使用**。可以删除。

---

## 4. 如何完整跑一遍（推荐顺序）

### 路径 A：纯 DiT 路线（验证 4 个 DiT 优化器各自的分数）

```bash
cd iccad2026contest

# 1) 训练 v1 DiT（**警告**：会清空 /home/xzy/eda/model/）
python train_dit.py
python iccad2026_evaluate.py --evaluate dit_optimizer.py

# 2) 训练 v2 DiT（向量化 cost 损失，产物存到 v2/ 子目录）
python train_dit_v2.py
python iccad2026_evaluate.py --evaluate dit_optimizer_v2.py

# 3) 训练 v2 回归（直接回归，产物存到 v2_regression/ 子目录）
python train_dit_v2_regression.py
python iccad2026_evaluate.py --evaluate dit_optimizer_v2_regression.py

# 4) 训练 v3 DiT（用 v3 模型 + v2 损失，产物存到 v3/ 子目录）
python train_dit_v3.py
python iccad2026_evaluate.py --evaluate dit_optimizer_v3.py
```

### 路径 B：纯经典路线（不需要训练，速度最快，当前推荐）

```bash
cd iccad2026contest
python iccad2026_evaluate.py --evaluate dit_optimizer_v2_sa.py
```

### 路径 C：调试单个 case

```bash
cd iccad2026contest
python iccad2026_evaluate.py --evaluate dit_optimizer_v2_sa.py --test-id 0
```

---

## 5. 几个易踩的坑

1. **`train_dit.py` 会清空 `/home/xzy/eda/model/`**。如果之前已经训了 v2/v3，**先备份**或者用 `train_dit_v2.py` / `train_dit_v3.py` 这种带子目录的版本。
2. **`dit_model.py` 的 `pos_embed` 在推理时 block 顺序改变会失效**——所以 v1 DiT 训练/推理都依赖"训练时是按 dataset 索引顺序"这个隐式假设。v3 修了这个问题。
3. **推理端的输出顺序差异**：
   - `dit_optimizer.py`：模型输出 `(w, h, x, y)`
   - `dit_optimizer_v2.py` / `dit_optimizer_v3.py`：也是 `(w, h, x, y)`
   - `dit_optimizer_v2_regression.py`：是 `(x, y, w, h)`（顺序反了）
4. **DDIM 步数**：v2=100 步，v3=50 步，v1 是 1000 步 DDPM。
5. **EMA 权重**：`dit_optimizer_v2.py` 和 `dit_optimizer_v3.py` 在加载 ckpt 时优先用 `ema_state_dict`（训练脚本里也有），推理时用 EMA 效果更稳。
6. **后处理"hard 块不推"** 这一点：只有 `dit_optimizer_v2.py` / `dit_optimizer_v2_regression.py` / `dit_optimizer_v3.py` / `dit_optimizer_v2_sa.py` 这几个新版做到了；老的 `dit_optimizer.py` 会把 hard 块也推，导致 infeasible。
