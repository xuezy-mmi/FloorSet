"""
dit_optimizer_v3.py - Hybrid DiT + B*-tree SA optimizer.

Pipeline:
  1. Load DiT checkpoint (v3 EMA weights).
  2. Run DDIM sampling to get initial (w, h, x, y) per block.
  3. Build initial widths/heights:
       - fixed blocks: target (w, h)
       - free blocks: sqrt(area)  (DiT predictions NOT used in 档 1)
  4. Build B*-tree with locked (fixed + preplaced) blocks, run SA to place
     free blocks (small budget: 1.5s).
  5. Post-processing (in this order):
       a. Translate so preplaced lands at target (x, y).
       b. Override (w, h) for fixed blocks.
       c. Bbox normalize.
       d. Explicit (x, y, w, h) override for preplaced.
       e. MIB propagation (locked member as canonical).
       f. De-overlap pass 1: shove free out of locked.
       g. De-overlap pass 2: greedy candidate search.
       h. Re-normalize bbox to origin.
       i. Re-override preplaced.
       j. Soft-constraint area fix (free blocks: scale w*h to area_target).
       k. Boundary soft fix: nudge blocks with boundary bit to the matching edge.
"""
import math
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))
from iccad2026_evaluate import (
    FloorplanOptimizer, calculate_hpwl_b2b, calculate_hpwl_p2b, calculate_bbox_area,
)
from dit_model_v3 import DiffusionTransformer
from dit_utils_v3 import CosineSchedule


CKPT_PATH = Path("/home/xzy/eda/model/v3/diffusion_final.pth")
N_DDIM_STEPS = 50
SA_TIME_BUDGET_S = 1.5
SA_INITIAL_TEMP = 60.0
SA_COOLING_RATE = 0.9
SA_MOVES_PER_TEMP = 8


# ---------------------------------------------------------------------------
# B*-tree (replicated from dit_optimizer_v2_sa.py; locked-block aware)
# ---------------------------------------------------------------------------
class BStarTree:
    def __init__(self, n_blocks, widths, heights, root_index=0, locked=None):
        self.n = n_blocks
        self.widths = list(widths)
        self.heights = list(heights)
        self.parent = [-1] * n_blocks
        self.left = [-1] * n_blocks
        self.right = [-1] * n_blocks
        self.root = root_index
        self.locked = set(locked) if locked else set()
        self._build_random_tree()

    def _build_random_tree(self):
        if self.n == 0:
            return
        self.parent = [-1] * self.n
        self.left = [-1] * self.n
        self.right = [-1] * self.n
        order = list(range(self.n))
        random.shuffle(order)
        if self.root in order:
            order.remove(self.root)
            order = [self.root] + order
        else:
            self.root = order[0]
        for i in range(1, self.n):
            block = order[i]
            existing = order[random.randint(0, i - 1)]
            if random.random() < 0.5:
                if self.left[existing] == -1:
                    self.left[existing] = block; self.parent[block] = existing
                elif self.right[existing] == -1:
                    self.right[existing] = block; self.parent[block] = existing
                else:
                    self._insert_at_leaf(block, existing)
            else:
                if self.right[existing] == -1:
                    self.right[existing] = block; self.parent[block] = existing
                elif self.left[existing] == -1:
                    self.left[existing] = block; self.parent[block] = existing
                else:
                    self._insert_at_leaf(block, existing)

    def _insert_at_leaf(self, block, start):
        current = start
        while True:
            if random.random() < 0.5:
                if self.left[current] == -1:
                    self.left[current] = block; self.parent[block] = current
                    return
                current = self.left[current]
            else:
                if self.right[current] == -1:
                    self.right[current] = block; self.parent[block] = current
                    return
                current = self.right[current]

    def pack(self):
        positions = [(0.0, 0.0, self.widths[i], self.heights[i]) for i in range(self.n)]
        if self.n == 0:
            return positions
        contour = [(0.0, 0.0)]

        def get_contour_y(x_start, x_end):
            max_y = 0.0
            for i, (cx_end, cy_top) in enumerate(contour):
                cx_start = contour[i-1][0] if i > 0 else 0.0
                if x_start < cx_end and x_end > cx_start:
                    max_y = max(max_y, cy_top)
            return max_y

        def update_contour(x_start, x_end, y_top):
            nonlocal contour
            new_contour = []
            for i, (cx_end, cy_top) in enumerate(contour):
                cx_start = contour[i-1][0] if i > 0 else 0.0
                if cx_end <= x_start:
                    new_contour.append((cx_end, cy_top))
                elif cx_start >= x_end:
                    new_contour.append((cx_end, cy_top))
                else:
                    if cx_start < x_start:
                        new_contour.append((x_start, cy_top))
                    if cx_end > x_end:
                        new_contour.append((cx_end, cy_top))
            insert_pos = 0
            for i, (cx_end, _) in enumerate(new_contour):
                if cx_end <= x_start:
                    insert_pos = i + 1
            new_contour.insert(insert_pos, (x_end, y_top))
            new_contour.sort(key=lambda x: x[0])
            merged = []
            for x_end, y_top in new_contour:
                if merged and merged[-1][1] == y_top:
                    merged[-1] = (x_end, y_top)
                else:
                    merged.append((x_end, y_top))
            contour = merged if merged else [(x_end, 0.0)]

        def dfs(node, parent_right_edge):
            if node == -1: return
            w, h = self.widths[node], self.heights[node]
            if node == self.root:
                x, y = 0.0, 0.0
            else:
                x = parent_right_edge
                y = get_contour_y(x, x + w)
            positions[node] = (x, y, w, h)
            update_contour(x, x + w, y + h)
            dfs(self.left[node], x + w)
            dfs(self.right[node], x)

        dfs(self.root, 0.0)
        return positions

    def copy(self):
        new = BStarTree.__new__(BStarTree)
        new.n = self.n
        new.widths = self.widths.copy()
        new.heights = self.heights.copy()
        new.parent = self.parent.copy()
        new.left = self.left.copy()
        new.right = self.right.copy()
        new.root = self.root
        new.locked = set(self.locked)
        return new

    def move_rotate(self, block):
        if block == self.root or block in self.locked:
            return
        self.widths[block], self.heights[block] = self.heights[block], self.widths[block]

    def move_delete_insert(self, block):
        if self.n <= 1 or block == self.root or block in self.locked:
            return
        w, h = self.widths[block], self.heights[block]
        self._delete_node(block)
        target = random.randint(0, self.n - 1)
        while target == block or target == self.root or target in self.locked:
            target = random.randint(0, self.n - 1)
        self._insert_node(block, target, random.choice([True, False]))
        self.widths[block], self.heights[block] = w, h

    def _delete_node(self, node):
        parent = self.parent[node]
        left_child = self.left[node]
        right_child = self.right[node]
        if left_child == -1 and right_child == -1:
            replacement = -1
        elif left_child == -1:
            replacement = right_child
        elif right_child == -1:
            replacement = left_child
        else:
            replacement = left_child
            rightmost = left_child
            while self.right[rightmost] != -1:
                rightmost = self.right[rightmost]
            self.right[rightmost] = right_child
            self.parent[right_child] = rightmost
        if parent == -1:
            self.root = replacement
        elif self.left[parent] == node:
            self.left[parent] = replacement
        else:
            self.right[parent] = replacement
        if replacement != -1:
            self.parent[replacement] = parent
        self.parent[node] = -1
        self.left[node] = -1
        self.right[node] = -1

    def _insert_node(self, node, target, as_left):
        if as_left:
            old_child = self.left[target]
            self.left[target] = node
        else:
            old_child = self.right[target]
            self.right[target] = node
        self.parent[node] = target
        if old_child != -1:
            self.left[node] = old_child
            self.parent[old_child] = node


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.mu = None
        self.sigma = None
        self.alpha_cumprod = None
        self.n_train_steps = 1000
        self._load()

    def _load(self):
        if not CKPT_PATH.exists():
            if self.verbose:
                print(f"[v3] No checkpoint at {CKPT_PATH}, will use SA fallback")
            return
        ckpt = torch.load(CKPT_PATH, map_location=self.device, weights_only=False)
        kw = ckpt.get('model_kwargs', {'dim': 256, 'depth': 6, 'heads': 8, 'cond_in': 8, 'n_steps': 1000})
        self.model = DiffusionTransformer(**kw).to(self.device)
        state = ckpt.get('ema_state_dict') or ckpt.get('model_state_dict')
        self.model.load_state_dict(state)
        self.model.eval()

        ns = ckpt['norm_stats']
        self.mu = ns['mu'].to(self.device)
        self.sigma = ns['sigma'].to(self.device)
        self.n_train_steps = ckpt.get('n_steps', 1000)
        sched = CosineSchedule(self.n_train_steps)
        self.alpha_cumprod = sched.alpha_cumprod.to(self.device)
        if self.verbose:
            print(f"[v3] Loaded ckpt from {CKPT_PATH}")

    @torch.no_grad()
    def _ddim_sample(self, area, b2b, p2b, pins, constr, n_steps: int) -> torch.Tensor:
        N = area.shape[1]
        x = torch.randn(1, N, 4, device=self.device)
        T = self.n_train_steps
        ts = torch.linspace(T - 1, 0, n_steps + 1).long()
        for i in range(n_steps):
            t_cur = int(ts[i].item())
            t_prev = int(ts[i + 1].item()) if i + 1 < n_steps else -1
            t_tensor = torch.full((1,), t_cur, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t_tensor, area, b2b, p2b, pins, constr)
            a_cur = self.alpha_cumprod[t_cur].clamp(min=1e-6)
            a_prev = self.alpha_cumprod[t_prev].clamp(min=0.0) if t_prev >= 0 else torch.tensor(1.0, device=self.device)
            x0_hat = (x - torch.sqrt(1.0 - a_cur) * pred_noise) / torch.sqrt(a_cur)
            x0_hat = x0_hat.clamp(-5.0, 5.0)
            dir_xt = torch.sqrt((1.0 - a_prev).clamp(min=0.0)) * pred_noise
            x = torch.sqrt(a_prev) * x0_hat + dir_xt
            valid = (area != -1)
            x = x * valid.unsqueeze(-1).float()
        x0_real = x * self.sigma + self.mu
        return x0_real.clamp(min=0.0)

    def _sa_cost(self, positions, b2b, p2b, pins):
        return (calculate_hpwl_b2b(positions, b2b)
                + calculate_hpwl_p2b(positions, p2b, pins)
                + calculate_bbox_area(positions) * 0.01)

    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity,
              pins_pos, constraints, target_positions=None):
        if self.model is None:
            return self._sa_only_solve(block_count, area_targets, b2b_connectivity,
                                       p2b_connectivity, pins_pos, constraints,
                                       target_positions)

        # ----- 1. DiT inference -----
        area = area_targets.unsqueeze(0).to(self.device)
        b2b = (b2b_connectivity.unsqueeze(0).to(self.device)
               if b2b_connectivity is not None and b2b_connectivity.numel()
               else torch.zeros(1, 0, 3, device=self.device))
        p2b = (p2b_connectivity.unsqueeze(0).to(self.device)
               if p2b_connectivity is not None and p2b_connectivity.numel()
               else torch.zeros(1, 0, 3, device=self.device))
        pins = (pins_pos.unsqueeze(0).to(self.device)
                if pins_pos is not None and pins_pos.numel()
                else torch.zeros(1, 0, 2, device=self.device))
        constr = constraints.unsqueeze(0).to(self.device)

        x0_real = self._ddim_sample(area, b2b, p2b, pins, constr, N_DDIM_STEPS)
        x0_real = x0_real[0, :block_count]                # [N, 4] = (w, h, x, y)
        w_dit = x0_real[:, 0].cpu().numpy()
        h_dit = x0_real[:, 1].cpu().numpy()

        # ----- 2. Identify locked blocks -----
        locked = set()
        fixed_w, fixed_h = {}, {}
        preplaced_root = 0
        if target_positions is not None and constraints is not None:
            nc = constraints.shape[1] if constraints.dim() > 1 else 0
            for i in range(block_count):
                is_fixed = nc > 0 and float(constraints[i, 0]) != 0
                is_preplaced = nc > 1 and float(constraints[i, 1]) != 0
                if is_preplaced:
                    locked.add(i)
                    preplaced_root = i
                elif is_fixed:
                    locked.add(i)
                    fixed_w[i] = float(target_positions[i, 2])
                    fixed_h[i] = float(target_positions[i, 3])

        # ----- 3. Build initial widths/heights (sqrt(area) only) -----
        # 档 1: ignore DiT's (w, h) output. Use sqrt(area) for all free blocks.
        # This makes the optimizer equivalent to v2_sa.py.
        widths, heights = [], []
        for i in range(block_count):
            if i in fixed_w:
                widths.append(fixed_w[i])
                heights.append(fixed_h[i])
            else:
                area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
                s = math.sqrt(area)
                widths.append(s)
                heights.append(s)

        # ----- 4. B*-tree SA (with locked blocks) -----
        tree = BStarTree(block_count, widths, heights, root_index=preplaced_root, locked=locked)
        current_positions = tree.pack()
        current_cost = self._sa_cost(current_positions, b2b_connectivity,
                                      p2b_connectivity, pins_pos)
        best_positions, best_cost = current_positions, current_cost

        t0 = time.time()
        temp = SA_INITIAL_TEMP
        while temp > 1.0:
            if time.time() - t0 > SA_TIME_BUDGET_S:
                break
            for _ in range(SA_MOVES_PER_TEMP):
                if time.time() - t0 > SA_TIME_BUDGET_S:
                    break
                old_tree = tree.copy()
                free = [i for i in range(block_count) if i not in locked]
                if not free:
                    break
                b = random.choice(free)
                if random.random() < 0.5:
                    tree.move_rotate(b)
                else:
                    tree.move_delete_insert(b)
                new_positions = tree.pack()
                new_cost = self._sa_cost(new_positions, b2b_connectivity,
                                          p2b_connectivity, pins_pos)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-3)):
                    current_positions = new_positions
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_positions = new_positions
                else:
                    tree = old_tree
            temp *= SA_COOLING_RATE

        return self._postprocess(best_positions, block_count, preplaced_root,
                                  fixed_w, fixed_h, target_positions, constraints,
                                  area_targets)

    # -----------------------------------------------------------------
    def _postprocess(self, positions, block_count, preplaced_root, fixed_w, fixed_h,
                     target_positions, constraints, area_targets):
        positions = [list(p) for p in positions]

        # 4a. Translate so preplaced lands at target
        if preplaced_root in {*fixed_w} or (target_positions is not None and constraints is not None and
                                              float(constraints[preplaced_root, 1]) != 0):
            tp = target_positions[preplaced_root]
            dx = float(tp[0]) - positions[preplaced_root][0]
            dy = float(tp[1]) - positions[preplaced_root][1]
            for k in range(block_count):
                positions[k][0] += dx
                positions[k][1] += dy

        # 4b. Override (w, h) for fixed blocks
        for i in fixed_w:
            positions[i][2] = fixed_w[i]
            positions[i][3] = fixed_h[i]

        # 4c. Bbox normalize
        x_min = min(p[0] for p in positions)
        y_min = min(p[1] for p in positions)
        for k in range(block_count):
            positions[k][0] -= x_min
            positions[k][1] -= y_min

        # 4d. Explicit (x, y, w, h) override for preplaced; collect locked_all
        locked_all = set()
        if target_positions is not None and constraints is not None:
            nc = constraints.shape[1] if constraints.dim() > 1 else 0
            for i in range(block_count):
                is_fixed = nc > 0 and float(constraints[i, 0]) != 0
                is_preplaced = nc > 1 and float(constraints[i, 1]) != 0
                if is_preplaced:
                    tp = target_positions[i]
                    positions[i][0] = float(tp[0])
                    positions[i][1] = float(tp[1])
                    positions[i][2] = float(tp[2])
                    positions[i][3] = float(tp[3])
                    locked_all.add(i)
                elif is_fixed:
                    locked_all.add(i)

        # 4e. MIB propagation (use locked member's (w, h) as canonical)
        if constraints is not None:
            nc = constraints.shape[1] if constraints.dim() > 1 else 0
            if nc > 2:
                mib_groups = {}
                for i in range(block_count):
                    mid = int(constraints[i, 2])
                    if mid > 0:
                        mib_groups.setdefault(mid, []).append(i)
                for mid, members in mib_groups.items():
                    locked_members = [j for j in members if j in locked_all]
                    if locked_members:
                        ref_i = locked_members[0]
                    else:
                        ref_i = max(members, key=lambda j: positions[j][2] * positions[j][3])
                    w = positions[ref_i][2]
                    h = positions[ref_i][3]
                    for j in members:
                        if j in locked_all:
                            continue
                        positions[j][2] = w
                        positions[j][3] = h

        # 4f. De-overlap pass 1: shove out of locked
        def fits(x, y, w, h, ignore):
            for j in range(block_count):
                if j == ignore:
                    continue
                xj, yj, wj, hj = positions[j]
                ox = max(0.0, min(x + w, xj + wj) - max(x, xj))
                oy = max(0.0, min(y + h, yj + hj) - max(y, yj))
                if ox > 1e-6 and oy > 1e-6:
                    return False
            return True

        for outer in range(50):
            moved = False
            for i in range(block_count):
                if i in locked_all:
                    continue
                xi, yi, wi, hi = positions[i]
                for j in locked_all:
                    xj, yj, wj, hj = positions[j]
                    ox = max(0.0, min(xi + wi, xj + wj) - max(xi, xj))
                    oy = max(0.0, min(yi + hi, yj + hj) - max(yi, yj))
                    if ox > 1e-6 and oy > 1e-6:
                        if ox <= oy:
                            cand_x = xj - wi - 1.0 if xi < xj else xj + wj + 1.0
                            cand_y = yi
                        else:
                            cand_x = xi
                            cand_y = yj - hi - 1.0 if yi < yj else yj + hj + 1.0
                        if fits(cand_x, cand_y, wi, hi, i):
                            positions[i][0] = cand_x
                            positions[i][1] = cand_y
                            xi, yi = cand_x, cand_y
                            moved = True
            if not moved:
                break

        # 4g. De-overlap pass 2: greedy candidate search
        for i in range(block_count):
            if i in locked_all:
                continue
            wi, hi = positions[i][2], positions[i][3]
            xi, yi = positions[i][0], positions[i][1]
            still_overlap = False
            for j in locked_all:
                xj, yj, wj, hj = positions[j]
                ox = max(0.0, min(xi + wi, xj + wj) - max(xi, xj))
                oy = max(0.0, min(yi + hi, yj + hj) - max(yi, yj))
                if ox > 1e-6 and oy > 1e-6:
                    still_overlap = True
                    break
            if not still_overlap:
                continue
            best = None
            candidates = set()
            candidates.add((0.0, 0.0))
            for j in range(block_count):
                if j == i:
                    continue
                xj, yj, wj, hj = positions[j]
                for cx, cy in [
                    (xj + wj, yj), (xj - wi, yj),
                    (xj, yj + hj), (xj, yj - hi),
                    (xj + wj, yj - hi), (xj - wi, yj - hi),
                    (xj + wj, yj + hj), (xj - wi, yj + hj),
                ]:
                    candidates.add((cx, cy))
            for (x, y) in candidates:
                if x < -1e6 or y < -1e6:
                    continue
                if fits(x, y, wi, hi, i):
                    if best is None or (y, x) < (best[1], best[0]):
                        best = (x, y)
            if best is not None:
                positions[i][0] = best[0]
                positions[i][1] = best[1]

        # 4h. Bbox re-normalize
        x_min = min(p[0] for p in positions)
        y_min = min(p[1] for p in positions)
        for k in range(block_count):
            positions[k][0] -= x_min
            positions[k][1] -= y_min

        # 4i. Re-override preplaced (bbox shift may have moved them)
        if target_positions is not None and constraints is not None:
            nc = constraints.shape[1] if constraints.dim() > 1 else 0
            for i in range(block_count):
                is_preplaced = nc > 1 and float(constraints[i, 1]) != 0
                if is_preplaced:
                    tp = target_positions[i]
                    positions[i][0] = float(tp[0])
                    positions[i][1] = float(tp[1])
                    positions[i][2] = float(tp[2])
                    positions[i][3] = float(tp[3])

        # 4j. Soft area fix: scale free block (w, h) so w*h matches area_target
        for i in range(block_count):
            if i in locked_all:
                continue
            if constraints is not None:
                nc = constraints.shape[1] if constraints.dim() > 1 else 0
                is_fixed = nc > 0 and float(constraints[i, 0]) != 0
                if is_fixed:
                    continue
            area = float(area_targets[i]) if area_targets[i] > 0 else 0.0
            if area <= 0:
                continue
            w, h = positions[i][2], positions[i][3]
            actual = w * h
            if actual <= 0:
                s = math.sqrt(area)
                positions[i][2] = s
                positions[i][3] = s
                continue
            rel_err = abs(actual - area) / area
            if rel_err > 0.005:
                scale = math.sqrt(area / actual)
                positions[i][2] = w * scale
                positions[i][3] = h * scale

        # 4k. Boundary soft fix: nudge blocks with boundary bit to edge
        if constraints is not None:
            nc = constraints.shape[1] if constraints.dim() > 1 else 0
            if nc > 4:
                # bbox
                x_max = max(p[0] + p[2] for p in positions)
                y_max = max(p[1] + p[3] for p in positions)
                for i in range(block_count):
                    if i in locked_all:
                        continue
                    bc = int(constraints[i, 4])
                    if bc == 0:
                        continue
                    wi, hi = positions[i][2], positions[i][3]
                    xi, yi = positions[i][0], positions[i][1]
                    new_xi, new_yi = xi, yi
                    if bc & 1:  # left
                        new_xi = 0.0
                    if bc & 2:  # right
                        new_xi = max(0.0, x_max - wi)
                    if bc & 4:  # top
                        new_yi = max(0.0, y_max - hi)
                    if bc & 8:  # bottom
                        new_yi = 0.0
                    # Only apply if it doesn't create an overlap with locked blocks
                    if (new_xi, new_yi) != (xi, yi) and fits(new_xi, new_yi, wi, hi, i):
                        positions[i][0] = new_xi
                        positions[i][1] = new_yi

        # Final bbox re-normalize (after boundary shifts)
        x_min = min(p[0] for p in positions)
        y_min = min(p[1] for p in positions)
        for k in range(block_count):
            positions[k][0] -= x_min
            positions[k][1] -= y_min

        # Final re-override preplaced
        if target_positions is not None and constraints is not None:
            nc = constraints.shape[1] if constraints.dim() > 1 else 0
            for i in range(block_count):
                is_preplaced = nc > 1 and float(constraints[i, 1]) != 0
                if is_preplaced:
                    tp = target_positions[i]
                    positions[i][0] = float(tp[0])
                    positions[i][1] = float(tp[1])
                    positions[i][2] = float(tp[2])
                    positions[i][3] = float(tp[3])

        return [tuple(p) for p in positions]

    # -----------------------------------------------------------------
    def _sa_only_solve(self, block_count, area_targets, b2b, p2b, pins,
                       constraints, target_positions):
        """SA-only fallback when no DiT checkpoint is available."""
        # Use sqrt(area) for all blocks
        widths, heights = [], []
        locked = set()
        fixed_w, fixed_h = {}, {}
        preplaced_root = 0
        if target_positions is not None and constraints is not None:
            nc = constraints.shape[1] if constraints.dim() > 1 else 0
            for i in range(block_count):
                is_fixed = nc > 0 and float(constraints[i, 0]) != 0
                is_preplaced = nc > 1 and float(constraints[i, 1]) != 0
                if is_preplaced:
                    locked.add(i)
                    preplaced_root = i
                elif is_fixed:
                    locked.add(i)
                    fixed_w[i] = float(target_positions[i, 2])
                    fixed_h[i] = float(target_positions[i, 3])
        for i in range(block_count):
            if i in fixed_w:
                widths.append(fixed_w[i])
                heights.append(fixed_h[i])
            else:
                a = float(area_targets[i]) if area_targets[i] > 0 else 1.0
                s = math.sqrt(a)
                widths.append(s)
                heights.append(s)

        tree = BStarTree(block_count, widths, heights, root_index=preplaced_root, locked=locked)
        current_positions = tree.pack()
        current_cost = self._sa_cost(current_positions, b2b, p2b, pins)
        best_positions, best_cost = current_positions, current_cost
        t0 = time.time()
        temp = SA_INITIAL_TEMP
        while temp > 1.0:
            if time.time() - t0 > SA_TIME_BUDGET_S:
                break
            for _ in range(SA_MOVES_PER_TEMP):
                if time.time() - t0 > SA_TIME_BUDGET_S:
                    break
                old_tree = tree.copy()
                free = [i for i in range(block_count) if i not in locked]
                if not free:
                    break
                b = random.choice(free)
                if random.random() < 0.5:
                    tree.move_rotate(b)
                else:
                    tree.move_delete_insert(b)
                new_positions = tree.pack()
                new_cost = self._sa_cost(new_positions, b2b, p2b, pins)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-3)):
                    current_positions = new_positions
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_positions = new_positions
                else:
                    tree = old_tree
            temp *= SA_COOLING_RATE

        return self._postprocess(best_positions, block_count, preplaced_root,
                                  fixed_w, fixed_h, target_positions, constraints,
                                  area_targets)
