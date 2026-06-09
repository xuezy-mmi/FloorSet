#!/usr/bin/env python3
"""
dit_optimizer_v2_sa.py - Fast constraint-aware floorplan optimizer.

Strategy:
  1. Place all locked (fixed / preplaced) blocks at their target positions.
  2. For each free block, place it using a B*-tree SA but with a much
     smaller iteration budget so the total wall-clock per test case is
     bounded (under 2 seconds even for 120 blocks).
  3. Locked blocks' (w, h) are immutable so the SA never corrupts them.
  4. The preplaced block is the SA's root, so after SA we shift the layout
     so it lands at the target (x, y). Fixed blocks have (w, h) overridden
     at the end.
"""
import math
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))
from iccad2026_evaluate import (
    FloorplanOptimizer, calculate_hpwl_b2b, calculate_hpwl_p2b, calculate_bbox_area,
)


class BStarTree:
    """B*-tree with optional locked-blocks protection."""
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

    def move_swap(self, b1, b2):
        if b1 == self.root or b2 == self.root:
            return
        if b1 in self.locked or b2 in self.locked:
            return
        self.widths[b1], self.widths[b2] = self.widths[b2], self.widths[b1]
        self.heights[b1], self.heights[b2] = self.heights[b2], self.heights[b1]

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


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.initial_temp = 60.0
        self.final_temp = 1.0
        self.cooling_rate = 0.9
        self.moves_per_temp = 8
        self.time_budget_s = 1.5  # per solve()

    def _sa_cost(self, positions, b2b, p2b, pins):
        return (calculate_hpwl_b2b(positions, b2b)
                + calculate_hpwl_p2b(positions, p2b, pins)
                + calculate_bbox_area(positions) * 0.01)

    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity,
              pins_pos, constraints, target_positions=None):
        t0 = time.time()

        # Identify locked blocks
        locked = set()
        preplaced_root = 0
        fixed_w, fixed_h = {}, {}
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

        widths, heights = [], []
        for i in range(block_count):
            if i in locked:
                tp = target_positions[i]
                widths.append(float(tp[2]))
                heights.append(float(tp[3]))
            else:
                area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
                w = h = math.sqrt(area)
                widths.append(w)
                heights.append(h)

        tree = BStarTree(block_count, widths, heights, root_index=preplaced_root,
                         locked=locked)
        current_positions = tree.pack()
        current_cost = self._sa_cost(current_positions, b2b_connectivity,
                                      p2b_connectivity, pins_pos)
        best_positions = current_positions
        best_cost = current_cost

        temp = self.initial_temp
        while temp > self.final_temp:
            if time.time() - t0 > self.time_budget_s:
                break
            for _ in range(self.moves_per_temp):
                if time.time() - t0 > self.time_budget_s:
                    break
                old_tree = tree.copy()
                free = [i for i in range(block_count) if i not in locked]
                if not free:
                    break
                b = random.choice(free)
                move = random.randint(0, 1)
                if move == 0:
                    tree.move_rotate(b)
                else:
                    tree.move_delete_insert(b)
                new_positions = tree.pack()
                new_cost = self._sa_cost(new_positions, b2b_connectivity,
                                          p2b_connectivity, pins_pos)
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_positions = new_positions
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_positions = new_positions
                else:
                    tree = old_tree
            temp *= self.cooling_rate

        positions = [list(p) for p in best_positions]

        # Translate so the preplaced root lands at its target (x, y)
        if preplaced_root in locked and target_positions is not None:
            tp = target_positions[preplaced_root]
            dx = float(tp[0]) - positions[preplaced_root][0]
            dy = float(tp[1]) - positions[preplaced_root][1]
            for k in range(block_count):
                positions[k][0] += dx
                positions[k][1] += dy

        # Override (w, h) for fixed blocks
        for i in fixed_w:
            positions[i][2] = fixed_w[i]
            positions[i][3] = fixed_h[i]

        # Shift to put bbox origin at (0, 0) FIRST (so locked blocks stay locked)
        x_min = min(p[0] for p in positions)
        y_min = min(p[1] for p in positions)
        for k in range(block_count):
            positions[k][0] -= x_min
            positions[k][1] -= y_min

        # Explicit (x, y, w, h) override for ALL preplaced blocks
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

        # ====== Soft-constraint pre-de-overlap: MIB ======
        # Make MIB group blocks share the same w, h BEFORE the de-overlap,
        # so the de-overlap accounts for the new dimensions.
        if constraints is not None and block_count > 0:
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

        # De-overlap pass: free blocks that overlap with a locked block
        # get pushed out of the overlap. This is necessary because the SA
        # may have placed free blocks at the SA position of the preplaced
        # root, which is now occupied by the preplaced target.
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

        # First pass: simple shove-out (fast)
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

        # Second pass: for any free block still overlapping a locked block,
        # find the smallest (y, x) valid candidate by abutting each
        # already-placed block.  Guarantees feasibility.
        for i in range(block_count):
            if i in locked_all:
                continue
            wi, hi = positions[i][2], positions[i][3]
            # Quick check: does this block still overlap any locked block?
            still_overlap = False
            for j in locked_all:
                xj, yj, wj, hj = positions[j]
                xi, yi = positions[i][0], positions[i][1]
                ox = max(0.0, min(xi + wi, xj + wj) - max(xi, xj))
                oy = max(0.0, min(yi + hi, yj + hj) - max(yi, yj))
                if ox > 1e-6 and oy > 1e-6:
                    still_overlap = True
                    break
            if not still_overlap:
                continue
            # Greedy: try candidates near every placed block
            best = None
            candidates = set()
            candidates.add((0.0, 0.0))
            for j in range(block_count):
                if j == i:
                    continue
                xj, yj, wj, hj = positions[j]
                candidates.add((xj + wj, yj))
                candidates.add((xj - wi, yj))
                candidates.add((xj, yj + hj))
                candidates.add((xj, yj - hi))
                candidates.add((xj + wj, yj - hi))
                candidates.add((xj - wi, yj - hi))
                candidates.add((xj + wj, yj + hj))
                candidates.add((xj - wi, yj + hj))
            for (x, y) in candidates:
                if x < -1e6 or y < -1e6:
                    continue
                if fits(x, y, wi, hi, i):
                    if best is None or (y, x) < (best[1], best[0]):
                        best = (x, y)
            if best is not None:
                positions[i][0] = best[0]
                positions[i][1] = best[1]

        # Re-shift bbox to origin (the de-overlap may have moved things)
        x_min = min(p[0] for p in positions)
        y_min = min(p[1] for p in positions)
        for k in range(block_count):
            positions[k][0] -= x_min
            positions[k][1] -= y_min

        # Post-process: re-explicit override for preplaced (bbox shift may
        # have moved them slightly)
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
