#!/usr/bin/env python3
"""
dit_optimizer_v2_sa.py - Hybrid SA optimizer (v2 baseline + hard-constraint fix)

Strategy:
  - Run the B*-tree SA to choose good (w, h) for each free block.
  - Locked (fixed / preplaced) blocks have immutable (w, h) so the SA
    never corrupts their dimensions.
  - The preplaced block is the B*-tree root, so pack() places it at the
    SA's origin. After SA finishes, the whole layout is translated so
    the preplaced block lands exactly at its target (x, y).
  - Fixed-shape blocks keep their SA-assigned (x, y) but get their (w, h)
    set to the target values.
"""
import math
import random
import sys
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
        self.initial_temp = 80.0
        self.final_temp = 1.0
        self.cooling_rate = 0.92
        self.moves_per_temp = 15

    def _sa_cost(self, positions, b2b, p2b, pins):
        return (calculate_hpwl_b2b(positions, b2b)
                + calculate_hpwl_p2b(positions, p2b, pins)
                + calculate_bbox_area(positions) * 0.01)

    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity,
              pins_pos, constraints, target_positions=None):
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
            for _ in range(self.moves_per_temp):
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

        # Explicit (x, y, w, h) override for ALL preplaced blocks
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
