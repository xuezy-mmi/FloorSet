#!/usr/bin/env python3
"""
dit_optimizer_hd_sa.py - Hybrid: diffusion initializer + SA refinement.

Idea (borrowed in spirit from HouseDiffusion's evaluator, which always
post-processes the sampled polygons to fix overlaps and snap to constraints):

  1. Use `dit_optimizer_hd.MyOptimizer` to obtain an initial diffusion
     layout. The diffusion model is good at *where to put blocks
     relative to each other* (low HPWL) but tends to leave small
     overlap/area-tolerance violations.
  2. Run a short simulated-annealing pass on top of the diffusion
     output, using the differentiable contest cost (HPWL + bbox area)
     as the objective. Moves that worsen the cost are accepted with the
     usual Metropolis probability.
  3. Hard-constraint enforcement (fixed/preplaced/area tolerance) and
     overlap removal happen on every move.

The two-stage design is one of the simplest ways to combine ML and
classical search, and is reported to be highly effective for floorplan
problems (see e.g. "ML-Guided SA" papers and HouseDiffusion's own
post-processing step).
"""
import math
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent))
from iccad2026_evaluate import FloorplanOptimizer
from dit_optimizer_hd import MyOptimizer as HDDiffusionOptimizer
from dit_utils_hd import (
    enforce_hard_constraints,
    fix_area_tolerance,
    remove_overlaps,
)


# ----------------------------------------------------------------------------
# Vectorized cost (cheap & on GPU) for SA scoring
# ----------------------------------------------------------------------------
def _hpwl_cost(positions: torch.Tensor, b2b: torch.Tensor) -> float:
    if b2b is None or b2b.numel() == 0:
        return 0.0
    m = b2b[:, 0] >= 0
    if not m.any():
        return 0.0
    e = b2b[m]
    i = e[:, 0].long()
    j = e[:, 1].long()
    w = e[:, 2]
    cx = positions[:, 0] + positions[:, 2] / 2
    cy = positions[:, 1] + positions[:, 3] / 2
    return float((w * (cx[i].sub(cx[j]).abs() + cy[i].sub(cy[j]).abs())).sum().item())


def _bbox_area_cost(positions: torch.Tensor) -> float:
    if positions.shape[0] == 0:
        return 0.0
    x = positions[:, 0]
    y = positions[:, 1]
    w = positions[:, 2]
    h = positions[:, 3]
    return float(((x.max() - x.min()).clamp(min=0) *
                  (y.max() - y.min()).clamp(min=0)).item())


# ----------------------------------------------------------------------------
# Hybrid optimizer
# ----------------------------------------------------------------------------
class MyOptimizer(FloorplanOptimizer):
    """Diffusion initialization + short SA refinement."""

    def __init__(self, verbose: bool = False, sa_iters: int = 300,
                 sa_t0: float = 50.0, sa_cooling: float = 0.97,
                 sa_move_scale: float = 30.0):
        super().__init__(verbose)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use HD diffusion for initialization
        self._diffusion = HDDiffusionOptimizer(verbose=verbose, sampler="ddpm",
                                                num_inference_steps=250)
        # SA hyperparams
        self.sa_iters = sa_iters
        self.sa_t0 = sa_t0
        self.sa_cooling = sa_cooling
        self.sa_move_scale = sa_move_scale
        if self.verbose:
            print(f"[HD-SA] SA iters={sa_iters} t0={sa_t0} "
                  f"cooling={sa_cooling} move_scale={sa_move_scale}")

    # ------------------------------------------------------------------
    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor] = None,
    ) -> List[Tuple[float, float, float, float]]:

        # ---- 1. diffusion init ----
        init_positions = self._diffusion.solve(
            block_count, area_targets, b2b_connectivity,
            p2b_connectivity, pins_pos, constraints, target_positions,
        )
        # if diffusion failed, _diffusion already returned SA fallback
        # either way we refine with SA on top
        if not init_positions or len(init_positions) != block_count:
            return init_positions

        # ---- 2. SA refinement ----
        return self._sa_refine(
            init_positions, area_targets, b2b_connectivity,
            p2b_connectivity, pins_pos, constraints, target_positions,
        )

    # ------------------------------------------------------------------
    def _sa_refine(
        self,
        init_positions: List[Tuple[float, float, float, float]],
        area_targets: torch.Tensor,
        b2b: torch.Tensor,
        p2b: torch.Tensor,
        pins: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: Optional[torch.Tensor],
    ) -> List[Tuple[float, float, float, float]]:

        N = len(init_positions)
        pos = [list(p) for p in init_positions]

        # determine which blocks are locked
        locked = [False] * N
        if target_positions is not None:
            for i in range(N):
                if i < target_positions.shape[0] and target_positions[i, 0] != -1:
                    locked[i] = True   # preplaced: x, y frozen
                # fixed-shape: only w, h frozen; x, y still movable

        # compute current cost
        cur_pos = torch.tensor(pos, dtype=torch.float32)
        cur_cost = _hpwl_cost(cur_pos, b2b) + 0.01 * _bbox_area_cost(cur_pos)
        best_pos = [tuple(p) for p in pos]
        best_cost = cur_cost

        T = self.sa_t0
        for it in range(self.sa_iters):
            # pick a random non-preplaced block
            i = random.randrange(N)
            if locked[i]:
                T *= self.sa_cooling
                continue

            old = pos[i][:]
            move = random.choice(["translate", "swap", "resize"])
            if move == "translate":
                dx = random.gauss(0, self.sa_move_scale)
                dy = random.gauss(0, self.sa_move_scale)
                pos[i][0] = max(0, pos[i][0] + dx)
                pos[i][1] = max(0, pos[i][1] + dy)
            elif move == "swap" and N > 1:
                j = random.randrange(N)
                if j != i and not locked[j]:
                    pos[i][0], pos[j][0] = pos[j][0], pos[i][0]
                    pos[i][1], pos[j][1] = pos[j][1], pos[i][1]
            else:  # resize (only free blocks)
                if target_positions is not None and i < target_positions.shape[0] \
                        and target_positions[i, 2] != -1:
                    pass  # fixed-shape: cannot resize
                else:
                    a = float(area_targets[i]) if area_targets[i] > 0 else old[2] * old[3]
                    asp = random.uniform(0.5, 2.0)
                    pos[i][2] = math.sqrt(a * asp)
                    pos[i][3] = math.sqrt(a / asp)

            # post-process
            cand = [tuple(p) for p in pos]
            cand = fix_area_tolerance(cand, area_targets, constraints, 0.01)
            cand = remove_overlaps(cand, locked=locked, n_iters=10)
            if target_positions is not None:
                cand_t = torch.tensor(cand, dtype=torch.float32)
                cand_t = enforce_hard_constraints(
                    cand_t, target_positions, constraints,
                )
                cand = [tuple(map(float, p)) for p in cand_t.tolist()]

            cand_pos = torch.tensor(cand, dtype=torch.float32)
            new_cost = _hpwl_cost(cand_pos, b2b) + 0.01 * _bbox_area_cost(cand_pos)
            dE = new_cost - cur_cost
            if dE < 0 or random.random() < math.exp(-dE / max(T, 1e-3)):
                cur_cost = new_cost
                pos = [list(p) for p in cand]
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_pos = [tuple(p) for p in pos]
            else:
                pos[i] = old

            T *= self.sa_cooling
            if T < 1e-3:
                T = 1e-3

        # final clean-up
        best_pos = fix_area_tolerance(best_pos, area_targets, constraints, 0.01)
        best_pos = remove_overlaps(best_pos, n_iters=50)
        if target_positions is not None:
            best_t = torch.tensor(best_pos, dtype=torch.float32)
            best_t = enforce_hard_constraints(
                best_t, target_positions, constraints,
            )
            best_pos = [tuple(map(float, p)) for p in best_t.tolist()]
        return best_pos
