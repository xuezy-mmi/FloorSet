#!/usr/bin/env python3
"""
visualize_optimizer.py - Run an optimizer on the ICCAD 2026 validation set
and save side-by-side floorplan plots (ground truth vs. solution).

Mirrors the visualization in `iccad2026_evaluate.visualize_test_case` but
plots the *solution* alongside the ground truth and writes PNGs to
`iccad2026contest/fig/`.

Edit the ``CONFIG`` block at the top and run:

    python visualize_optimizer.py
"""
import importlib.util
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from iccad2026_evaluate import (
    FloorplanDatasetLiteTest,
    calculate_bbox_area,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
)


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "optimizer_file":    "dit_optimizer_v3.py",   # path (relative to iccad2026contest/) of the optimizer
    "data_path":         "../",                   # FloorSet data root for the validator
    "test_ids":          [0, 1, 2, 3, 4, 20, 40, 60, 80, 99],         # validation case indices to plot
    "out_dir":           "fig",                   # output directory for PNGs
    "filename_prefix":   "v3",                    # output filename prefix: <prefix>_<test_id>.png
    "show":              False,                   # whether to also display the figures
    "dpi":               150,
    "figsize":           (14, 7),                 # (W, H) per figure
}


# =============================================================================
# Optimizer loading (same importlib trick as iccad2026_evaluate)
# =============================================================================
def load_optimizer(optimizer_file: str):
    """Import the user's optimizer file and return a fresh MyOptimizer instance."""
    file_path = Path(optimizer_file)
    if not file_path.is_absolute():
        file_path = Path(__file__).parent / optimizer_file
    if not file_path.exists():
        raise FileNotFoundError(f"Optimizer file not found: {file_path}")
    spec = importlib.util.spec_from_file_location("user_optimizer", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # pick the first non-FloorplanOptimizer subclass named MyOptimizer / Optimizer / ContestOptimizer
    from iccad2026_evaluate import FloorplanOptimizer
    candidate = None
    for name in dir(module):
        obj = getattr(module, name)
        if not isinstance(obj, type):
            continue
        if obj is FloorplanOptimizer:
            continue
        if issubclass(obj, FloorplanOptimizer):
            candidate = obj
            break
    if candidate is None:
        for fallback in ("MyOptimizer", "Optimizer", "ContestOptimizer"):
            if hasattr(module, fallback):
                candidate = getattr(module, fallback)
                break
    if candidate is None:
        raise RuntimeError(
            f"No FloorplanOptimizer subclass found in {optimizer_file}"
        )
    return candidate


# =============================================================================
# Helpers (mostly copied from iccad2026_evaluate.visualize_test_case)
# =============================================================================
def _gt_positions(polygons, block_count: int) -> List[Tuple[float, float, float, float]]:
    out = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            out.append((
                float(x_min), float(y_min),
                float(x_max - x_min), float(y_max - y_min),
            ))
        else:
            out.append((0.0, 0.0, 1.0, 1.0))
    return out


def _hpwl(positions, b2b, p2b, pins) -> float:
    return (
        calculate_hpwl_b2b(positions, b2b)
        + calculate_hpwl_p2b(positions, p2b, pins)
    )


def _bbox(positions) -> float:
    return calculate_bbox_area(positions)


# =============================================================================
# Drawing
# =============================================================================
def _draw_layout(ax, positions, block_count, title, *, overlay_constraints=None):
    """Draw a list of (x, y, w, h) rectangles on `ax`."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ax.set_title(title)
    colors = plt.cm.tab20(np.linspace(0, 1, max(block_count, 1)))
    for i, (x, y, w, h) in enumerate(positions):
        rect = mpatches.Rectangle(
            (x, y), w, h,
            fill=True, facecolor=colors[i % 20], edgecolor="black",
            alpha=0.7, linewidth=0.5,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, str(i),
            ha="center", va="center", fontsize=7,
        )
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle=":", alpha=0.3)
    return ax


def plot_case(
    test_id: int,
    gt_positions: List[Tuple[float, float, float, float]],
    sol_positions: List[Tuple[float, float, float, float]],
    block_count: int,
    *,
    metrics: dict,
    save_path: Path,
    show: bool,
    figsize: Tuple[float, float],
    dpi: int,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ----- left: ground truth -----
    _draw_layout(
        axes[0], gt_positions, block_count,
        f"Ground Truth (test {test_id}, n={block_count})",
    )

    # ----- right: solution -----
    feasible = metrics.get("is_feasible", None)
    hpwl_gap = metrics.get("hpwl_gap", float("nan"))
    area_gap = metrics.get("area_gap", float("nan"))
    # "partial" cost: assumes V_rel=0 and RuntimeFactor=1.0 (i.e. the
    # best case the optimizer could achieve on the layout-quality axis).
    # The full contest cost also multiplies by exp(2·V_rel), which we
    # don't compute here because it requires the full evaluator.
    partial_cost = 1.0 + 0.5 * (hpwl_gap + area_gap)
    sol_title = (
        f"Solution  partial_cost={partial_cost:.2f}  "
        f"hpwl_gap={hpwl_gap:.2f}  area_gap={area_gap:.2f}  "
        f"feasible={feasible}"
    )
    _draw_layout(
        axes[1], sol_positions, block_count, sol_title,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"  -> saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main(cfg=CONFIG):
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.resolve()}")

    # --- load dataset ---
    data_path = cfg["data_path"]
    print(f"Loading validation dataset from {data_path} ...")
    dataset = FloorplanDatasetLiteTest(data_path)
    print(f"  {len(dataset)} validation cases available")

    # --- load optimizer class ---
    print(f"Loading optimizer from {cfg['optimizer_file']} ...")
    OptClass = load_optimizer(cfg["optimizer_file"])
    print(f"  optimizer class: {OptClass.__name__}")

    # --- iterate test cases ---
    summary = []
    for test_id in cfg["test_ids"]:
        print(f"\n=== test {test_id} ===")
        sample = dataset[test_id]
        inputs, labels = sample["input"], sample["label"]
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        fp_sol, metrics_sol = labels

        block_count = int((area_target != -1).sum().item())
        gt_positions = _gt_positions(fp_sol, block_count)

        # ---- fresh optimizer instance per test case (so any per-case
        # state is cleared) ----
        opt = OptClass(verbose=True)
        t0 = time.time()
        try:
            sol_positions = opt.solve(
                block_count=block_count,
                area_targets=area_target,
                b2b_connectivity=b2b_conn,
                p2b_connectivity=p2b_conn,
                pins_pos=pins_pos,
                constraints=constraints,
                target_positions=None,
            )
        except Exception as e:
            print(f"  optimizer failed: {e!r}")
            sol_positions = []
        runtime = time.time() - t0

        if not sol_positions or len(sol_positions) != block_count:
            print(f"  no usable solution (got {len(sol_positions) if sol_positions else 0} "
                  f"positions, expected {block_count})")
            metrics = {"is_feasible": False, "cost": 10.0,
                       "hpwl_gap": float("nan"), "area_gap": float("nan")}
        else:
            # compute hpwl / area gaps to display in the figure title
            hpwl_sol = _hpwl(sol_positions, b2b_conn, p2b_conn, pins_pos)
            area_sol = _bbox(sol_positions)
            hpwl_base = _hpwl(gt_positions, b2b_conn, p2b_conn, pins_pos)
            area_base = _bbox(gt_positions)
            hpwl_gap = max(0.0, (hpwl_sol - hpwl_base) / max(hpwl_base, 1e-9))
            area_gap = max(0.0, (area_sol - area_base) / max(area_base, 1e-9))
            # quick feasibility check: any negative coords / zero area
            feasible = all(
                (w > 0 and h > 0 and x >= 0 and y >= 0)
                for (x, y, w, h) in sol_positions
            )
            # cost = (1 + 0.5·(hpwl_gap + area_gap)) · exp(2·V_rel) ; here V_rel
            # is unknown without the full evaluator, so we just report raw gaps
            metrics = {
                "is_feasible": feasible,
                "hpwl_gap": hpwl_gap,
                "area_gap": area_gap,
            }

        # ---- draw ----
        save_path = out_dir / f"{cfg['filename_prefix']}_test{test_id:03d}.png"
        plot_case(
            test_id=test_id,
            gt_positions=gt_positions,
            sol_positions=sol_positions or [(0, 0, 1, 1)] * block_count,
            block_count=block_count,
            metrics=metrics,
            save_path=save_path,
            show=cfg["show"],
            figsize=tuple(cfg["figsize"]),
            dpi=int(cfg["dpi"]),
        )

        summary.append({"test_id": test_id, **metrics, "runtime": runtime})

    print("\n=== summary ===")
    for s in summary:
        print(f"  test {s['test_id']}: feasible={s.get('is_feasible')} "
              f"hpwl_gap={s.get('hpwl_gap', float('nan')):.3f} "
              f"area_gap={s.get('area_gap', float('nan')):.3f} "
              f"runtime={s['runtime']:.2f}s")


if __name__ == "__main__":
    main()
