#!/usr/bin/env python3
"""
Usage:
    python visualize_json.py xxx_result.json
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def draw_layout(ax, positions, block_count, title):
    """Draw a list of (x, y, w, h) rectangles on `ax`."""
    ax.set_title(title, fontsize=10)
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
            ha="center", va="center", fontsize=6,
        )
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle=":", alpha=0.3)


def plot_test_result(test_result, save_dir, prefix, dpi=150, figsize=(12, 8)):
    """Plot a single test result and save the figure."""
    test_id = test_result["test_id"]
    block_count = test_result["block_count"]
    positions = test_result.get("positions")
    is_feasible = test_result.get("is_feasible", None)
    hpwl_gap = test_result.get("hpwl_gap", float("nan"))
    area_gap = test_result.get("area_gap", float("nan"))
    violations_relative = test_result.get("violations_relative", float("nan"))
    runtime_seconds = test_result.get("runtime_seconds", float("nan"))
    cost = test_result.get("cost", float("nan"))
    error = test_result.get("error")

    if positions is None or len(positions) == 0:
        print(f"  test {test_id:03d}: no positions, skipping")
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    feasible_str = "FEASIBLE" if is_feasible else "INFEASIBLE"
    title = (
        f"Test {test_id}  |  blocks={block_count}  |  {feasible_str}\n"
        f"cost={cost:.4f}  |  hpwl_gap={hpwl_gap:.4f}  |  area_gap={area_gap:.4f}  |  "
        f"V_rel={violations_relative:.4f}  |  runtime={runtime_seconds:.3f}s"
    )
    if error:
        title += f"\nerror: {error}"

    draw_layout(ax, positions, block_count, title)

    plt.tight_layout()

    out_name = f"{prefix}_test{test_id:03d}.png"
    save_path = save_dir / out_name
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"  -> saved {save_path}")
    plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_json.py <result.json>")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"Error: file not found: {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    submission_name = data.get("submission_name", "unknown")

    test_results = data.get("test_results", [])

    if not test_results:
        print("Error: no test_results found in JSON")
        sys.exit(1)


    script_dir = Path(__file__).parent
    fig_dir = script_dir / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Submission: {submission_name}")
    print(f"Total test cases: {len(test_results)}")
    print(f"Output directory: {fig_dir}")

    for result in test_results:
        plot_test_result(result, fig_dir, submission_name)

    print(f"\nDone. {len(test_results)} image(s) saved to {fig_dir}")


if __name__ == "__main__":
    main()