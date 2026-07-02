"""Smoke test: run the optimizer on a few real validation cases
to verify is_feasible=True on at least the basic cases."""
import sys, time
sys.path.insert(0, "/home/xzy/eda/FloorSet/iccad2026contest")
sys.path.insert(0, "/home/xzy/eda/FloorSet")
import torch
import numpy as np

from iccad2026_evaluate import (
    FloorplanDatasetLiteTest, test_floorplan_collate,
    evaluate_solution, M_PENALTY,
)
from dit_optimizer_v3 import MyOptimizer

opt = MyOptimizer(verbose=False)

ds = FloorplanDatasetLiteTest("/home/xzy/eda/FloorSet")

n_test = 5
n_feasible = 0
total_cost = 0.0

for i in range(n_test):
    sample = ds[i]
    batch = test_floorplan_collate([sample])
    (area_t, b2b, p2b, pins, constraints), (fp_sol, metrics) = batch
    area_t = area_t[0]
    b2b = b2b[0]
    p2b = p2b[0]
    pins = pins[0]
    constraints = constraints[0]
    block_count = int((area_t != -1).sum().item())

    polygons = fp_sol[0]
    target_positions_np = np.full((block_count, 4), -1.0, dtype=np.float32)
    for j in range(block_count):
        poly = polygons[j] if j < len(polygons) else None
        if poly is not None and len(poly) > 0:
            v = poly[poly[:, 0] != -1]
            if len(v) > 0:
                x_min = v[:, 0].min().item()
                y_min = v[:, 1].min().item()
                x_max = v[:, 0].max().item()
                y_max = v[:, 1].max().item()
                w = max(x_max - x_min, 1.0)
                h = max(y_max - y_min, 1.0)
                target_positions_np[j] = (x_min, y_min, w, h)

    t0 = time.time()
    positions = opt.solve(block_count, area_t, b2b, p2b, pins,
                           constraints, target_positions=target_positions_np)
    dt = time.time() - t0

    baseline_metrics = {
        'area': float(metrics[0][0]),
        'b2b_weighted_wl': float(metrics[0][6]),
        'p2b_weighted_wl': float(metrics[0][7]),
    }
    metrics_obj = evaluate_solution(
        solution={'positions': positions, 'runtime': dt},
        baseline_metrics=baseline_metrics,
        target_constraints=constraints[:block_count],
        b2b_connectivity=b2b,
        p2b_connectivity=p2b,
        pins_pos=pins,
        target_areas=area_t,
        target_positions=target_positions_np,
    )
    is_feasible = metrics_obj.is_feasible
    cost = M_PENALTY if not is_feasible else (
        1.0 + 0.5 * (metrics_obj.hpwl_gap + metrics_obj.area_gap)
    )
    n_feasible += int(is_feasible)
    total_cost += cost
    print(
        f"  test {i:3d}: n={block_count:3d}  "
        f"feasible={is_feasible}  "
        f"overlap_v={metrics_obj.overlap_violations}  "
        f"area_v={metrics_obj.area_violations}  "
        f"dim_v={metrics_obj.dimension_violations}  "
        f"hpwl_gap={metrics_obj.hpwl_gap:.3f}  "
        f"area_gap={metrics_obj.area_gap:.3f}  "
        f"cost={cost:.3f}  "
        f"t={dt:.2f}s"
    )

print(f"\n  {n_feasible}/{n_test} feasible  avg_cost={total_cost/max(n_test,1):.3f}")
