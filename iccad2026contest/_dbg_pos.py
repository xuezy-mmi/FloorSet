"""Debug: check overlap details for a single test case."""
import sys
sys.path.insert(0, "/home/xzy/eda/FloorSet/iccad2026contest")
sys.path.insert(0, "/home/xzy/eda/FloorSet")
import torch
import numpy as np

from iccad2026_evaluate import (
    FloorplanDatasetLiteTest, test_floorplan_collate,
    check_overlap,
)
from dit_optimizer_v3 import MyOptimizer

opt = MyOptimizer(verbose=False)
ds = FloorplanDatasetLiteTest("/home/xzy/eda/FloorSet")

sample = ds[0]
batch = test_floorplan_collate([sample])
(area_t, b2b, p2b, pins, constraints), (fp_sol, metrics) = batch
area_t = area_t[0]; b2b = b2b[0]; p2b = p2b[0]
pins = pins[0]; constraints = constraints[0]
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

positions = opt.solve(block_count, area_t, b2b, p2b, pins,
                       constraints, target_positions=target_positions_np)

print(f"block_count={block_count}")
print("First 3 positions:", positions[:3])

# Find overlapping pairs
n_overlap = 0
for i in range(len(positions)):
    for j in range(i+1, len(positions)):
        x1, y1, w1, h1 = positions[i]
        x2, y2, w2, h2 = positions[j]
        ox = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        oy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        if ox > 1e-6 and oy > 1e-6:
            n_overlap += 1
            print(f"  overlap: {i}={positions[i]}  {j}={positions[j]}  ox={ox:.6f}  oy={oy:.6f}")

print(f"Total overlap_violations: {n_overlap}")
print(f"check_overlap: {check_overlap(positions)}")
