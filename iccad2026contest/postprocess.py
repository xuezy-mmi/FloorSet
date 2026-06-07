# postprocess.py
import torch
import numpy as np

def adjust_areas(positions, area_targets, fixed_mask=None, preplaced_mask=None, tolerance=0.01):
    """
    Rescale soft blocks to meet area target within tolerance.
    positions: [N, 4] (x,y,w,h)
    area_targets: [N] target area
    fixed_mask: bool [N] True for fixed-shape blocks (no change)
    preplaced_mask: bool [N] True for preplaced blocks (no change)
    Returns adjusted positions.
    """
    positions = positions.clone()
    for i in range(len(positions)):
        if fixed_mask is not None and fixed_mask[i]:
            continue
        if preplaced_mask is not None and preplaced_mask[i]:
            continue
        area = positions[i, 2] * positions[i, 3]
        target = area_targets[i]
        if target <= 0:
            continue
        ratio = (target / (area + 1e-6)).sqrt()
        # Clamp ratio to avoid extreme shape change
        ratio = torch.clamp(ratio, 0.5, 2.0)
        positions[i, 2] *= ratio
        positions[i, 3] *= ratio
    return positions

def resolve_overlaps(positions, max_iter=50, shift_factor=0.1):
    """
    Iterative force-based overlap removal.
    positions: [N, 4] (x,y,w,h)
    Returns positions with no overlap (touching allowed).
    """
    positions = positions.clone().float()
    N = len(positions)
    for _ in range(max_iter):
        overlap = False
        forces = torch.zeros_like(positions[:, :2])  # only shift x,y
        for i in range(N):
            x1, y1, w1, h1 = positions[i]
            for j in range(i+1, N):
                x2, y2, w2, h2 = positions[j]
                if x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1:
                    continue
                overlap = True
                # compute overlap rectangle
                overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
                overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)
                # push apart
                dx = (x1 + w1/2) - (x2 + w2/2)
                dy = (y1 + h1/2) - (y2 + h2/2)
                if abs(dx) > 1e-6:
                    push = overlap_x * shift_factor * (1 if dx > 0 else -1)
                    forces[i, 0] += push
                    forces[j, 0] -= push
                if abs(dy) > 1e-6:
                    push = overlap_y * shift_factor * (1 if dy > 0 else -1)
                    forces[i, 1] += push
                    forces[j, 1] -= push
        if not overlap:
            break
        # apply forces (clamp to avoid out-of-canvas)
        positions[:, :2] += forces
        # clip to non-negative
        positions[:, 0] = torch.clamp(positions[:, 0], min=0.0)
        positions[:, 1] = torch.clamp(positions[:, 1], min=0.0)
    return positions

def enforce_fixed_preplaced(positions, target_positions, constraints):
    """
    Overwrite positions of fixed-shape/preplaced blocks from target_positions.
    constraints: [N,5] tensor: col0=fixed, col1=preplaced
    target_positions: [N,4] (x,y,w,h) or None
    Returns modified positions.
    """
    if target_positions is None:
        return positions
    N = len(positions)
    for i in range(N):
        is_fixed = (constraints[i, 0] != 0).item()
        is_preplaced = (constraints[i, 1] != 0).item()
        if is_preplaced:
            positions[i, 0] = target_positions[i, 0]
            positions[i, 1] = target_positions[i, 1]
            positions[i, 2] = target_positions[i, 2]
            positions[i, 3] = target_positions[i, 3]
        elif is_fixed:
            positions[i, 2] = target_positions[i, 2]
            positions[i, 3] = target_positions[i, 3]
    return positions

def postprocess(positions, area_targets, constraints, target_positions=None):
    """
    Full postprocessing to satisfy hard constraints:
    - fixed/preplaced dimensions/locations
    - area tolerance for soft blocks
    - no overlaps
    """
    N = positions.shape[0]
    device = positions.device
    constraints = constraints.to(device) if constraints is not None else torch.zeros(N,5, device=device)
    target_positions = target_positions.to(device) if target_positions is not None else None

    # Step 1: enforce fixed/preplaced
    positions = enforce_fixed_preplaced(positions, target_positions, constraints)

    # Step 2: adjust areas of soft blocks
    fixed_mask = (constraints[:, 0] != 0).bool()
    preplaced_mask = (constraints[:, 1] != 0).bool()
    positions = adjust_areas(positions, area_targets, fixed_mask, preplaced_mask)

    # Step 3: remove overlaps
    positions = resolve_overlaps(positions, max_iter=100, shift_factor=0.05)

    # Step 4: final area check (if still exceeds tolerance, scale again)
    positions = adjust_areas(positions, area_targets, fixed_mask, preplaced_mask)

    return positions