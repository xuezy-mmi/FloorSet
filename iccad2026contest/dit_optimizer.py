# dit_optimizer.py
#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - DiT-based Optimizer
Uses pre-trained Diffusion Transformer to generate initial floorplan,
then post-processes to satisfy hard constraints.
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import FloorplanOptimizer, calculate_hpwl_b2b, calculate_hpwl_p2b, calculate_bbox_area
from dit_model import ConditionEncoder, DiffusionTransformer
from diffusion import GaussianDiffusion
from postprocess import postprocess

class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False, model_path: str = "checkpoints/final_model.pt"):
        super().__init__(verbose)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load pre-trained models."""
        self.cond_encoder = ConditionEncoder(feat_dim=64, hidden_dim=256).to(self.device)
        self.diffusion_model = DiffusionTransformer(dim=256, depth=12, num_heads=8, cond_dim=256).to(self.device)
        self.diffusion = GaussianDiffusion(num_timesteps=1000, schedule='cosine').to(self.device)

        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.cond_encoder.load_state_dict(checkpoint['cond_encoder'])
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model'])
        self.cond_encoder.eval()
        self.diffusion_model.eval()

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None
    ) -> List[Tuple[float, float, float, float]]:
        """
        Generate floorplan using DiT + postprocessing.
        """
        # Prepare input tensors (batch size = 1)
        area_target = area_targets.clone().unsqueeze(0).to(self.device)  # [1, K]
        b2b_conn = b2b_connectivity.clone().unsqueeze(0).to(self.device)  # [1, E, 3]
        # p2b_conn, pins_pos not used in condition encoder, but we keep
        constraints = constraints.clone().unsqueeze(0).to(self.device)  # [1, K, 5]
        target_pos = target_positions.clone().unsqueeze(0).to(self.device) if target_positions is not None else None

        K = area_target.shape[1]
        # Create mask for valid blocks
        valid_mask = (area_target[0] != -1)

        # Condition encoding
        with torch.no_grad():
            cond, mask = self.cond_encoder(area_target, constraints, b2b_conn, target_pos)

        # Generate layout (fast sampling, e.g., 50 steps)
        num_steps = 50
        with torch.no_grad():
            layout_noisy = self.diffusion.sample(self.diffusion_model, cond, mask, num_steps=num_steps)
        # layout_noisy: [1, K, 4] (x,y,w,h)

        # Take first batch, only valid blocks
        layout_pred = layout_noisy[0, :block_count]  # [block_count, 4]

        # Postprocess to satisfy hard constraints
        layout_final = postprocess(
            layout_pred,
            area_targets[:block_count].to(layout_pred.device),
            constraints[0, :block_count],
            target_positions[:block_count].to(layout_pred.device) if target_positions is not None else None
        )

        # Convert to list of tuples
        result = [(float(x), float(y), float(w), float(h)) for x, y, w, h in layout_final.cpu().numpy()]
        return result