#!/usr/bin/env python3
"""
dit_optimizer.py - Diffusion Transformer based floorplan optimizer
"""
import torch
import math
import sys
import random
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from iccad2026_evaluate import FloorplanOptimizer
from dit_model import DiffusionTransformer
from dit_utils import DiffusionScheduler, q_sample


class MyOptimizer(FloorplanOptimizer):
    """
    Diffusion Transformer optimizer.
    Loads pretrained model and performs DDPM sampling.
    """
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_steps = 1000
        self.norm_factor = 1000.0   # 必须与训练时一致
        self.model = None
        self.scheduler = None
        self._load_model_if_exists()
    
    def _load_model_if_exists(self):
        """加载预训练的扩散模型"""
        # 尝试加载模型权重，支持 model/ 目录或当前目录
        # model_path = Path(__file__).parent / "model" / "diffusion_final.pth"
        model_path = Path("/home/xzy/eda/model/diffusion_final.pth")
        if not model_path.exists():
            model_path = Path(__file__).parent / "diffusion_final.pth"
        
        if model_path.exists():
            self.model = DiffusionTransformer(
                dim=512, depth=8, heads=8, cond_dim=128, n_steps=self.n_steps
            ).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.scheduler = DiffusionScheduler(self.n_steps)
            if self.verbose:
                print(f"Loaded diffusion model from {model_path}")
        else:
            if self.verbose:
                print("No pretrained diffusion model found, using SA baseline")
            # 回退模式：不加载模型，但 scheduler 仍需定义（避免属性缺失）
            self.scheduler = DiffusionScheduler(self.n_steps)
    
    # def solve(
    #     self,
    #     block_count: int,
    #     area_targets: torch.Tensor,
    #     b2b_connectivity: torch.Tensor,
    #     p2b_connectivity: torch.Tensor,
    #     pins_pos: torch.Tensor,
    #     constraints: torch.Tensor,
    #     target_positions: torch.Tensor = None
    # ) -> List[Tuple[float, float, float, float]]:
    #     """Solve floorplanning problem."""
    #     if self.model is not None:
    #         return self._solve_with_diffusion(
    #             block_count, area_targets, b2b_connectivity,
    #             p2b_connectivity, pins_pos, constraints, target_positions
    #         )
    #     else:
    #         return self._solve_sa_baseline(
    #             block_count, area_targets, b2b_connectivity,
    #             p2b_connectivity, pins_pos, constraints, target_positions
    #         )

    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity,
          pins_pos, constraints, target_positions=None):
        print("="*50, flush=True)
        print(f"DEBUG: solve called with block_count = {block_count}", flush=True)
        print(f"DEBUG: area_targets shape = {area_targets.shape}, dtype={area_targets.dtype}", flush=True)
        print(f"DEBUG: b2b_connectivity shape = {b2b_connectivity.shape if b2b_connectivity is not None else None}", flush=True)
        print(f"DEBUG: p2b_connectivity shape = {p2b_connectivity.shape if p2b_connectivity is not None else None}", flush=True)
        print(f"DEBUG: pins_pos shape = {pins_pos.shape if pins_pos is not None else None}", flush=True)
        print(f"DEBUG: constraints shape = {constraints.shape if constraints is not None else None}", flush=True)
        print(f"DEBUG: target_positions type = {type(target_positions)}", flush=True)
        if target_positions is not None:
            print(f"DEBUG: target_positions shape = {target_positions.shape}", flush=True)
        
        try:
            # 您的原有逻辑
            if self.model is not None:
                result = self._solve_with_diffusion(
                    block_count, area_targets, b2b_connectivity,
                    p2b_connectivity, pins_pos, constraints, target_positions
                )
            else:
                result = self._solve_sa_baseline(
                    block_count, area_targets, b2b_connectivity,
                    p2b_connectivity, pins_pos, constraints, target_positions
                )
            print(f"DEBUG: solve returned {len(result)} positions", flush=True)
            return result
        except Exception as e:
            import traceback
            print("="*50, flush=True)
            print("EXCEPTION in solve:", str(e), flush=True)
            traceback.print_exc()
            print("="*50, flush=True)
            # 返回一个简单的可行布局，防止框架崩溃
            positions = []
            x = 0.0
            y = 0.0
            for i in range(block_count):
                area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
                w = h = math.sqrt(area)
                # 检查是否是固定/预置块，使用目标尺寸
                if target_positions is not None and target_positions[i, 2] != -1:
                    w = float(target_positions[i, 2])
                    h = float(target_positions[i, 3])
                if target_positions is not None and target_positions[i, 0] != -1:
                    x = float(target_positions[i, 0])
                    y = float(target_positions[i, 1])
                positions.append((x, y, w, h))
                x += w + 10

            return positions
    
    def _solve_with_diffusion(self, block_count, area_targets, b2b_conn,
                              p2b_conn, pins_pos, constraints, target_positions):
        """使用扩散模型推理"""
        # 1. 准备条件（batch_size=1）
        area = area_targets[:block_count].unsqueeze(0).to(self.device)
        # 注意：b2b_conn, p2b_conn, pins_pos 可能为空，需安全处理
        if b2b_conn is not None and b2b_conn.numel():
            b2b = b2b_conn.unsqueeze(0).to(self.device)
        else:
            b2b = torch.zeros(1, 0, 3, device=self.device)
        if p2b_conn is not None and p2b_conn.numel():
            p2b = p2b_conn.unsqueeze(0).to(self.device)
        else:
            p2b = torch.zeros(1, 0, 3, device=self.device)
        if pins_pos is not None and pins_pos.numel():
            pins = pins_pos.unsqueeze(0).to(self.device)
        else:
            pins = torch.zeros(1, 0, 2, device=self.device)
        constr = constraints[:block_count].unsqueeze(0).to(self.device)

        # 2. 初始化噪声布局 (w,h,x,y) 归一化空间
        x = torch.randn(1, block_count, 4, device=self.device)
        alpha_cumprod = self.scheduler.alpha_cumprod.to(self.device)

        # 3. DDPM 采样
        with torch.no_grad():
            for t in reversed(range(self.n_steps)):
                t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)
                pred_noise = self.model(x, t_tensor, area, b2b, p2b, pins, constr)
                
                alpha_t = alpha_cumprod[t]
                alpha_prev = alpha_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=self.device)
                beta_t = 1 - alpha_t / alpha_prev
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0
                x = (x - (beta_t / torch.sqrt(1 - alpha_t)) * pred_noise) / torch.sqrt(1 - beta_t) + torch.sqrt(beta_t) * noise

        # 4. 反归一化并转换到 (x, y, w, h) 顺序
        # x = x * self.norm_factor
        x = torch.clamp(x, -1.0, 1.0)  # 确保在合理范围
        x = x * self.norm_factor
        x = torch.clamp(x, 0.0, 1e6)   # 限制最大坐标/尺寸

        positions = []
        for i in range(block_count):
            w, h, px, py = x[0, i].cpu().numpy()
            # 确保尺寸为正
            w = max(w, 1e-3)
            h = max(h, 1e-3)
            # 若存在预置块或固定形状，覆盖其尺寸/位置
            if target_positions is not None:
                if target_positions[i, 2] != -1:   # 固定宽度
                    w = float(target_positions[i, 2])
                    h = float(target_positions[i, 3])
                if target_positions[i, 0] != -1:   # 预置位置
                    px = float(target_positions[i, 0])
                    py = float(target_positions[i, 1])
            positions.append((px, py, w, h))

        # 5. 后处理：去除重叠（简单迭代）
        positions = self._postprocess(positions, area_targets, constraints, target_positions)
     
        return positions
    
    def _postprocess(self, positions, area_targets, constraints, target_positions):
        """简单后处理：确保无重叠，调整面积误差"""
        block_count = len(positions)
        # 转换为列表方便修改
        pos = [list(p) for p in positions]
        
        # 多次迭代消除重叠
        for _ in range(50):
            has_overlap = False
            for i in range(block_count):
                x1, y1, w1, h1 = pos[i]
                for j in range(i+1, block_count):
                    x2, y2, w2, h2 = pos[j]
                    overlap_x = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                    overlap_y = max(0, min(y1+h1, y2+h2) - max(y1, y2))
                    if overlap_x > 1e-6 and overlap_y > 1e-6:
                        # 简单推动：将 j 向右上移动
                        pos[j][0] = x2 + overlap_x + 1.0
                        pos[j][1] = y2 + overlap_y + 1.0
                        has_overlap = True
                        break
                if has_overlap:
                    break
            if not has_overlap:
                break
        
        # 面积修正：软模块面积误差控制在1%以内
        for i in range(block_count):
            # 跳过固定形状和预置块（它们的尺寸不可变）
            if constraints is not None and i < len(constraints):
                if constraints[i, 0] != 0 or constraints[i, 1] != 0:
                    continue
            target_area = float(area_targets[i]) if i < len(area_targets) else 1.0
            w, h = pos[i][2], pos[i][3]
            actual_area = w * h
            if abs(actual_area - target_area) / target_area > 0.01:
                # 调整宽高，保持面积不变，限制宽高比
                scale = math.sqrt(target_area / actual_area)
                pos[i][2] = w * scale
                pos[i][3] = h * scale
        
        return [tuple(p) for p in pos]
    
    def _solve_sa_baseline(self, block_count, area_targets, b2b_conn,
                           p2b_conn, pins_pos, constraints, target_positions):
        """回退的模拟退火基线（与之前相同）"""
        # 初始化尺寸
        widths, heights = [], []
        for i in range(block_count):
            if target_positions is not None and target_positions[i, 2] != -1:
                w = float(target_positions[i, 2])
                h = float(target_positions[i, 3])
            else:
                area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
                w = h = math.sqrt(area)
            widths.append(w)
            heights.append(h)
        
        total_area = sum(w * h for w, h in zip(widths, heights))
        canvas_size = math.sqrt(total_area) * 1.5
        positions = []
        for i in range(block_count):
            if target_positions is not None and target_positions[i, 0] != -1:
                x = float(target_positions[i, 0])
                y = float(target_positions[i, 1])
            else:
                x = random.uniform(0, max(0, canvas_size - widths[i]))
                y = random.uniform(0, max(0, canvas_size - heights[i]))
            positions.append((x, y, widths[i], heights[i]))
        
        # 去除重叠
        for _ in range(100):
            overlaps = True
            for i in range(block_count):
                for j in range(i+1, block_count):
                    x1, y1, w1, h1 = positions[i]
                    x2, y2, w2, h2 = positions[j]
                    overlap_x = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                    overlap_y = max(0, min(y1+h1, y2+h2) - max(y1, y2))
                    if overlap_x > 1e-6 and overlap_y > 1e-6:
                        new_x = x2 + overlap_x + 1
                        new_y = y2 + overlap_y + 1
                        positions[j] = (new_x, new_y, w2, h2)
                        overlaps = False
                        break
                if not overlaps:
                    break
            if overlaps:
                break
    
        return positions