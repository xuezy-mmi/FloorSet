import torch
import os
import json
from pathlib import Path
from datetime import datetime

def parse_th_file(file_path):
    """
    加载 .th 文件并提取关键信息，返回一个字典。
    根据 FloorSet-Lite 的格式，文件内容通常是一个 dict 或 tuple。
    实际加载后需要查看结构，这里假设 load 后是包含以下字段的 dict：
        - area_target: Tensor [n_blocks]
        - b2b_connectivity: Tensor [n_edges, 3]
        - p2b_connectivity: Tensor [n_edges, 3]
        - pins_pos: Tensor [n_pins, 2]
        - constraints: Tensor [n_blocks, 5]
        - sol: Tensor [n_blocks, 4]   (w, h, x, y)
        - metrics: Tensor [8]  (area, num_pins, ... , b2b_weighted_wl, p2b_weighted_wl)
    """
    data = torch.load(file_path, map_location='cpu')
    
    # 兼容多种可能的存储格式：如果 data 是 tuple，通常顺序与 dataloader 返回一致
    # 常见顺序: (area, b2b, p2b, pins, constraints, tree_sol, fp_sol, metrics)
    if isinstance(data, tuple):
        # 根据长度猜测
        if len(data) == 8:
            area_target, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, sol, metrics = data
        elif len(data) == 7:  # 没有 tree_sol
            area_target, b2b_conn, p2b_conn, pins_pos, constraints, sol, metrics = data
        else:
            raise ValueError(f"Unexpected tuple length: {len(data)}")
    elif isinstance(data, dict):
        area_target = data.get('area_target')
        b2b_conn = data.get('b2b_connectivity')
        p2b_conn = data.get('p2b_connectivity')
        pins_pos = data.get('pins_pos')
        constraints = data.get('placement_constraints')
        sol = data.get('sol') or data.get('fp_sol')
        metrics = data.get('metrics')
    else:
        raise TypeError(f"Unknown data type: {type(data)}")
    
    # 转换为 numpy 便于打印
    n_blocks = area_target.shape[0] if area_target is not None else 0
    info = {
        'file': str(file_path),
        'n_blocks': int(n_blocks),
        'has_sol': sol is not None,
        'sol_shape': list(sol.shape) if sol is not None else None,
        'area_target_min': float(area_target.min()) if area_target is not None else None,
        'area_target_max': float(area_target.max()) if area_target is not None else None,
        'num_b2b_edges': b2b_conn.shape[0] if b2b_conn is not None else 0,
        'num_p2b_edges': p2b_conn.shape[0] if p2b_conn is not None else 0,
        'num_pins': pins_pos.shape[0] if pins_pos is not None else 0,
        'constraints_summary': {},
        'metrics': metrics.tolist() if metrics is not None else None,
    }
    
    # 解析约束类型统计
    if constraints is not None:
        # constraints: [n_blocks, 5] 列为 [fixed, preplaced, mib_group, cluster_group, boundary_mask]
        fixed_cnt = (constraints[:, 0] == 1).sum().item()
        preplaced_cnt = (constraints[:, 1] == 1).sum().item()
        mib_groups = len(set(constraints[:, 2].tolist())) - (1 if -1 in constraints[:, 2] else 0)
        cluster_groups = len(set(constraints[:, 3].tolist())) - (1 if -1 in constraints[:, 3] else 0)
        boundary_cnt = (constraints[:, 4] != 0).sum().item()
        info['constraints_summary'] = {
            'fixed': fixed_cnt,
            'preplaced': preplaced_cnt,
            'mib_groups': mib_groups,
            'cluster_groups': cluster_groups,
            'boundary': boundary_cnt,
        }
    
    # 可选：取前几个模块的 sol 示例
    if sol is not None and n_blocks > 0:
        sample_blocks = min(3, n_blocks)
        info['sol_samples'] = sol[:sample_blocks].tolist()
    
    return info

def generate_md_report(root_dir, output_md="th_files_report.md"):
    """
    遍历 root_dir 下所有子目录中的 .th 文件，解析并汇总成 markdown 表格。
    """
    root = Path(root_dir)
    all_th_files = list(root.rglob("*.th"))
    if not all_th_files:
        print(f"未找到任何 .th 文件，请检查路径: {root_dir}")
        return
    
    results = []
    for f in sorted(all_th_files):
        try:
            info = parse_th_file(f)
            results.append(info)
        except Exception as e:
            print(f"解析失败 {f}: {e}")
            results.append({'file': str(f), 'error': str(e)})
    
    # 写入 markdown
    with open(output_md, 'w', encoding='utf-8') as md:
        md.write(f"# FloorSet-Lite 数据文件解析报告\n\n")
        md.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        md.write(f"扫描根目录：`{root_dir}`\n")
        md.write(f"共找到 **{len(all_th_files)}** 个 .th 文件\n\n")
        
        # 汇总统计
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            block_counts = [r['n_blocks'] for r in valid_results]
            md.write("## 整体统计\n")
            md.write(f"- 模块数范围：{min(block_counts)} ~ {max(block_counts)}\n")
            md.write(f"- 平均模块数：{sum(block_counts)/len(block_counts):.1f}\n\n")
        
        # 表格
        md.write("## 文件详情\n")
        md.write("| 文件路径 | 模块数 | 固定 | 预置 | MIB组 | 分组组 | 边界约束 | b2b边数 | p2b边数 | 引脚数 | 有解 |\n")
        md.write("|----------|--------|------|------|-------|--------|----------|---------|---------|--------|------|\n")
        for r in results:
            if 'error' in r:
                md.write(f"| {r['file']} | error | - | - | - | - | - | - | - | - | ❌ |\n")
            else:
                c = r['constraints_summary']
                md.write(f"| {r['file']} | {r['n_blocks']} | {c.get('fixed',0)} | {c.get('preplaced',0)} | "
                         f"{c.get('mib_groups',0)} | {c.get('cluster_groups',0)} | {c.get('boundary',0)} | "
                         f"{r['num_b2b_edges']} | {r['num_p2b_edges']} | {r['num_pins']} | {'✅' if r['has_sol'] else '❌'} |\n")
        
        # 附加示例数据（前三个文件的 sol 示例）
        md.write("\n## 布局解示例 (前三个文件，每个文件前3个模块)\n")
        for r in results[:3]:
            if 'error' not in r and r.get('sol_samples'):
                md.write(f"\n### {Path(r['file']).name}\n")
                md.write("| 模块索引 | 宽度(w) | 高度(h) | 左下角x | 左下角y |\n")
                md.write("|----------|--------|--------|---------|---------|\n")
                for i, (w, h, x, y) in enumerate(r['sol_samples']):
                    md.write(f"| {i} | {w:.3f} | {h:.3f} | {x:.3f} | {y:.3f} |\n")
    
    print(f"报告已生成：{output_md}")

if __name__ == "__main__":
    # 请根据您的实际路径修改
    DATA_ROOT = "/home/xzy/eda/floorset_lite"
    generate_md_report(DATA_ROOT, "floorset_lite_report.md")