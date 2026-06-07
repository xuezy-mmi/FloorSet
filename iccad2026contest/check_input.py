#!/usr/bin/env python3
"""Inspect FloorSet LiteTensor datasets.

This script loads a specified sample from either the training set or the
validation/test set and prints the raw fields, shapes, dtypes, and a small
preview so you can see what is stored inside the dataset files.

The extraction logic follows the project's dataset definitions exactly:
- training samples follow `lite_dataset.py`
- validation/test samples follow `lite_dataset_test.py`

This script only adds human-readable annotations to explain what each field
means; it does not change the underlying extraction logic.

Examples:
    python check_input.py --mode train --root /home/liupeng22/ICCAD2026/FloorSet --worker-idx 3 --layout-idx 2
    python check_input.py --mode test --root /home/liupeng22/ICCAD2026/FloorSet --config-idx 21 --input-idx 1
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any
import glob
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lite_dataset_test import FloorplanDatasetLiteTest


def _describe_tensor(name: str, value: torch.Tensor, max_rows: int | None = None) -> None:
    print(f"{name}:")
    print(f"  type: {type(value).__name__}")
    print(f"  dtype: {value.dtype}")
    print(f"  shape: {tuple(value.shape)}")
    print(f"  numel: {value.numel()}")
    if value.numel() > 0:
        flat = value.flatten()
        preview = flat.tolist()
        print(f"  flat values ({len(preview)} total):")
        print(preview)
        if value.ndim >= 1:
            print("  full tensor:")
            with torch.no_grad():
                try:
                    torch.set_printoptions(profile="full", linewidth=200)
                    print(value)
                finally:
                    torch.set_printoptions(profile="default")
    print()


def _count_rows(value: torch.Tensor) -> int:
    return int(value.shape[0]) if value.ndim >= 1 else int(value.numel())


def _count_elements(value: torch.Tensor) -> int:
    return int(value.numel())


def _count_valid_rows(value: torch.Tensor) -> int:
    if value.ndim == 0:
        return int(value.numel() > 0)
    if value.numel() == 0:
        return 0
    if value.ndim == 1:
        return int((value != -1).sum().item())
    row_valid = (value != -1).any(dim=tuple(range(1, value.ndim)))
    return int(row_valid.sum().item())


def _count_nonzero_rows(value: torch.Tensor, col: int = 0) -> int:
    if value.ndim < 2 or value.shape[0] == 0 or value.shape[1] <= col:
        return 0
    return int((value[:, col] != 0).sum().item())


def _make_output_tag(file_path: Path, mode: str, layout_idx: int | None = None) -> str:
    if mode == "test":
        return f"Test__{file_path.parent.name}__{file_path.stem}"
    if mode == "train":
        base = f"train__{file_path.parent.name}__{file_path.name}"
        return f"{base}__layout_{layout_idx}" if layout_idx is not None else base
    return file_path.stem


def _select_layout_from_training_file(raw: Any, layout_idx: int):
    if not isinstance(raw, (list, tuple)) or len(raw) < 7:
        raise ValueError(f"Unexpected training file structure: {type(raw).__name__}")
    tensors = []
    layout_count = None
    for item in raw:
        if not isinstance(item, torch.Tensor):
            raise ValueError(f"Unexpected non-tensor entry in training file: {type(item).__name__}")
        if item.ndim == 0:
            raise ValueError("Unexpected scalar tensor in training file")
        layout_count = item.shape[0] if layout_count is None else min(layout_count, item.shape[0])
        tensors.append(item)
    if layout_count is None or layout_count == 0:
        raise ValueError("Training file contains no layouts")
    if layout_idx < 0 or layout_idx >= layout_count:
        raise IndexError(f"layout_idx {layout_idx} out of range for {layout_count} layouts in training file")
    return layout_idx, layout_count, tensors


def _load_training_file(root: str, worker_idx: int, file_idx: int):
    worker_dir = Path(root) / "LiteTensorData" / f"worker_{worker_idx}"
    file_path = worker_dir / f"layouts_{file_idx}.th"
    if not file_path.is_file():
        raise FileNotFoundError(f"Training layout file not found: {file_path}")
    raw = torch.load(file_path)
    total_files = len(list(worker_dir.glob("layouts_*.th")))
    return file_path, raw, total_files


def _build_training_sample_from_raw(raw: Any, layout_idx: int):
    layout_pos, layout_count, tensors = _select_layout_from_training_file(raw, layout_idx)
    area_target = tensors[0][layout_pos][:, 0]
    placement_constraints = tensors[0][layout_pos][:, 1:]
    b2b_connectivity = tensors[1][layout_pos]
    p2b_connectivity = tensors[2][layout_pos]
    pins_pos = tensors[3][layout_pos]
    tree_sol = tensors[4][layout_pos]
    fp_sol = tensors[5][layout_pos]
    metrics_sol = tensors[6][layout_pos]
    sample = {
        "input": (area_target, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints),
        "label": (tree_sol, fp_sol, metrics_sol),
    }
    return sample, layout_pos, layout_count


def _load_test_input_file(root: str, config_idx: int, layout_idx: int):
    config_dir = Path(root) / "LiteTensorDataTest" / f"config_{config_idx}"
    input_file = config_dir / f"litedata_{layout_idx}.pth"
    if not input_file.is_file():
        raise FileNotFoundError(f"Test input file not found: {input_file}")
    raw = torch.load(input_file)
    return input_file, raw, str(config_dir)


def print_training_sample(sample: dict[str, Any], summary_only: bool = False) -> None:
    inputs = sample["input"]
    labels = sample["label"]

    area_target, b2b_connectivity, p2b_connectivity, pins_pos, placement_constraints = inputs
    tree_sol, fp_sol, metrics_sol = labels

    print("=" * 80)
    print("INPUT FIELDS")
    print("=" * 80)
    print("【说明】下面打印的是训练集样本的输入要素，提取方式与 lite_dataset.py 保持一致。")
    print("【说明】这些字段会作为优化器输入，不是标签。")
    print()
    print("area_target  -> 每个 block 的目标面积 [输入特征]")
    _describe_tensor("area_target", area_target)
    print(f"  inferred block rows: {_count_valid_rows(area_target)}")
    print(f"  total elements: {_count_elements(area_target)}")
    print("b2b_connectivity -> block-to-block 网络连接 [输入特征]")
    _describe_tensor("b2b_connectivity", b2b_connectivity)
    print(f"  inferred edge rows: {_count_valid_rows(b2b_connectivity)}")
    print(f"  total elements: {_count_elements(b2b_connectivity)}")
    print("p2b_connectivity -> pin-to-block 网络连接 [输入特征]")
    _describe_tensor("p2b_connectivity", p2b_connectivity)
    print(f"  inferred edge rows: {_count_valid_rows(p2b_connectivity)}")
    print(f"  total elements: {_count_elements(p2b_connectivity)}")
    print("pins_pos -> pin 的二维坐标 [输入特征]")
    _describe_tensor("pins_pos", pins_pos)
    print(f"  inferred pin rows: {_count_valid_rows(pins_pos)}")
    print(f"  total elements: {_count_elements(pins_pos)}")
    print("placement_constraints -> 约束信息 [fixed, preplaced, mib, cluster, boundary] [输入特征]")
    _describe_tensor("placement_constraints", placement_constraints)
    print(f"  inferred block rows: {_count_valid_rows(placement_constraints)}")
    print(f"  total elements: {_count_elements(placement_constraints)}")

    print("=" * 80)
    print("LABEL / REFERENCE FIELDS")
    print("=" * 80)
    print("【说明】下面打印的是训练集的参考标签，不是优化器输入。")
    print("【说明】这些字段用于监督学习、评测或对照分析。")
    print()
    print("label_mode: train-style labels (tree_sol, fp_sol, metrics_sol)")
    print("tree_sol -> 参考树结构解 [标签]")
    _describe_tensor("tree_sol", tree_sol)
    print("fp_sol -> 参考 floorplan 矩形布局 [标签]")
    _describe_tensor("fp_sol", fp_sol)
    print("metrics_sol -> 参考统计指标 [标签]")
    _describe_tensor("metrics_sol", metrics_sol)

    if summary_only:
        return

    print("=" * 80)
    print("DERIVED COUNTS")
    print("=" * 80)
    block_count = _count_valid_rows(area_target)
    b2b_count = _count_valid_rows(b2b_connectivity)
    p2b_count = _count_valid_rows(p2b_connectivity)
    pin_count = _count_valid_rows(pins_pos)
    fixed_count = _count_nonzero_rows(placement_constraints, 0)
    preplaced_count = _count_nonzero_rows(placement_constraints, 1)
    mib_count = _count_nonzero_rows(placement_constraints, 2)
    cluster_count = _count_nonzero_rows(placement_constraints, 3)
    boundary_count = _count_nonzero_rows(placement_constraints, 4)

    print(f"block_count: {block_count}")
    print(f"b2b_edge_count: {b2b_count}")
    print(f"p2b_edge_count: {p2b_count}")
    print(f"pin_count: {pin_count}")
    print(f"fixed_count: {fixed_count}")
    print(f"preplaced_count: {preplaced_count}")
    print(f"mib_count: {mib_count}")
    print(f"cluster_count: {cluster_count}")
    print(f"boundary_count: {boundary_count}")
    print()

    print("=" * 80)
    print("FP SOL PREVIEW")
    print("=" * 80)
    if fp_sol.numel() > 0:
        print(fp_sol[: min(fp_sol.shape[0], 5)])
        print()

    print("=" * 80)
    print("METRICS PREVIEW")
    print("=" * 80)
    if metrics_sol.numel() > 0:
        print(metrics_sol)
        print()


def print_test_input_file(file_path: Path, raw: Any, summary_only: bool = False) -> None:
    print(f"Input file: {file_path}")
    print(f"Raw type: {type(raw).__name__}")
    try:
        print(f"Raw length: {len(raw)}")
    except Exception:
        pass
    print()

    payload = raw[0] if isinstance(raw, (list, tuple)) and len(raw) == 1 and isinstance(raw[0], (list, tuple)) else raw

    if isinstance(payload, (list, tuple)) and len(payload) >= 4:
        # Follow lite_dataset_test.py extraction logic exactly.
        block_data = payload[0][0] if isinstance(payload[0], (list, tuple)) else payload[0]
        area_target = block_data[:, 0]
        placement_constraints = block_data[:, 1:]
        b2b_connectivity = payload[1]
        p2b_connectivity = payload[2]
        pins_pos = payload[3]

        print("=" * 80)
        print("TEST INPUT FIELDS FROM LiteTensorDataTest input file")
        print("=" * 80)
        print("【说明】下面打印的是测试集输入文件中的要素，提取方式与 lite_dataset_test.py 保持一致。")
        print("【说明】这些字段会作为优化器输入，不包含标签文件中的工业最优解。")
        print()
        print("area_target -> 每个 block 的目标面积 [输入特征]")
        _describe_tensor("area_target", area_target, max_rows=None)
        print(f"  inferred block rows: {_count_valid_rows(area_target)}")
        print(f"  total elements: {_count_elements(area_target)}")
        print("b2b_connectivity -> block-to-block 网络连接 [输入特征]")
        _describe_tensor("b2b_connectivity", b2b_connectivity, max_rows=None)
        print(f"  inferred edge rows: {_count_valid_rows(b2b_connectivity)}")
        print(f"  total elements: {_count_elements(b2b_connectivity)}")
        print("p2b_connectivity -> pin-to-block 网络连接 [输入特征]")
        _describe_tensor("p2b_connectivity", p2b_connectivity, max_rows=None)
        print(f"  inferred edge rows: {_count_valid_rows(p2b_connectivity)}")
        print(f"  total elements: {_count_elements(p2b_connectivity)}")
        print("pins_pos -> pin 的二维坐标 [输入特征]")
        _describe_tensor("pins_pos", pins_pos, max_rows=None)
        print(f"  inferred pin rows: {_count_valid_rows(pins_pos)}")
        print(f"  total elements: {_count_elements(pins_pos)}")
        print("placement_constraints -> 约束信息 [fixed, preplaced, mib, cluster, boundary] [输入特征]")
        _describe_tensor("placement_constraints", placement_constraints, max_rows=None)
        print(f"  inferred block rows: {_count_valid_rows(placement_constraints)}")
        print(f"  total elements: {_count_elements(placement_constraints)}")

        if summary_only:
            return

        print("=" * 80)
        print("DERIVED COUNTS")
        print("=" * 80)
        block_count = _count_valid_rows(area_target)
        b2b_count = _count_valid_rows(b2b_connectivity)
        p2b_count = _count_valid_rows(p2b_connectivity)
        pin_count = _count_valid_rows(pins_pos)
        fixed_count = _count_nonzero_rows(placement_constraints, 0)
        preplaced_count = _count_nonzero_rows(placement_constraints, 1)
        mib_count = _count_nonzero_rows(placement_constraints, 2)
        cluster_count = _count_nonzero_rows(placement_constraints, 3)
        boundary_count = _count_nonzero_rows(placement_constraints, 4)

        print(f"block_count: {block_count}")
        print(f"b2b_edge_count: {b2b_count}")
        print(f"p2b_edge_count: {p2b_count}")
        print(f"pin_count: {pin_count}")
        print(f"fixed_count: {fixed_count}")
        print(f"preplaced_count: {preplaced_count}")
        print(f"mib_count: {mib_count}")
        print(f"cluster_count: {cluster_count}")
        print(f"boundary_count: {boundary_count}")
        print()
    else:
        print("Unexpected raw structure; unable to parse as test input file.")
        print(raw)


def build_training_dataset(root: str):
    return FloorplanDatasetLite(root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect FloorSet LiteTensor samples")
    parser.add_argument("--mode", choices=["train", "test"], default="test",
                        help="Choose training set or validation/test input files")
    parser.add_argument("--root", required=True,
                        help="Dataset root directory, e.g. /home/liupeng22/ICCAD2026/FloorSet")
    parser.add_argument("--worker-idx", type=int, default=None,
                        help="For train mode, select the worker directory index")
    parser.add_argument("--file-idx", type=int, default=None,
                        help="For train mode, select the layouts_<idx>.th file index")
    parser.add_argument("--layout-idx", type=int, default=None,
                        help="For train mode, select the layout index inside the chosen file")
    parser.add_argument("--config-idx", type=int, default=None,
                        help="For test mode, select LiteTensorDataTest/config_<idx>")
    parser.add_argument("--input-idx", type=int, default=None,
                        help="For test mode, select litedata_<idx>.pth inside the chosen config directory")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print only tensor shapes and counts, not full previews")
    parser.add_argument("--output-dir", default="/home/liupeng22/ICCAD2026/FloorSet/my_create/input_result",
                        help="Directory to save the printed inspection result")
    parser.add_argument("--output-ext", default="md", choices=["md", "txt"],
                        help="Output file extension")
    args = parser.parse_args()

    buffer = StringIO()
    with redirect_stdout(buffer):
        if args.mode == "train":
            if args.worker_idx is None or args.file_idx is None or args.layout_idx is None:
                raise ValueError("train mode requires --worker-idx, --file-idx and --layout-idx")
            file_path, raw, total_files = _load_training_file(args.root, args.worker_idx, args.file_idx)
            sample, layout_pos, layout_count = _build_training_sample_from_raw(raw, args.layout_idx)
            print("Dataset mode: train-worker-file")
            print(f"Worker idx: {args.worker_idx}")
            print(f"Layout file idx (file name): {args.file_idx}")
            print(f"Layout idx used inside file: {layout_pos}")
            print(f"Layouts in file: {layout_count}")
            print(f"Available layout files in worker: {total_files}")
            print()
            print_training_sample(sample, summary_only=args.summary_only)
        else:
            if args.config_idx is None or args.input_idx is None:
                raise ValueError("test mode requires --config-idx and --input-idx")
            file_path, raw, config_dir = _load_test_input_file(args.root, args.config_idx, args.input_idx)
            print("Dataset mode: test-input-file")
            print(f"Config dir: {config_dir}")
            print(f"Config idx: {args.config_idx}")
            print(f"Input file idx: {args.input_idx}")
            print()
            print_test_input_file(file_path, raw, summary_only=args.summary_only)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_tag = _make_output_tag(file_path, args.mode, args.layout_idx if args.mode == "train" else None)
    output_file = output_dir / f"{source_tag}.{args.output_ext}"
    output_text = buffer.getvalue()
    output_file.write_text(output_text, encoding="utf-8")
    print(output_text, end="")
    print(f"[saved to] {output_file}")


if __name__ == "__main__":
    main()

