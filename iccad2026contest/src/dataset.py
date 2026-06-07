# src/dataset.py
import torch
import os
import glob
from torch.utils.data import Dataset
# from torch_geometric.data import Data
import numpy as np

class FloorSetLiteDataset(Dataset):
    """
    FloorSet-Lite 数据集加载器，适配竞赛提供的 .pth 文件格式。
    支持训练集（包含最优布局标签）和测试集（无标签）。
    输出 PyTorch Geometric 的 Data 对象，便于直接使用 GNN。
    """
    def __init__(self, data_dir, split='train', max_blocks=120, normalize_pin_coords=True):
        """
        Args:
            data_dir (str): 数据集根目录，例如 '/path/to/FloorSet/LiteTensorDataTrain'
            split (str): 'train' 或 'test' 或 'val'
            max_blocks (int): 最大模块数，用于静态图构建（实际未使用，仅占位）
            normalize_pin_coords (bool): 是否将引脚坐标归一化到 [0,1]
        """
        self.data_dir = data_dir
        self.split = split
        self.normalize_pin_coords = normalize_pin_coords
        # 收集所有 .pth 文件路径
        self.file_list = []
        # 目录结构可能是：data_dir/config_21/*.pth 或 data_dir/*.pth
        # 尝试两种模式
        config_dirs = glob.glob(os.path.join(data_dir, 'config_*'))
        if config_dirs:
            for config_dir in sorted(config_dirs):
                for f in glob.glob(os.path.join(config_dir, '*.pth')):
                    self.file_list.append(f)
        else:
            self.file_list = glob.glob(os.path.join(data_dir, '*.pth'))
        self.file_list.sort()
        print(f"[{split}] Found {len(self.file_list)} .pth files in {data_dir}")

        # 统计全局坐标范围（用于归一化）
        self.canvas_size = None
        if normalize_pin_coords:
            self._estimate_canvas_size()

    def _estimate_canvas_size(self):
        """估算画布大小，从所有文件中的 pins_pos 最大值获取"""
        max_coord = 0.0
        # 仅采样前 100 个文件以加速（可修改）
        for fpath in self.file_list[:min(100, len(self.file_list))]:
            data = torch.load(fpath)
            if isinstance(data, list):
                data = data[0]
            pins = data['pins_pos']
            if pins.numel() > 0:
                max_coord = max(max_coord, pins.max().item())
        self.canvas_size = max_coord if max_coord > 0 else 124.0
        print(f"Estimated canvas size (max coordinate): {self.canvas_size}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        raw = torch.load(filepath)
        # 数据存储为 list，第一个元素是字典
        if isinstance(raw, list):
            sample = raw[0]
        else:
            sample = raw

        # ---------- 1. 提取原始字段 ----------
        area_target = sample['area_target']                     # [k]
        b2b = sample['b2b_connectivity']                        # [E_bb, 3]
        p2b = sample['p2b_connectivity']                        # [E_pb, 3]
        pins_pos = sample['pins_pos']                           # [num_pins, 2]
        constraints = sample['placement_constraints']           # [k, 5]

        k = area_target.shape[0]
        num_pins = pins_pos.shape[0]

        # ---------- 2. 节点特征 ----------
        # 2.1 Block 节点特征
        # 特征列依次为：log_area, is_fixed, is_preplaced, boundary_onehot(8), mib_onehot(21), cluster_onehot(21)
        log_area = torch.log(area_target + 1e-6).unsqueeze(-1)   # [k, 1]
        is_fixed = constraints[:, 0].unsqueeze(-1)               # [k, 1]
        is_preplaced = constraints[:, 1].unsqueeze(-1)           # [k, 1]

        # Boundary constraint (col4) -> one-hot，最多8类（可根据竞赛文档修改类别数）
        # TODO: 确认 boundary code 的取值范围，目前假设 0~7
        boundary_code = constraints[:, 4].long()
        num_boundary_classes = 8   # 根据实际调整
        boundary_onehot = torch.nn.functional.one_hot(boundary_code, num_classes=num_boundary_classes).float()

        # MIB group id (col2) -> one-hot
        # TODO: 确认最大 group id，根据训练数据统计，暂设 21
        mib_id = constraints[:, 2].long()
        num_mib_classes = 21
        mib_onehot = torch.nn.functional.one_hot(mib_id, num_classes=num_mib_classes).float()

        # Cluster (grouping) group id (col3) -> one-hot
        cluster_id = constraints[:, 3].long()
        num_cluster_classes = 21
        cluster_onehot = torch.nn.functional.one_hot(cluster_id, num_classes=num_cluster_classes).float()

        block_feat = torch.cat([log_area, is_fixed, is_preplaced,
                                boundary_onehot, mib_onehot, cluster_onehot], dim=1)
        # block_feat 维度 = 1+1+1 + 8 + 21 + 21 = 53

        # 2.2 Pin 节点特征：归一化坐标 (x, y)
        if self.normalize_pin_coords and self.canvas_size is not None:
            pin_feat = pins_pos / self.canvas_size   # [num_pins, 2]
        else:
            pin_feat = pins_pos

        # 合并所有节点：先 block 后 pin
        x = torch.cat([block_feat, pin_feat], dim=0)   # [total_nodes, feat_dim]
        # 记录每个节点所属类型，便于 GNN 中分别处理（若使用异构图，需要分别存储）
        node_type = torch.cat([torch.zeros(k, dtype=torch.long),
                               torch.ones(num_pins, dtype=torch.long)], dim=0)   # 0:block, 1:pin

        # ---------- 3. 边构建 ----------
        # 3.1 Block-Block 边（无向，双向）
        src_bb = b2b[:, 0].long()
        dst_bb = b2b[:, 1].long()
        weight_bb = b2b[:, 2].unsqueeze(-1)   # [E_bb, 1]
        # 双向边
        edge_index_bb = torch.stack([torch.cat([src_bb, dst_bb]),
                                     torch.cat([dst_bb, src_bb])], dim=0)
        edge_attr_bb = torch.cat([weight_bb, weight_bb], dim=0)

        # 3.2 Block-Pin 边（无向，双向）
        # p2b 格式: [pin_id, block_id, weight]
        pin_ids = p2b[:, 0].long()
        block_ids = p2b[:, 1].long()
        weight_pb = p2b[:, 2].unsqueeze(-1)   # [E_pb, 1]
        # 全局节点索引: pin 节点在 x 中的索引 = k + pin_id
        global_pin_idx = pin_ids + k
        global_block_idx = block_ids
        edge_index_pb = torch.stack([torch.cat([global_block_idx, global_pin_idx]),
                                     torch.cat([global_pin_idx, global_block_idx])], dim=0)
        edge_attr_pb = torch.cat([weight_pb, weight_pb], dim=0)

        # 合并所有边
        edge_index = torch.cat([edge_index_bb, edge_index_pb], dim=1)   # [2, total_edges]
        edge_attr = torch.cat([edge_attr_bb, edge_attr_pb], dim=0)      # [total_edges, 1]

        # ---------- 4. 目标布局（仅训练/验证集）----------
        layout = None
        fixed_shapes = None
        preplaced_info = None
        if 'layout' in sample:
            layout = sample['layout']   # [k, 4]
        # 某些文件可能还包含固定形状尺寸和预放置信息（测试集中一般没有）
        # TODO: 根据实际字段名称调整
        if 'fixed_shapes' in sample:
            fixed_shapes = sample['fixed_shapes']
        if 'preplaced' in sample:
            preplaced_info = sample['preplaced']

        # 可选：存储其他元信息
        meta = {
            'num_blocks': k,
            'num_pins': num_pins,
            'area_target': area_target.clone(),
            'constraints': constraints.clone(),
            'filepath': filepath,
        }

        # 返回 PyG Data 对象，方便直接使用 torch_geometric.loader.DataLoader
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    node_type=node_type, layout=layout, meta=meta,
                    fixed_shapes=fixed_shapes, preplaced_info=preplaced_info)
        return data

    @staticmethod
    def collate_fn(batch):
        """PyG 的 DataLoader 已内置批处理，此方法无需实现，仅保留接口"""
        from torch_geometric.loader import Batch
        return Batch.from_data_list(batch)



# if __name__ == "__main__":

    
#     train_dir = "/home/xzy/eda/FloorSet/LiteTensorDatatest"
#     test_dir = "/home/xzy/eda/FloorSet/LiteTensorDataTest"

#     train_dataset = FloorSetLiteDataset(train_dir, split='train')
#     print(f"Train dataset size: {len(train_dataset)}")
#     sample = train_dataset[0]
#     print(f"Sample keys: {sample.keys}")
#     print(f"x shape: {sample.x.shape}, edge_index shape: {sample.edge_index.shape}")
#     if sample.layout is not None:
#         print(f"Layout shape: {sample.layout.shape}")
#     else:
#         print("No layout in this sample (should not happen for training set)")

#     # test with no layout (label)
#     test_dataset = FloorSetLiteDataset(test_dir, split='test')
#     print(f"Test dataset size: {len(test_dataset)}")
#     sample_test = test_dataset[0]
#     print(f"Test sample has layout: {sample_test.layout is not None}")
    


def parse_pth_file(data_path, label_path=None):
    """
    Args:
        data_path (str): litedata_*.
        label_path (str, optional): litelabel_*.pth
        
    Returns:
        dict: 'data' and 'label' dictionary. If no label_path, 'label' dictionary is None
    """

    data_raw = torch.load(data_path)

    if isinstance(data_raw, list):
        data_dict = data_raw[0]
    else:
        data_dict = data_raw
    
    result = {'data': data_dict}
    
    # 加载标签文件（如果提供或自动推断）
    if label_path is None:
        # 自动推断：将 data_path 中的 'litedata' 替换为 'litelabel'
        label_path = data_path.replace('litedata', 'litelabel')
    
    if os.path.exists(label_path):
        label_raw = torch.load(label_path)
        if isinstance(label_raw, list):
            label_dict = label_raw[0]
        else:
            label_dict = label_raw
        result['label'] = label_dict
    else:
        result['label'] = None
    
    return result


def load_config_data(config_dir):
    """
    加载一个 config_xx 目录下的所有数据-标签对（通常只有一个 .pth 文件）
    
    Args:
        config_dir (str): 例如 '/home/xzy/eda/FloorSet/LiteTensorDatatest/config_100'
        
    Returns:
        list of dict: 每个元素是 parse_pth_file 的返回结果
    """
    # 查找所有 litedata_*.pth 文件
    data_files = glob.glob(os.path.join(config_dir, 'litedata_*.pth'))
    results = []
    for df in data_files:
        # 自动寻找对应的 litelabel_*.pth
        label_file = df.replace('litedata', 'litelabel')
        if not os.path.exists(label_file):
            label_file = None
        results.append(parse_pth_file(df, label_file))
    return results


# ---------- 使用示例（测试）----------
if __name__ == '__main__':
    # 测试单个文件解析
    test_data_path = '/home/xzy/eda/FloorSet/LiteTensorDataTest/config_100/litedata_1.pth'
    parsed = parse_pth_file(test_data_path)
    # print('Data keys:', parsed['data'].keys())
    # if parsed['label'] is not None:
    #     print('Label keys:', parsed['label'].keys())
    # else:
    #     print('No label found')
    
    # # 测试整个 config 目录
    # config_dir = '/home/xzy/eda/FloorSet/LiteTensorDataTest/config_100'
    # all_pairs = load_config_data(config_dir)
    # print(f'Found {len(all_pairs)} data-label pairs')