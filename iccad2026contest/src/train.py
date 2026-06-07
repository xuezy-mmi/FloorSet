from torch_geometric.loader import DataLoader
from src.dataset import FloorSetLiteDataset

train_ds = FloorSetLiteDataset('/path/to/train', split='train')
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, 
                          follow_batch=['x', 'edge_index'])   # follow_batch 让 PyG 自动处理动态图

for batch in train_loader:
    # batch 是 PyG Batch 对象
    x = batch.x                     # [total_nodes, 53]
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr
    layout = batch.layout           # [batch_size * k, 4] 需要根据 batch 的 ptr 分割
    # 你的训练代码...