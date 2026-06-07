# src/graph_encoder.py
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GINEConv, Linear

class HeteroBlockPinGNN(nn.Module):
    def __init__(self, block_in_dim, pin_in_dim, hidden_dim=256, out_dim=256, num_layers=4):
        super().__init__()
        self.block_lin = Linear(block_in_dim, hidden_dim)
        self.pin_lin = Linear(pin_in_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('block', 'connects', 'block'): GINEConv(nn.Linear(hidden_dim, hidden_dim), edge_dim=1),
                ('block', 'connects_to', 'pin'): GINEConv(nn.Linear(hidden_dim, hidden_dim), edge_dim=1),
                ('pin', 'connected_by', 'block'): GINEConv(nn.Linear(hidden_dim, hidden_dim), edge_dim=1),
            }, aggr='mean')
            self.convs.append(conv)
        
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # x_dict: {'block': (B*k, block_in_dim), 'pin': (B*num_pins, pin_in_dim)}
        # 首先投影到统一维度
        x_dict = {
            'block': self.block_lin(x_dict['block']),
            'pin': self.pin_lin(x_dict['pin'])
        }
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
        
        # 只返回 block 节点的嵌入
        return self.out_mlp(x_dict['block'])   # [total_blocks, out_dim]