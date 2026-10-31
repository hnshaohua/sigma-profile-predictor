#!/usr/bin/env python
"""MPNN 模型定义。

本模块包含基于消息传递神经网络(MPNN)的 sigma-profile 预测模型：
    - EdgeConvLayer: 基于边特征的消息传递层，使用 Add 聚合方式
    - MPNN_V2: 完整的 MPNN 模型，包含节点编码器、多层消息传递、
      全局平均池化和全连接输出层

模型输出维度为 51（对应 51 个 sigma-profile 区间）。
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from .featurizer import bond_feat


class EdgeConvLayer(MessagePassing):
    """基于边特征的消息传递层。

    该层对每个节点，收集其邻居节点的特征，将邻居特征与边特征拼接后
    通过线性变换，最后使用 Add 方式聚合所有邻居的消息。

    在前向传播中，对聚合结果应用 BatchNorm + ReLU 激活，并加入残差连接。

    Args:
        in_channels: 输入节点特征维度
        out_channels: 输出节点特征维度
        dropout: Dropout 概率（默认 0.1）
    """

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels + bond_feat.dim, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        """前向传播。

        Args:
            x: 节点特征矩阵，形状 (n_nodes, in_channels)
            edge_index: 边索引矩阵，形状 (2, n_edges)
            edge_attr: 边特征矩阵，形状 (n_edges, bond_feat.dim)

        Returns:
            torch.Tensor: 输出节点特征矩阵，形状 (n_nodes, out_channels)
        """
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.bn(out)
        out = torch.relu(out)
        out = self.dropout(out)
        return out

    def message(self, x_j, edge_attr):
        """构建消息：将邻居节点特征与边特征拼接后通过线性层。

        Args:
            x_j: 邻居节点特征，形状 (n_edges, in_channels)
            edge_attr: 边特征，形状 (n_edges, bond_feat.dim)

        Returns:
            torch.Tensor: 消息向量，形状 (n_edges, out_channels)
        """
        return self.lin(torch.cat([x_j, edge_attr], dim=-1))


class MPNN_V2(nn.Module):
    """消息传递神经网络 V2。

    模型架构：
        1. 节点编码器：将原始原子特征映射到隐藏维度
        2. 多层 EdgeConvLayer：执行消息传递（默认 4 层）
        3. 全局平均池化：将节点级表示聚图为分子级表示
        4. 全连接层：将分子级表示映射到输出维度（51 维 sigma profile）

    全连接层结构：
        Linear(hidden_dim -> 512) -> BatchNorm -> ReLU -> Dropout
        -> Linear(512 -> 256) -> BatchNorm -> ReLU -> Dropout
        -> Linear(256 -> output_dim)
        -> Softplus（可选）

    Args:
        node_in_dim: 输入节点特征维度（等于 atom_feat.dim）
        edge_dim: 输入边特征维度（等于 bond_feat.dim）
        hidden_dim: 隐藏层维度（默认 256）
        num_layers: 消息传递层数（默认 4）
        output_dim: 输出维度，固定为 51（sigma profile 区间数）
        dropout: Dropout 概率（默认 0.2）
        use_softplus: 是否在输出层使用 Softplus 激活（默认 True）
    """

    def __init__(
        self,
        node_in_dim,
        hidden_dim=256,
        num_layers=4,
        output_dim=51,
        dropout=0.2,
        use_softplus=True,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [EdgeConvLayer(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.global_pool = global_mean_pool

        layers = [
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        ]
        if use_softplus:
            layers.append(nn.Softplus())

        self.fc = nn.Sequential(*layers)

    def forward(self, data):
        """前向传播。

        Args:
            data: torch_geometric.data.Data 对象，包含：
                - x: 节点特征，形状 (n_atoms, node_in_dim)
                - edge_index: 边索引，形状 (2, n_edges)
                - edge_attr: 边特征，形状 (n_edges, edge_dim)
                - batch: 节点到图的映射，形状 (n_atoms,)

        Returns:
            torch.Tensor: 预测的 sigma profile，形状 (n_molecules, output_dim)
        """
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        x = self.node_encoder(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr) + x
        x = self.global_pool(x, batch)
        return self.fc(x)
