#!/usr/bin/env python
"""
V2-ABS 训练脚本（面积增强版，支持自定义随机种子）
核心改进：训练时损失在绝对面积上计算（用真实总面积反归一化）
推理时：预测相对面积 × 预测总面积 = 绝对面积
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from rdkit import Chem
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


# ======================================================================
# 特征编码器
# ======================================================================
class AtomFeaturizer:
    def __init__(self):
        self.symbol_set = {"As", "B", "Br", "C", "Ca", "Cl", "F", "H", "I", "K", "Mg", "N", "Na", "O", "P", "S", "Se",
                           "Si"}
        self.valence_set = {0, 1, 2, 3, 4, 5, 6, 7, 8}
        self.hydrogen_set = {0, 1, 2, 3, 4}
        self.hybridization_set = {"s", "sp", "sp2", "sp3"}
        self.degree_set = {0, 1, 2, 3, 4, 5, 6}
        self.charge_set = {-1, 0, 1}
        self.ring_set = {0, 1}
        self.chiral_set = {0, 1}
        self.dim = 0
        self.feature_maps = {}
        for name, s in [("symbol", self.symbol_set), ("n_valence", self.valence_set),
                        ("n_hydrogens", self.hydrogen_set), ("hybridization", self.hybridization_set),
                        ("degree", self.degree_set), ("charge", self.charge_set),
                        ("in_ring", self.ring_set), ("chiral", self.chiral_set)]:
            s = sorted(list(s))
            self.feature_maps[name] = {v: i for i, v in enumerate(s)}
            self.dim += len(s)

    def encode(self, atom):
        features = [0] * self.dim
        sym = atom.GetSymbol()
        if sym in self.feature_maps["symbol"]:
            features[self.feature_maps["symbol"][sym]] = 1.0
        offset = len(self.feature_maps["symbol"])
        valence = atom.GetTotalValence()
        if valence in self.feature_maps["n_valence"]:
            features[offset + self.feature_maps["n_valence"][valence]] = 1.0
        offset += len(self.feature_maps["n_valence"])
        hs = atom.GetTotalNumHs()
        if hs in self.feature_maps["n_hydrogens"]:
            features[offset + self.feature_maps["n_hydrogens"][hs]] = 1.0
        offset += len(self.feature_maps["n_hydrogens"])
        hyb = atom.GetHybridization().name.lower()
        if hyb in self.feature_maps["hybridization"]:
            features[offset + self.feature_maps["hybridization"][hyb]] = 1.0
        offset += len(self.feature_maps["hybridization"])
        degree = atom.GetDegree()
        if degree in self.feature_maps["degree"]:
            features[offset + self.feature_maps["degree"][degree]] = 1.0
        offset += len(self.feature_maps["degree"])
        charge = atom.GetFormalCharge()
        if charge in self.feature_maps["charge"]:
            features[offset + self.feature_maps["charge"][charge]] = 1.0
        offset += len(self.feature_maps["charge"])
        in_ring = 1 if atom.IsInRing() else 0
        features[offset + self.feature_maps["in_ring"][in_ring]] = 1.0
        offset += len(self.feature_maps["in_ring"])
        chiral = 1 if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0
        features[offset + self.feature_maps["chiral"][chiral]] = 1.0
        return np.array(features, dtype=np.float32)


class BondFeaturizer:
    def __init__(self):
        self.bond_type_set = ["single", "double", "triple", "aromatic"]
        self.conj_set = {True, False}
        self.ring_set = {True, False}
        self.dim = len(self.bond_type_set) + len(self.conj_set) + len(self.ring_set)

    def encode(self, bond):
        features = [0] * self.dim
        offset = 0
        bt = bond.GetBondType().name.lower()
        if bt in self.bond_type_set:
            features[offset + list(self.bond_type_set).index(bt)] = 1.0
        offset += len(self.bond_type_set)
        conj = bond.GetIsConjugated()
        features[offset + (0 if conj else 1)] = 1.0
        offset += len(self.conj_set)
        in_ring = bond.IsInRing()
        features[offset + (0 if in_ring else 1)] = 1.0
        return np.array(features, dtype=np.float32)


atom_feat = AtomFeaturizer()
bond_feat = BondFeaturizer()


# ======================================================================
# 分子图构建
# ======================================================================
def molecule_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    node_feat_list = [atom_feat.encode(atom) for atom in mol.GetAtoms()]
    node_features = np.array(node_feat_list, dtype=np.float32)
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = []
    edge_attr_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
        feats = bond_feat.encode(bond)
        edge_attr_list.append(feats)
        edge_attr_list.append(feats)
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = np.array(edge_attr_list, dtype=np.float32)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, bond_feat.dim), dtype=torch.float)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


# ======================================================================
# GNN 模型（V2结构）
# ======================================================================
class EdgeConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels + bond_feat.dim, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out

    def message(self, x_j, edge_attr):
        return self.lin(torch.cat([x_j, edge_attr], dim=-1))


class MPNN_V2(nn.Module):
    def __init__(self, node_in_dim, edge_dim, hidden_dim=256, num_layers=4, output_dim=51, dropout=0.2,
                 use_softplus=True):
        super().__init__()
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.convs = nn.ModuleList([EdgeConvLayer(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)])
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
            nn.Linear(256, output_dim)
        ]
        if use_softplus:
            layers.append(nn.Softplus())
        self.fc = nn.Sequential(*layers)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr) + x
        x = self.global_pool(x, batch)
        return self.fc(x)


# ======================================================================
# 损失函数
# ======================================================================
def mse_loss_absolute(pred_profile, true_profile, true_area):
    pred_abs = pred_profile * true_area
    true_abs = true_profile * true_area
    return F.mse_loss(pred_abs, true_abs)


def composite_loss_absolute(pred_profile, true_profile, true_area, eps_rel=0.01):
    pred_abs = pred_profile * true_area
    true_abs = true_profile * true_area

    pred_abs = torch.clamp(pred_abs, min=1e-6)
    true_abs = torch.clamp(true_abs, min=1e-6)

    abs_err = torch.abs(pred_abs - true_abs)
    safe_denom = torch.where(true_abs > eps_rel, true_abs, torch.full_like(true_abs, eps_rel))
    rel_abs = abs_err / safe_denom

    log_pred = torch.log(pred_abs + 0.1)
    log_true = torch.log(true_abs + 0.1)
    log_sq_err = (log_true - log_pred) ** 2

    sample_loss = torch.mean(rel_abs + log_sq_err, dim=1)
    return torch.mean(sample_loss)


# ======================================================================
# 数据加载
# ======================================================================
def parse_sigma_profile(s):
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    return np.array([float(x) for x in s.split(',')], dtype=np.float32)


def load_data(csv_path="sigmaprofile.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file {csv_path} not found.")
    df = pd.read_csv(csv_path)
    graph_data = []
    total_areas = []

    for i, row in df.iterrows():
        g = molecule_to_graph(row['SMILES'])
        if g is not None:
            profile = parse_sigma_profile(row['SigmaProfile'])
            if (profile < 0).any():
                profile = np.clip(profile, 0, None)
            total_area = np.sum(profile)
            if total_area > 0:
                profile_norm = profile / total_area
            else:
                profile_norm = profile
            g.y = torch.tensor(profile_norm, dtype=torch.float)
            g.total_area = torch.tensor(total_area, dtype=torch.float)
            graph_data.append(g)
            total_areas.append(total_area)
        else:
            print(f"Warning: invalid SMILES at index {i}")

    print(f"Total valid graphs: {len(graph_data)}")
    print(f"Total area range: [{np.min(total_areas):.2f}, {np.max(total_areas):.2f}]")
    return graph_data, df


# ======================================================================
# 训练与评估
# ======================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        batch_size = data.num_graphs
        optimizer.zero_grad()
        pred_profile = model(data)
        true_profile = data.y.view(batch_size, 51)
        true_area = data.total_area.view(batch_size, 1)
        loss = criterion(pred_profile, true_profile, true_area)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch_size
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            batch_size = data.num_graphs
            pred_profile = model(data)
            true_profile = data.y.view(batch_size, 51)
            true_area = data.total_area.view(batch_size, 1)
            loss = criterion(pred_profile, true_profile, true_area)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            total_loss += loss.item() * batch_size
    return total_loss / len(loader.dataset)


# ======================================================================
# 早停类
# ======================================================================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


# ======================================================================
# 主函数
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Train V2 with absolute area loss (multi-seed)")
    parser.add_argument('--loss', type=str, choices=['mse', 'composite'], default='mse',
                        help='Loss function: mse or composite (default: mse)')
    parser.add_argument('--epochs', type=int, default=300, help='Maximum epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--data', type=str, default='sigmaprofile.csv', help='CSV data file')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides default)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for data split')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loss function: {args.loss}")
    print(f"Random Seed: {args.seed}")
    print(f"Learning rate: {args.lr if args.lr else 'Default'}")
    print("关键改进: 损失在绝对面积上计算")

    # 加载数据
    graph_data, df = load_data(args.data)

    # 使用传入的 seed 进行划分
    train_idx, temp_idx = train_test_split(range(len(graph_data)), test_size=0.1, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=args.seed)

    train_loader = DataLoader([graph_data[i] for i in train_idx], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader([graph_data[i] for i in val_idx], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader([graph_data[i] for i in test_idx], batch_size=args.batch_size, shuffle=False)

    print(f"Test data length: {len(test_loader.dataset)}")

    # 模型
    model = MPNN_V2(
        node_in_dim=atom_feat.dim,
        edge_dim=bond_feat.dim,
        hidden_dim=256,
        num_layers=4,
        output_dim=51,
        dropout=0.2,
        use_softplus=True
    ).to(device)

    # 损失与学习率
    if args.loss == 'mse':
        criterion = mse_loss_absolute
        default_lr = 1e-3
        suffix = 'mse'
    else:
        criterion = composite_loss_absolute
        default_lr = 1e-4
        suffix = 'composite'

    lr = args.lr if args.lr is not None else default_lr

    # 动态生成文件名，带 seed 后缀
    model_path = f"best_model_v2_abs_{suffix}_seed{args.seed}.pt"
    log_file = f"loss_log_v2_abs_{suffix}_seed{args.seed}.csv"

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)
    early_stopping = EarlyStopping(patience=30, min_delta=1e-6)

    start_epoch = 0
    best_val_loss = float('inf')

    log_data = []
    print("\n开始训练...")
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        log_data.append([epoch, train_loss, val_loss, current_lr])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'seed': args.seed,
            }, model_path)
            print(
                f"Epoch {epoch:3d} | LR {current_lr:.2e} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} (saved)")
        else:
            print(f"Epoch {epoch:3d} | LR {current_lr:.2e} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    # 保存训练日志
    df_log = pd.DataFrame(log_data, columns=['epoch', 'train_loss', 'val_loss', 'learning_rate'])
    df_log.to_csv(log_file, index=False)
    print(f"Training log saved to {log_file}")

    # ======================================================================
    # 测试评估（在绝对面积上）
    # ======================================================================
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    def calc_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    all_true_abs = []
    all_pred_abs = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            batch_size = data.num_graphs
            pred_profile = model(data)
            true_profile = data.y.view(batch_size, 51)
            true_area = data.total_area.view(batch_size, 1)

            pred_abs = pred_profile * true_area
            true_abs = true_profile * true_area

            all_true_abs.append(true_abs.cpu().numpy())
            all_pred_abs.append(pred_abs.cpu().numpy())

    all_true_abs = np.vstack(all_true_abs)
    all_pred_abs = np.vstack(all_pred_abs)

    r2 = calc_r2(all_true_abs, all_pred_abs)
    mae = np.mean(np.abs(all_true_abs - all_pred_abs))
    rmse = np.sqrt(np.mean((all_true_abs - all_pred_abs) ** 2))

    print(f"\n{'=' * 50}")
    print(f"Test Set Results (Absolute Area) - Seed {args.seed}")
    print(f"{'=' * 50}")
    print(f"Loss function: {args.loss}")
    print(f"R²:  {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Best model saved as {model_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()