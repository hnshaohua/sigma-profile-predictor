#!/usr/bin/env python
"""分子特征编码模块。

本模块提供以下功能：
    - AtomFeaturizer: 原子特征编码器，将 RDKit 原子对象编码为 One-Hot 特征向量
    - BondFeaturizer: 键特征编码器，将 RDKit 键对象编码为 One-Hot 特征向量
    - molecule_to_graph: 将 SMILES 字符串转换为 PyTorch Geometric 图数据对象
    - calc_descriptors_for_area: 计算面积预测所需的分子描述符
    - get_supported_descriptors: 返回支持的描述符名称列表

全局实例 atom_feat 和 bond_feat 在模块导入时自动初始化，
可直接用于 molecule_to_graph 函数。如需自定义特征维度，
请自行实例化 AtomFeaturizer / BondFeaturizer 并传入。
"""

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors

# 面积预测模型支持的描述符名称列表
SUPPORTED_DESCRIPTORS = [
    "labute_asa",
    "tpsa",
    "molwt",
    "labute_volume",
    "heavy_atom_count",
    "total_atom_count",
    "ring_count",
    "aromatic_ring_count",
    "rotatable_bonds",
    "hbd",
    "hba",
]


class AtomFeaturizer:
    """原子特征编码器。

    将 RDKit 原子对象编码为 One-Hot 特征向量，编码的属性包括：
        - symbol: 原子符号（18 种：As, B, Br, C, Ca, Cl, F, H, I, K, Mg, N, Na, O, P, S, Se, Si）
        - n_valence: 价电子数（0~8）
        - n_hydrogens: 连接的氢原子数（0~4）
        - hybridization: 杂化方式（s, sp, sp2, sp3）
        - degree: 连接数（0~6）
        - charge: 形式电荷（-1, 0, 1）
        - in_ring: 是否在环中（0/1）
        - chiral: 是否具有手性（0/1）

    总特征维度为各属性维度之和。

    Attributes:
        dim: 特征向量总维度
        feature_maps: 各属性的值到索引的映射字典
    """

    def __init__(self):
        self.symbol_set = {
            "As", "B", "Br", "C", "Ca", "Cl", "F", "H", "I",
            "K", "Mg", "N", "Na", "O", "P", "S", "Se", "Si",
        }
        self.valence_set = {0, 1, 2, 3, 4, 5, 6, 7, 8}
        self.hydrogen_set = {0, 1, 2, 3, 4}
        self.hybridization_set = {"s", "sp", "sp2", "sp3"}
        self.degree_set = {0, 1, 2, 3, 4, 5, 6}
        self.charge_set = {-1, 0, 1}
        self.ring_set = {0, 1}
        self.chiral_set = {0, 1}

        self.dim = 0
        self.feature_maps = {}
        for name, s in [
            ("symbol", self.symbol_set),
            ("n_valence", self.valence_set),
            ("n_hydrogens", self.hydrogen_set),
            ("hybridization", self.hybridization_set),
            ("degree", self.degree_set),
            ("charge", self.charge_set),
            ("in_ring", self.ring_set),
            ("chiral", self.chiral_set),
        ]:
            s = sorted(list(s))
            self.feature_maps[name] = {v: i for i, v in enumerate(s)}
            self.dim += len(s)

    def encode(self, atom):
        """将单个 RDKit 原子编码为 One-Hot 特征向量。

        Args:
            atom: RDKit 原子对象 (rdkit.Chem.rdchem.Atom)

        Returns:
            numpy.ndarray: One-Hot 特征向量，形状为 (dim,)，dtype 为 float32
        """
        features = [0] * self.dim

        # 原子符号
        sym = atom.GetSymbol()
        if sym in self.feature_maps["symbol"]:
            features[self.feature_maps["symbol"][sym]] = 1.0

        offset = len(self.feature_maps["symbol"])

        # 价电子数
        valence = atom.GetTotalValence()
        if valence in self.feature_maps["n_valence"]:
            features[offset + self.feature_maps["n_valence"][valence]] = 1.0
        offset += len(self.feature_maps["n_valence"])

        # 氢原子数
        hs = atom.GetTotalNumHs()
        if hs in self.feature_maps["n_hydrogens"]:
            features[offset + self.feature_maps["n_hydrogens"][hs]] = 1.0
        offset += len(self.feature_maps["n_hydrogens"])

        # 杂化方式
        hyb = atom.GetHybridization().name.lower()
        if hyb in self.feature_maps["hybridization"]:
            features[offset + self.feature_maps["hybridization"][hyb]] = 1.0
        offset += len(self.feature_maps["hybridization"])

        # 连接数
        degree = atom.GetDegree()
        if degree in self.feature_maps["degree"]:
            features[offset + self.feature_maps["degree"][degree]] = 1.0
        offset += len(self.feature_maps["degree"])

        # 形式电荷
        charge = atom.GetFormalCharge()
        if charge in self.feature_maps["charge"]:
            features[offset + self.feature_maps["charge"][charge]] = 1.0
        offset += len(self.feature_maps["charge"])

        # 是否在环中
        in_ring = 1 if atom.IsInRing() else 0
        features[offset + self.feature_maps["in_ring"][in_ring]] = 1.0
        offset += len(self.feature_maps["in_ring"])

        # 手性
        chiral = 1 if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0
        features[offset + self.feature_maps["chiral"][chiral]] = 1.0

        return np.array(features, dtype=np.float32)


class BondFeaturizer:
    """键特征编码器。（修改版）"""

    def __init__(self):
        # 必须使用 list 保证顺序绝对稳定，与训练代码完全一致
        self.bond_type_set = ["single", "double", "triple", "aromatic"]
        self.conj_set = [True, False]
        self.ring_set = [True, False]
        self.dim = len(self.bond_type_set) + len(self.conj_set) + len(self.ring_set)

    def encode(self, bond):
        features = [0] * self.dim
        offset = 0

        # 键类型
        bt = bond.GetBondType().name.lower()
        if bt in self.bond_type_set:
            # 直接使用 list 的 index，千万不要加 sorted()
            features[offset + self.bond_type_set.index(bt)] = 1.0
        offset += len(self.bond_type_set)

        # 共轭
        conj = bond.GetIsConjugated()
        features[offset + (0 if conj else 1)] = 1.0
        offset += len(self.conj_set)

        # 是否在环中
        in_ring = bond.IsInRing()
        features[offset + (0 if in_ring else 1)] = 1.0

        return np.array(features, dtype=np.float32)


# 全局特征编码器实例（模块导入时自动初始化）
atom_feat = AtomFeaturizer()
bond_feat = BondFeaturizer()


def molecule_to_graph(smiles):
    """将 SMILES 字符串转换为 PyTorch Geometric 图数据对象。

    使用全局的 atom_feat 和 bond_feat 实例对分子进行特征编码。

    Args:
        smiles: SMILES 字符串

    Returns:
        torch_geometric.data.Data 对象，包含：
            - x: 节点特征，形状 (n_atoms, atom_feat.dim)
            - edge_index: 边索引，形状 (2, n_edges)
            - edge_attr: 边特征，形状 (n_edges, bond_feat.dim)
        如果 SMILES 无效（无法被 RDKit 解析），返回 None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 节点特征
    node_feat_list = [atom_feat.encode(atom) for atom in mol.GetAtoms()]
    node_features = torch.tensor(np.array(node_feat_list, dtype=np.float32), dtype=torch.float)

    # 边特征（双向边）
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
        edge_attr = torch.tensor(np.array(edge_attr_list, dtype=np.float32), dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, bond_feat.dim), dtype=torch.float)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


def calc_descriptors_for_area(mol, feature_names):
    """计算面积预测所需的分子描述符。

    根据 RDKit 分子对象计算一组连续型描述符，用于输入面积预测模型。

    Args:
        mol: RDKit 分子对象 (rdkit.Chem.rdchem.Mol)
        feature_names: 描述符名称列表，支持的名称见 get_supported_descriptors()

    Returns:
        numpy.ndarray: 描述符数组，形状为 (1, len(feature_names))，dtype 为 float64

    Raises:
        ValueError: 当 feature_names 中包含不支持的描述符名称时
    """
    supported = set(SUPPORTED_DESCRIPTORS)
    for name in feature_names:
        if name not in supported:
            raise ValueError(f"不支持的描述符: '{name}'，支持的描述符: {sorted(supported)}")

    desc_vals = []
    for feat in feature_names:
        try:
            if feat == "labute_asa":
                val = rdMolDescriptors.CalcLabuteASA(mol)
            elif feat == "tpsa":
                val = rdMolDescriptors.CalcTPSA(mol)
            elif feat == "molwt":
                val = Descriptors.MolWt(mol)
            elif feat == "labute_volume":
                val = rdMolDescriptors.CalcLabuteVolume(mol)
            elif feat == "heavy_atom_count":
                val = mol.GetNumHeavyAtoms()
            elif feat == "total_atom_count":
                val = mol.GetNumAtoms()
            elif feat == "ring_count":
                val = rdMolDescriptors.CalcNumRings(mol)
            elif feat == "aromatic_ring_count":
                val = rdMolDescriptors.CalcNumAromaticRings(mol)
            elif feat == "rotatable_bonds":
                val = rdMolDescriptors.CalcNumRotatableBonds(mol)
            elif feat == "hbd":
                val = rdMolDescriptors.CalcNumHBD(mol)
            elif feat == "hba":
                val = rdMolDescriptors.CalcNumHBA(mol)
            else:
                val = float("nan")
        except Exception:
            val = float("nan")
        desc_vals.append(val)

    return np.array(desc_vals, dtype=np.float64).reshape(1, -1)


def get_supported_descriptors():
    """返回面积预测模型支持的描述符名称列表。

    Returns:
        list[str]: 描述符名称列表
    """
    return list(SUPPORTED_DESCRIPTORS)
