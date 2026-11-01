#!/usr/bin/env python
"""
test_reproducibility.py
End-to-end test to verify if the local repository structure, 
src.model definitions, and src.featurizer orders perfectly match the trained weights.
"""
import os
import sys
import argparse
import pickle
import inspect
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

# 按照与根目录 predict.py 完全相同的逻辑进行导入
from src.featurizer import atom_feat, bond_feat, molecule_to_graph, calc_descriptors_for_area
from src.model import MPNN_V2

def parse_profile(s):
    s = str(s).strip()
    if s.startswith('[') and s.endswith(']'): s = s[1:-1]
    return np.array([float(x) for x in s.split(',')], dtype=np.float32)

def calc_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return r2, mae

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. 设定文件路径
    data_path = "data/external_642_eval.csv"
    model_path = "models/mse_best.pt"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    # 2. 完全复用 predict.py 中的动态实例化逻辑
    print(f"Loading MPNN model from {model_path}...")
    sig = inspect.signature(MPNN_V2.__init__)
    init_kwargs = {}
    
    if 'node_in_dim' in sig.parameters: init_kwargs['node_in_dim'] = atom_feat.dim
    if 'edge_dim' in sig.parameters: init_kwargs['edge_dim'] = bond_feat.dim
    if 'hidden_dim' in sig.parameters: init_kwargs['hidden_dim'] = 256
    if 'num_layers' in sig.parameters: init_kwargs['num_layers'] = 4
    if 'output_dim' in sig.parameters: init_kwargs['output_dim'] = 51
    if 'dropout' in sig.parameters: init_kwargs['dropout'] = 0.2
    if 'use_softplus' in sig.parameters: init_kwargs['use_softplus'] = True

    # 实例化并加载权重
    model = MPNN_V2(**init_kwargs).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print("Model weights loaded successfully (no structure mismatch).")

    # 3. 读取外部数据集
    df = pd.read_csv(data_path)
    print(f"Processing {len(df)} external molecules...")

    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            smi = row['SMILES']
            
            # 读取真值
            true_profile = np.clip(parse_profile(row['SigmaProfile']), 0, None)
            true_area = float(true_profile.sum())
            y_true_list.append(true_profile)

            # 图构建与推理（使用 src.featurizer 中的 molecule_to_graph）
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                y_pred_list.append(np.zeros(51))
                continue

            g = molecule_to_graph(smi)
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long).to(device)
            
            # 模型前向传播
            pred_rel = model(g.to(device)).cpu().numpy().flatten()
            
            # 绝对面积还原 (使用 true_area)
            pred_abs = np.maximum(pred_rel, 0.0) * true_area
            y_pred_list.append(pred_abs)

    # 4. 计算指标
    y_true = np.vstack(y_true_list)
    y_pred = np.vstack(y_pred_list)

    print("\n" + "="*60)
    print("REPRODUCIBILITY TEST RESULTS (Single Model, External 642)")
    print("="*60)
    
    # 绝对面积指标
    r2_abs, mae_abs = calc_metrics(y_true.flatten(), y_pred.flatten())
    print(f"Absolute Area : R2 = {r2_abs:.4f} | MAE = {mae_abs:.4f}")
    
    # 归一化指标
    y_true_rel = y_true / y_true.sum(axis=1, keepdims=True)
    y_pred_rel = y_pred / y_pred.sum(axis=1, keepdims=True)
    r2_norm, mae_norm = calc_metrics(y_true_rel.flatten(), y_pred_rel.flatten())
    print(f"Normalized    : R2 = {r2_norm:.4f} | MAE = {mae_norm:.6f}")
    
    print("-" * 60)
    
    # 自动判定是否通过测试
    if r2_abs > 0.96:
        print("✅ TEST PASSED: The repository structure and model weights are consistent.")
        print("   The reproduced R2 closely matches the paper's reported value (0.967).")
    else:
        print("❌ TEST FAILED: The reproduced R2 is too low.")
        print("   This indicates a mismatch in featurizer order (e.g., bond type sorted vs unsorted) or model architecture.")
        print("   Please double-check src/featurizer.py to ensure sets are changed to lists without sorting.")
    print("="*60)

if __name__ == "__main__":
    main()