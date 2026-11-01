#!/usr/bin/env python
"""
Sigma-profile prediction script.
Usage:
    python predict.py --input input.csv --model mse --output predictions.csv
"""
import os
import argparse
import pickle
import inspect
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from src.featurizer import atom_feat, bond_feat, molecule_to_graph, calc_descriptors_for_area
from src.model import MPNN_V2

def main():
    parser = argparse.ArgumentParser(description="Predict Sigma Profiles")
    parser.add_argument("--input", required=True, help="Input CSV with a 'SMILES' column")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV file")
    parser.add_argument("--model", default="mse", choices=["mse", "composite"], help="Choose model type: mse or composite")
    parser.add_argument("--area_model", default="models/area_predictor_model.pkl", help="Path to area model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # 1. 根据 --model 参数映射到具体的 .pt 文件路径
    model_path = "models/mse_best.pt" if args.model == "mse" else "models/composite_best.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 2. 动态探测 MPNN_V2 的参数，彻底解决 edge_dim 报错问题
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

    model = MPNN_V2(**init_kwargs).to(device)
    
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Load Area Model
    print(f"Loading area model from {args.area_model}...")
    if not os.path.exists(args.area_model):
        raise FileNotFoundError(f"Area model file not found: {args.area_model}")
    with open(args.area_model, 'rb') as f:
        area_data = pickle.load(f)
    area_model_obj = area_data['model']
    area_scaler = area_data['scaler']
    area_features = area_data['features']

    # 4. Read Input
    df = pd.read_csv(args.input)
    smiles_col = 'SMILES' if 'SMILES' in df.columns else 'smiles'
    if smiles_col not in df.columns:
        raise ValueError("Input CSV must contain a 'SMILES' or 'smiles' column.")

    results = []
    print(f"Processing {len(df)} molecules...")
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            smi = row[smiles_col]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            
            # 推理相对 profile
            g = molecule_to_graph(smi)
            g.batch = torch.zeros(g.x.size(0), dtype=torch.long).to(device)
            pred_rel = model(g.to(device)).cpu().numpy().flatten()
            
            # 判断是否使用输入提供的真实面积
            if 'area' in df.columns or 'true_area' in df.columns:
                area_col = 'area' if 'area' in df.columns else 'true_area'
                total_area = float(row[area_col])
            else:
                X = calc_descriptors_for_area(mol, area_features)
                total_area = area_model_obj.predict(area_scaler.transform(X))[0]
                
            # 计算绝对面积
            pred_abs = np.maximum(pred_rel, 0.0) * total_area
            
            res = {'SMILES': smi, 'predicted_total_area': total_area}
            for j in range(51):
                res[f'sigma_{j}'] = pred_abs[j]
            results.append(res)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx+1}/{len(df)}")

    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    main()