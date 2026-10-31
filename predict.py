"""
predict.py
Predict sigma-profiles from SMILES using the pre-trained MPNN model.
"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors

# Import from src
from src.featurizer import atom_feat, bond_feat, molecule_to_graph
from src.model import MPNN_V2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_area_desc(mol, features):
    """Calculate 11 2D descriptors for area prediction."""
    vals = []
    for feat in features:
        try:
            if feat == 'labute_asa': v = rdMolDescriptors.CalcLabuteASA(mol)
            elif feat == 'tpsa': v = rdMolDescriptors.CalcTPSA(mol)
            elif feat == 'molwt': v = Descriptors.MolWt(mol)
            elif feat == 'labute_volume': v = rdMolDescriptors.CalcLabuteVolume(mol)
            elif feat == 'heavy_atom_count': v = mol.GetNumHeavyAtoms()
            elif feat == 'total_atom_count': v = mol.GetNumAtoms()
            elif feat == 'ring_count': v = rdMolDescriptors.CalcNumRings(mol)
            elif feat == 'aromatic_ring_count': v = rdMolDescriptors.CalcNumAromaticRings(mol)
            elif feat == 'rotatable_bonds': v = rdMolDescriptors.CalcNumRotatableBonds(mol)
            elif feat == 'hbd': v = rdMolDescriptors.CalcNumHBD(mol)
            elif feat == 'hba': v = rdMolDescriptors.CalcNumHBA(mol)
            else: v = np.nan
        except Exception:
            v = np.nan
        vals.append(v)
    return np.array(vals).reshape(1, -1)

def main():
    parser = argparse.ArgumentParser(description="Predict Sigma Profiles")
    parser.add_argument('--input', required=True, help="Input CSV with a 'SMILES' column")
    parser.add_argument('--output', default='predictions.csv', help="Output CSV file")
    parser.add_argument('--model', default='models/mse_best.pt', help="Path to MPNN model")
    parser.add_argument('--area_model', default='models/area_predictor_model.pkl', help="Path to area model")
    parser.add_argument('--has_area', action='store_true', help="Set this flag if your input CSV has a 'true_area' column")
    args = parser.parse_args()

    # Load MPNN
    model = MPNN_V2(atom_feat.dim, bond_feat.dim, 256, 4, 51, 0.2).to(DEVICE)
    ckpt = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load Area Model
    with open(args.area_model, 'rb') as f:
        area_data = pickle.load(f)
    area_model = area_data['model']
    area_scaler = area_data['scaler']
    area_features = area_data['features']

    df = pd.read_csv(args.input)
    if 'SMILES' not in df.columns:
        raise ValueError("Input CSV must contain a 'SMILES' column.")

    results = []
    for _, row in df.iterrows():
        smi = row['SMILES']
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        
        # 1. Predict relative profile
        g = molecule_to_graph(smi)
        g.batch = torch.zeros(g.x.size(0), dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            pred_rel = model(g.to(DEVICE)).cpu().numpy().flatten()
        
        # 2. Determine Area
        if args.has_area and 'true_area' in df.columns:
            total_area = float(row['true_area'])
        else:
            X = calc_area_desc(mol, area_features)
            total_area = area_model.predict(area_scaler.transform(X))[0]
            
        # 3. Calculate absolute profile
        pred_abs = np.maximum(pred_rel, 0.0) * total_area
        
        res = {'SMILES': smi, 'predicted_total_area': total_area}
        for j in range(51):
            res[f'sigma_{j}'] = pred_abs[j]
        results.append(res)

    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == '__main__':
    main()