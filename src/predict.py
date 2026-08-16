#!/usr/bin/env python
"""Sigma-profile prediction main module。

This module provides two usage modes:

    1. Command-line interface (CLI)：
        python -m src.predict --input input.csv --output predictions.csv

    2. Programmatic call：
        from src.predict import predict_sigma_profile
        predict_sigma_profile(input_file="input.csv", output_file="predictions.csv")

Prediction flow：
    1. Read input CSV (must contain smiles column)
    2. Load area prediction model (pickle file)
    3. Load MPNN model weights (.pt file)
    4. For each molecule：
        a. Compute descriptors with RDKit and predict total area via area model
        b. Convert molecule to graph data and predict sigma profile via MPNN
        c. Multiply sigma profile by total area to get absolute area profile
    5. Save results CSV

Assumed directory structure：
    Project root/
    ├── src/
    │   ├── __init__.py
    │   ├── featurizer.py
    │   ├── model.py
    │   └── predict.py
    └── models/
        ├── best_model_v2_abs_mse.pt
        ├── best_model_v2_abs_composite.pt
        └── area_predictor_model.pkl

Model files are loaded from the models/ subdirectory of the project root by default, and can be overridden via command-line arguments。
"""

import os
import sys
import argparse
import warnings
import pickle
import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from .featurizer import (
    atom_feat,
    bond_feat,
    molecule_to_graph,
    calc_descriptors_for_area,
    get_supported_descriptors,
)
from .model import MPNN_V2

# 模型文件默认路径（相对于 src 目录的上一级即项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_MSE_MODEL = os.path.join(DEFAULT_MODEL_DIR, "best_model_v2_abs_mse.pt")
DEFAULT_COMPOSITE_MODEL = os.path.join(DEFAULT_MODEL_DIR, "best_model_v2_abs_composite.pt")
DEFAULT_AREA_MODEL = os.path.join(DEFAULT_MODEL_DIR, "area_predictor_model.pkl")

warnings.filterwarnings("ignore")


def predict_sigma_profile(
    input_file,
    output_file,
    model_type="mse",
    smiles_col="smiles",
    model_mse=None,
    model_comp=None,
    area_model=None,
    device=None,
):
    """Batch predict sigma-profile。

    This is the core prediction function, supporting two usage modes：
        - Mode 1: Input CSV contains only smiles column, automatically predicts total area via area model
        - Mode 2: Input CSV contains both smiles and area columns, directly uses the input area values

    Args:
        input_file: Input CSV file path, must contain smiles column
        output_file: Output CSV file path
        model_type: Chosen MPNN model type, "mse" or "composite"
        smiles_col: SMILES column name (default "smiles")
        model_mse: MSE model file path, defaults to models/best_model_v2_abs_mse.pt
        model_comp: Composite loss model file path, defaults to models/best_model_v2_abs_composite.pt
        area_model: Area prediction model file path, defaults to models/area_predictor_model.pkl
        device: Inference device, "cpu" or "cuda", default auto-detect

    Returns:
        pandas.DataFrame: Predicted results DataFrame, containing smiles, area columns and 51 val_0~val_50 columns

    Raises:
        FileNotFoundError: When input file or model file does not exist
        ValueError: When input CSV lacks smiles column or model_type is invalid
    """
    # 使用默认路径
    if model_mse is None:
        model_mse = DEFAULT_MSE_MODEL
    if model_comp is None:
        model_comp = DEFAULT_COMPOSITE_MODEL
    if area_model is None:
        area_model = DEFAULT_AREA_MODEL
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 检查文件
    for fpath in [input_file, area_model, model_mse, model_comp]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"File not found: {fpath}")

    if model_type not in ("mse", "composite"):
        raise ValueError(f"model_type must be 'mse' or 'composite', current is '{model_type}'")

    # 读取输入
    df_in = pd.read_csv(input_file)
    if smiles_col not in df_in.columns:
        raise ValueError(
            f"Column '{smiles_col}' does not exist, available columns: {df_in.columns.tolist()}"
        )
    print(f"Loading input file: {input_file}, total {len(df_in)} molecules")

    # 加载面积模型
    with open(area_model, "rb") as f:
        area_data = pickle.load(f)
    area_model_obj = area_data["model"]
    area_scaler = area_data["scaler"]
    area_features = area_data["features"]
    print(f"Area model loaded successfully, features: {area_features}")

    # 加载 MPNN 模型
    model_path = model_mse if model_type == "mse" else model_comp
    device = torch.device(device)
    model = MPNN_V2(
        node_in_dim=atom_feat.dim,
        edge_dim=bond_feat.dim,
        hidden_dim=256,
        num_layers=4,
        output_dim=51,
        dropout=0.2,
        use_softplus=True,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"MPNN model loaded and set to eval() mode, device: {device}")

    # 预测循环
    results = []
    with torch.no_grad():
        for idx, row in df_in.iterrows():
            smiles = row[smiles_col]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: cannot parse SMILES: {smiles}")
                continue

            # 判断是否使用模式二（输入包含 area 列）
            if "area" in df_in.columns:
                pred_total_area = row["area"]
            else:
                # 模式一：通过面积模型预测总面积
                try:
                    desc_vals = calc_descriptors_for_area(mol, area_features)
                    desc_vals_scaled = area_scaler.transform(desc_vals)
                    pred_total_area = area_model_obj.predict(desc_vals_scaled)[0]
                except Exception as e:
                    print(f"Warning: area prediction failed for {smiles}: {e}")
                    continue

            # 图推理
            graph = molecule_to_graph(smiles)
            if graph is None:
                continue

            graph = graph.to(device)
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=device)

            output = model(graph)  # [1, 51]
            pred_profile = output.cpu().numpy().flatten()

            # 计算绝对面积 profile
            abs_profile = pred_profile * pred_total_area

            results.append({
                "smiles": smiles,
                "area": pred_total_area,
                "profile": abs_profile.tolist(),
            })

            if (idx + 1) % 100 == 0:
                print(f"Processed: {idx + 1}/{len(df_in)}")

    # 保存结果
    if results:
        df_out = pd.DataFrame(results)
        profile_df = pd.DataFrame(
            df_out["profile"].tolist(), columns=[f"val_{i}" for i in range(51)]
        )
        df_final = pd.concat([df_out.drop("profile", axis=1), profile_df], axis=1)
        df_final.to_csv(output_file, index=False)
        print(f"Prediction complete, results saved to: {output_file}")
        return df_final
    else:
        print("No prediction results generated.")
        return pd.DataFrame()


def main():
    """Command-line entry function。

    Supported command-line arguments：
        --input: Input CSV file path (required)
        --output: Output CSV file path (default predictions.csv)
        --smiles_col: SMILES column name (default smiles)
        --area_model: Area prediction model file path
        --model: Model type, "mse" or "composite" (default mse)
        --model_mse: MSE model file path
        --model_comp: Composite loss model file path
        --device: Inference device, "cpu" or "cuda" (default auto-detect)
    """
    parser = argparse.ArgumentParser(
        description="Sigma-profile prediction tool (MPNN + area prediction model)"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file (must contain smiles column)")
    parser.add_argument("--output", default="predictions.csv", help="Path to output CSV file")
    parser.add_argument("--smiles_col", default="smiles", help="SMILES column name")
    parser.add_argument(
        "--area_model",
        default=DEFAULT_AREA_MODEL,
        help="Path to area prediction model file (.pkl)",
    )
    parser.add_argument(
        "--model",
        choices=["mse", "composite"],
        default="mse",
        help='Choose MPNN model type: "mse" or "composite"',
    )
    parser.add_argument(
        "--model_mse",
        default=DEFAULT_MSE_MODEL,
        help="Path to model file trained with MSE loss (.pt)",
    )
    parser.add_argument(
        "--model_comp",
        default=DEFAULT_COMPOSITE_MODEL,
        help="Path to model file trained with composite loss (.pt)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help='Inference device (default auto-detect, uses cuda if GPU available)',
    )

    args = parser.parse_args()

    try:
        predict_sigma_profile(
            input_file=args.input,
            output_file=args.output,
            model_type=args.model,
            smiles_col=args.smiles_col,
            model_mse=args.model_mse,
            model_comp=args.model_comp,
            area_model=args.area_model,
            device=args.device,
        )
    except Exception as e:
        print(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
