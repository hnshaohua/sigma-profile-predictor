"""
快速预测入口脚本（命令行使用）

Usage：
    python predict.py --input input.csv
    python predict.py --input input.csv --output result.csv --model mse
"""

import argparse
import sys
import os

# 将 src 目录加入搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.predict import predict_sigma_profile


def main():
    parser = argparse.ArgumentParser(description="Sigma-Profile Predictor Tool")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input CSV file (must contain smiles column, area column is optional)")
    parser.add_argument("--output", "-o", type=str, default="prediction_result.csv",
                        help="Path to output CSV file")
    parser.add_argument("--model", "-m", type=str, default="mse",
                        choices=["mse", "abs_mse", "composite"],
                        help="Model type (default: mse)")
    parser.add_argument("--device", "-d", type=str, default=None,
                        help="Computation device (e.g., cuda:0, cpu, auto-select if not specified)")

    args = parser.parse_args()

    predict_sigma_profile(
        input_file=args.input,
        output_file=args.output,
        model_type=args.model,
        device=args.device,
    )
    print(f"Prediction complete, results saved to: {args.output}")



if __name__ == "__main__":
    main()