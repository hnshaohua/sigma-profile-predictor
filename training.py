#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""sigma-profile model training script (Area-Enhanced V2)

Core strategy: loss is computed on absolute area during training
    - Model output: relative area (51 dimensions, normalized)
    - Loss computation: predicted relative area x true total area = absolute area, loss computed on absolute area
    - At inference: predicted relative area x predicted total area = absolute area

Project structure:
    sigma-profile-predictor/
    ├── src/
    │   ├── featurizer.py      # atom/bond feature encoder, molecule-to-graph
    │   ├── model.py           # MPNN_V2 model definition
    │   └── predict.py         # prediction inference
    ├── training.py            # this file - training script
    ├── models/                # model weight save directory
    ├── loss_log_v2_abs_*.csv  # training log (auto-generated)
    └── testset_predictions_*.csv  # test set prediction results (auto-generated)

Usage:
    # Basic training (use mse loss, default)
    python training.py

    # Use composite loss
    python training.py --loss composite

    # Specify data file and output directory
    python training.py --data sigmaprofile.csv --output-dir ./outputs

    # Resume interrupted training
    python training.py --resume

    # Custom hyperparameters
    python training.py --lr 5e-5 --batch-size 64 --epochs 300 --patience 40

    # Specify random seed (for reproducibility)
    python training.py --seed 42
"""

import os
import sys
import argparse
import logging
import random
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split

# ======================================================================
# 项目根目录自动推导（支持从任意目录运行）
# ======================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.featurizer import molecule_to_graph
from src.model import MPNN_V2

# ======================================================================
# 日志配置
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ======================================================================
# 随机种子设置（确保实验可复现）
# ======================================================================
def set_seed(seed: int):
    """Set random seed for experimental reproducibility.

    Args:
        seed: random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to: {seed}")


# ======================================================================
# 损失函数（在绝对面积上计算）
# ======================================================================
def mse_loss_absolute(pred_profile: torch.Tensor, true_profile: torch.Tensor, true_area: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss on absolute area.

    Multiply predicted relative area by true total area to recover absolute area, then compute MSE.

    Args:
        pred_profile: predicted relative area, shape (batch_size, 51)
        true_profile: true relative area, shape (batch_size, 51)
        true_area: true molecular total area, shape (batch_size, 1)

    Returns:
        scalar loss value
    """
    pred_abs = pred_profile * true_area
    true_abs = true_profile * true_area
    return F.mse_loss(pred_abs, true_abs)


def composite_loss_absolute(
    pred_profile: torch.Tensor,
    true_profile: torch.Tensor,
    true_area: torch.Tensor,
    eps_rel: float = 0.01,
) -> torch.Tensor:
    """Compute composite loss on absolute area (relative error + log-squared error).

    Composite loss combines relative version of absolute error and log-squared error, fitting both large and small values.

    Args:
        pred_profile: predicted relative area, shape (batch_size, 51)
        true_profile: true relative area, shape (batch_size, 51)
        true_area: true molecular total area, shape (batch_size, 1)
        eps_rel: safe denominator threshold for relative error to prevent division by zero

    Returns:
        scalar loss value
    """
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


LOSS_FUNCTIONS = {
    "mse": mse_loss_absolute,
    "composite": composite_loss_absolute,
}


# ======================================================================
# 数据加载
# ======================================================================
def parse_sigma_profile(s: str) -> np.ndarray:
    """Parse SigmaProfile string to numpy array.

    Args:
        s: string in the form of "[0.1,0.2,...]" or "0.1,0.2,..."

    Returns:
        float32 numpy array of shape (51,)

    Raises:
        ValueError: raised if parsing fails
    """
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    try:
        return np.array([float(x) for x in s.split(",")], dtype=np.float32)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse SigmaProfile: '{s[:50]}...'") from e


def load_data(csv_path: str) -> tuple:
    """Load molecular graph data and raw DataFrame from CSV file.

    For each molecule:
        1. Build graph data from SMILES (call src.featurizer.molecule_to_graph)
        2. Parse SigmaProfile and normalize to relative area (y)
        3. Compute true total area (total_area)
        4. Keep original absolute area (y_abs)

    Args:
        csv_path: path to data CSV file, must contain 'SMILES' and 'SigmaProfile' columns

    Returns:
        (graph_data, df): list of graph data and raw DataFrame

    Raises:
        FileNotFoundError: if data file does not exist
        ValueError: if required columns are missing or data format is incorrect
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # 列名检查
    required_cols = {"SMILES", "SigmaProfile"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Data file missing required columns: {missing}. Current columns: {list(df.columns)}")

    graph_data = []
    total_areas = []
    skipped = 0

    for i, row in df.iterrows():
        g = molecule_to_graph(row["SMILES"])
        if g is None:
            logger.warning(f"Skipping invalid SMILES (row {i}): {row['SMILES'][:30]}")
            skipped += 1
            continue

        try:
            profile = parse_sigma_profile(str(row["SigmaProfile"]))
        except ValueError as e:
            logger.warning(f"Skipping data parsing failure (row {i}): {e}")
            skipped += 1
            continue

        # 负值截断
        if (profile < 0).any():
            profile = np.clip(profile, 0, None)

        total_area = float(np.sum(profile))
        if total_area > 0:
            profile_norm = profile / total_area
        else:
            profile_norm = profile

        g.y = torch.tensor(profile_norm, dtype=torch.float)
        g.total_area = torch.tensor(total_area, dtype=torch.float)
        g.y_abs = torch.tensor(profile, dtype=torch.float)
        graph_data.append(g)
        total_areas.append(total_area)

    if len(graph_data) == 0:
        raise ValueError("No valid molecular data! Please check if SMILES are correct.")

    logger.info(f"Valid molecules: {len(graph_data)}")
    logger.info(f"Total area range: [{np.min(total_areas):.2f}, {np.max(total_areas):.2f}]")
    if skipped > 0:
        logger.info(f"Skipped {skipped} invalid data entries")

    return graph_data, df


# ======================================================================
# 训练与评估
# ======================================================================
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    criterion,
    device: torch.device,
) -> float:
    """Execute one epoch of training.

    Args:
        model: MPNN_V2 model
        loader: training data DataLoader (shuffle=True)
        optimizer: AdamW optimizer
        criterion: loss function
        device: compute device (CPU / CUDA)

    Returns:
        average loss value for this epoch
    """
    model.train()
    total_loss = 0.0
    batch_count = 0

    for data in loader:
        data = data.to(device)
        batch_size = data.num_graphs
        optimizer.zero_grad()

        pred_profile = model(data)
        true_profile = data.y.view(batch_size, 51)
        true_area = data.total_area.view(batch_size, 1)

        loss = criterion(pred_profile, true_profile, true_area)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Loss anomaly (NaN/Inf), skipping this batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch_size
        batch_count += 1

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> float:
    """Evaluate model on validation/test set.

    Args:
        model: MPNN_V2 model
        loader: validation/test data DataLoader (shuffle=False)
        criterion: loss function
        device: compute device

    Returns:
        average loss value
    """
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
# 早停机制
# ======================================================================
class EarlyStopping:
    """Early stopping controller. Triggers stop when validation loss does not improve for specified epochs.

    Args:
        patience: maximum allowed epochs without improvement (default 30)
        min_delta: minimum improvement threshold, below this is not counted as improvement (default 1e-6)
    """

    def __init__(self, patience: int = 30, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
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
# R² 计算
# ======================================================================
def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared (R²) coefficient.

    Args:
        y_true: true value array
        y_pred: predicted value array

    Returns:
        R² value, range (-inf, 1], 1 for perfect prediction
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ======================================================================
# 主函数
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="sigma-profile model training script (Area-Enhanced V2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic training (recommended: composite loss)
  python training.py

  # Use MSE loss
  python training.py --loss mse

  # Specify data file and output directory
  python training.py --data sigmaprofile.csv --output-dir ./outputs

  # Resume interrupted training
  python training.py --resume

  # Custom hyperparameters
  python training.py --lr 5e-5 --batch-size 64 --epochs 300 --patience 40

  # Specify random seed (for reproducibility)
  python training.py --seed 42
        """,
    )

    # 数据相关
    parser.add_argument(
        "--data",
        type=str,
        default="sigmaprofile.csv",
        help="Path to training data CSV file (must contain SMILES and SigmaProfile columns)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Model and log output directory (default: current directory)",
    )

    # 训练超参数
    parser.add_argument(
        "--loss",
        type=str,
        choices=["mse", "composite"],
        default="mse",
        help="Loss function: mse or composite (default: composite)",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Maximum training epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (epochs without improvement before stopping, default: 30)",
    )

    # 数据划分
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.5,
        help="Test set proportion of validation+test split (i.e., test = val_size * test_size of remaining, default: 0.5)",
    )

    # 其他
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置随机种子
    set_seed(args.seed)

    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")

    logger.info(f"Loss function: {args.loss}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max epochs: {args.epochs}")
    logger.info(f"Early stopping patience: {args.patience}")

    # 输出路径
    criterion = LOSS_FUNCTIONS[args.loss]
    suffix = args.loss
    model_path = os.path.join(args.output_dir, f"best_model_v2_abs_{suffix}.pt")
    log_file = os.path.join(args.output_dir, f"loss_log_v2_abs_{suffix}.csv")
    config_file = os.path.join(args.output_dir, f"training_config_{suffix}.json")

    # 保存训练配置（便于复现）
    config = {
        "script": "training.py",
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"Training config saved to: {config_file}")

    # 加载数据
    logger.info(f"Loading data: {args.data}")
    graph_data, df = load_data(args.data)

    # 数据集划分
    n_samples = len(graph_data)
    train_idx, temp_idx = train_test_split(
        range(n_samples), test_size=args.val_size, random_state=args.seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=args.test_size, random_state=args.seed
    )

    train_loader = DataLoader(
        [graph_data[i] for i in train_idx],
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        [graph_data[i] for i in val_idx],
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        [graph_data[i] for i in test_idx],
        batch_size=args.batch_size,
        shuffle=False,
    )

    logger.info(f"Train set: {len(train_loader.dataset)} samples")
    logger.info(f"Validation set: {len(val_loader.dataset)} samples")
    logger.info(f"Test set: {len(test_loader.dataset)} samples")

    # 模型初始化
    from src.featurizer import atom_feat, bond_feat

    model = MPNN_V2(
        node_in_dim=atom_feat.dim,
        edge_dim=bond_feat.dim,
        hidden_dim=256,
        num_layers=4,
        output_dim=51,
        dropout=0.2,
        use_softplus=True,
    ).to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total model parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # 优化器与调度器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, verbose=True
    )
    early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-6)

    # 恢复训练
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(model_path):
        logger.info(f"Resuming from checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        logger.info(f"Resumed epoch: {start_epoch}, best val loss: {best_val_loss:.6f}")

    # ==================== 训练循环 ====================
    logger.info("\n" + "=" * 60)
    logger.info("Training started")
    logger.info("=" * 60)

    log_data = []

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        log_data.append([epoch, train_loss, val_loss, current_lr])

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                model_path,
            )
            logger.info(
                f"Epoch {epoch:3d} | LR {current_lr:.2e} | Train loss: {train_loss:.6f} | "
                f"Val loss: {val_loss:.6f} [saved]"
            )
        else:
            logger.info(
                f"Epoch {epoch:3d} | LR {current_lr:.2e} | Train loss: {train_loss:.6f} | "
                f"Val loss: {val_loss:.6f}"
            )

        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered, training stopped at epoch {epoch}")
            break

    # 保存训练日志
    df_log = pd.DataFrame(log_data, columns=["epoch", "train_loss", "val_loss", "learning_rate"])
    df_log.to_csv(log_file, index=False)
    logger.info(f"Training log saved to: {log_file}")

    # ==================== 测试集评估 ====================
    logger.info("\n" + "=" * 60)
    logger.info("Test set evaluation")
    logger.info("=" * 60)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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
    mae = float(np.mean(np.abs(all_true_abs - all_pred_abs)))

    logger.info(f"Loss function: {args.loss}")
    logger.info(f"R² : {r2:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"Best model: {model_path}")
    logger.info("=" * 60)

    # ==================== 保存预测结果 ====================
    test_smiles = [df.iloc[i]["SMILES"] for i in test_idx]
    rows = []
    for smiles, true_abs_row, pred_abs_row in zip(test_smiles, all_true_abs, all_pred_abs):
        row = {"smiles": smiles}
        row["true_total_area"] = float(np.sum(true_abs_row))
        for j in range(51):
            row[f"true_{j}"] = float(true_abs_row[j])
        for j in range(51):
            row[f"pred_{j}"] = float(pred_abs_row[j])
        rows.append(row)

    result_df = pd.DataFrame(rows)
    output_file = os.path.join(
        args.output_dir,
        f"testset_predictions_{suffix}.csv",
    )
    result_df.to_csv(output_file, index=False)
    logger.info(f"Test set prediction results saved to: {output_file}")


if __name__ == "__main__":
    main()
