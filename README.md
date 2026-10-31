# Sigma-Profile Predictor

A lightweight, dual-branch Message-Passing Neural Network (MPNN) model for predicting molecular sigma-profiles directly from SMILES strings. 

## Key Features
* **Pure SMILES Input:** Bypasses 3D conformer generation, force-field types, and large-scale pretraining during inference.
* **Dual-Branch Architecture:** Decouples shape prediction (via MPNN) and molecular size estimation (via 11 2D RDKit descriptors) to overcome the information loss of global pooling.
* **High Generalization:** Robustly validated across 5 independent random seeds and extensively tested on an out-of-distribution (OOD) external set of 642 larger molecules.
---

## Installation

### Requirements

* Python 3.8 - 3.11
* PyTorch 2.0.x
* PyTorch Geometric >= 2.5.0
* RDKit

### Step 1: Create virtual environment

```bash
conda create -n sigma-profile python=3.8
conda activate sigma-profile
```

### Step 2: Install PyTorch

CUDA 11.7 (recommended, tested):

```bash
pip install torch==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

CUDA 11.8:

```bash
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

CUDA 12.1:

```bash
pip install torch==2.0.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

CPU only:

```bash
pip install torch==2.0.0
```

### Step 3: Install PyTorch Geometric

CUDA 11.7:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

CUDA 11.8:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

CUDA 12.1:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib -f https://data.pyg.org/whl/torch-2.0.0+cu121.html
```

CPU only:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Step 4: Install other dependencies

```bash
pip install pandas numpy scikit-learn rdkit
```

### Step 5: Verify installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## Usage

See [USAGE.md](USAGE.md) for detailed instructions on prediction and training.

---

## Data and Models

The dataset and pretrained models are included in this repository:

data/sigma-profile.csv: The original training dataset (1,491 molecules).

data/external_642_eval.csv: The independent out-of-distribution external test set.

models/mse_best.pt: Pre-trained MPNN weights (MSE loss, seed 42).

models/area_predictor_model.pkl: Pre-trained linear regression model for total area prediction.

---

## Project Structure

```
sigma-profile-predictor/
├── data/                  # Datasets
├── examples/              # Example input/output CSVs
├── models/                # Pre-trained models (.pt and .pkl)
├── src/                   # Core library code (featurizer, model, predict)
├── training.py            # Script to train the MPNN model
├── predict.py             # Script to predict sigma-profiles
├── requirements.txt       # Python dependencies
├── README.md              # Installation guide
└── USAGE.md               # Usage instructions
```
