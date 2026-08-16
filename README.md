# Sigma-Profile Predictor

A message-passing neural network (MPNN) model for predicting molecular sigma-profiles from SMILES strings.

---

## Installation

### Requirements

| Component | Version |
|---|---|
| Python | 3.8 (tested); <=3.11 |
| PyTorch | 2.0.x |
| CUDA | 11.7 (recommended); 11.8 or 12.1 |
| PyTorch Geometric | >=2.5.0 |

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

- Dataset: `data/sigma-profile.csv`
- Models: `models/`

---

## Project Structure

```
sigma-profile-predictor/
├── src/                # Library code
│   ├── __init__.py
│   ├── featurizer.py   # Molecular featurization
│   ├── model.py        # MPNN model definition
│   └── predict.py      # Prediction logic
├── data/               # Dataset
│   └── sigma-profile.csv
├── models/             # Pretrained models
│   ├── mse.pt
│   └── composite.pt
├── training.py         # Training script
├── predict.py          # Prediction script
├── requirements.txt    # Python dependencies
├── README.md           # This file (installation)
└── USAGE.md            # This file (usage instructions)
```
