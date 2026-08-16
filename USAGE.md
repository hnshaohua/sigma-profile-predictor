# Usage Instructions

Detailed instructions for prediction and training.

---

## 1. Prediction

Use the `predict.py` script to predict sigma-profiles for molecules.

### 1.1 Input CSV Format

The input CSV file supports **two formats**:

#### Format A: With Molecular Surface Area

The CSV has two columns: `smiles` and `area`.

| Column | Description |
|---|---|
| smiles | SMILES string of the molecule |
| area | Molecular total surface area (Å²) |

Example:

```
smiles,area
CCO,120.5
c1ccccc1,156.3
```

#### Format B: Without Molecular Surface Area

The CSV has only one column: `smiles`.

| Column | Description |
|---|---|
| smiles | SMILES string of the molecule |

Example:

```
smiles
CCO
c1ccccc1
```

> **Note**: In Format B, the molecular surface area will be predicted by the model. In Format A, the input area value is used directly.

### 1.2 Command Line Usage

#### Basic prediction (Format A input):

```bash
python predict.py --input input.csv --output result.csv --model mse
```

#### Basic prediction (Format B input):

```bash
python predict.py --input input_no_area.csv --output result.csv --model composite
```

#### Specify device (GPU):

```bash
python predict.py --input input.csv --output result.csv --model mse --device cuda:0
```

### 1.3 Parameters

| Parameter | Description | Optional Values | Default |
|---|---|---|---|
| `--input`, `-i` | Input CSV file path (must contain `smiles` column, `area` column is optional) | — | Required |
| `--output`, `-o` | Output CSV file path | — | `prediction_result.csv` |
| `--model`, `-m` | Model type | `mse`, `composite` | `mse` |
| `--device`, `-d` | Compute device | `cpu`, `cuda:0`, `cuda:1`, etc. (auto-select if not specified) | Auto-select |

### 1.4 Output CSV Format

The output CSV file contains the following columns:

| Column | Column Name | Description |
|---|---|---|
| 1st | `smiles` | SMILES string |
| 2nd | `area` | Molecular total surface area (Å²). If Format A input, uses the input value; if Format B, uses the predicted value. |
| 3rd ~ 53rd | `val_0` ~ `val_50` | Predicted sigma-profile y-axis values (51 values in total). Unit: Å². |

**X-axis description:**

- Range: [-0.25, 0.25]
- Interval: 0.01
- Physical meaning: Charge density per unit surface area

### 1.5 Programming Usage

You can also import and use the prediction function directly:

```python
from src.predict import predict_sigma_profile

predict_sigma_profile(
    input_file="input.csv",
    output_file="result.csv",
    model_type="mse",
    device="cuda:0"
)
```

---

## 2. Training

Use the `training.py` script to train the MPNN model.

### 2.1 Training Data Format

The training data is a CSV file with the following columns:

| Column | Description |
|---|---|
| SMILES | SMILES string of the molecule |
| SigmaProfile | Sigma-Profile data (51 values, comma-separated) |

Example:

```
SMILES,SigmaProfile
CCO,0.0012,0.0034,...,0.0021
c1ccccc1,0.0056,0.0078,...,0.0043
```

### 2.2 Command Line Usage

```bash
python training.py --data data/sigma-profile.csv --model mse
```

### 2.3 Parameters

| Parameter | Description | Optional Values | Default |
|---|---|---|---|
| `--data`, `-d` | Training data file path (CSV) | — | `data/sigma-profile.csv` |
| `--model`, `-m` | Model type to train | `mse`, `composite` | `mse` |
| `--epochs`, `-e` | Number of training epochs | integer | 100 |
| `--lr`, `-l` | Learning rate | float | 0.001 |
| `--batch-size`, `-b` | Batch size | integer | 32 |
| `--device`, `-dev` | Compute device | `cpu`, `cuda:0`, etc. (auto-select if not specified) | Auto-select |
| `--output-dir`, `-o` | Model output directory | — | `models/` |

### 2.4 Output

Trained model files will be saved to the `models/` directory:

- `models/mse.pt` — MSE model weights
- `models/composite.pt` — Composite model weights

---

## 3. Data and Models

### 3.1 Dataset

The dataset file is included in this repository at `data/sigma-profile.csv`.

- Format: CSV
- Columns: `SMILES`, `SigmaProfile`
- Content: Molecular SMILES strings and corresponding sigma-profile data

### 3.2 Pretrained Models

Pretrained model files are included in this repository under `models/`.

- `models/mse.pt` — Pretrained MSE model
- `models/composite.pt` — Pretrained Composite model

---

## 4. Project Structure

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
