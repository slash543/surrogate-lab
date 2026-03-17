# SurrogateLab

A modular, open-source ML pipeline for building surrogate models of mechanical finite element simulations. Predicts **contact pressure fields** from FEBio `.xplt` simulation outputs using a config-driven MLP with full MLflow experiment tracking.

---

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Prepare Your Data](#prepare-your-data)
- [Quick Start](#quick-start)
- [Tutorial: Train Your First Surrogate Model](#tutorial-train-your-first-surrogate-model)
  - [Step 1: Configure the pipeline](#step-1-configure-the-pipeline)
  - [Step 2: Load and explore data](#step-2-load-and-explore-data)
  - [Step 3: Engineer features and split data](#step-3-engineer-features-and-split-data)
  - [Step 4: Build and train the model](#step-4-build-and-train-the-model)
  - [Step 5: Evaluate and visualize results](#step-5-evaluate-and-visualize-results)
  - [Step 6: Inspect with MLflow](#step-6-inspect-with-mlflow)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Extending the Pipeline](#extending-the-pipeline)
- [Running Tests](#running-tests)
- [License](#license)

---

## Overview

SurrogateLab replaces expensive FEBio finite element solves with a fast neural network. Given per-facet geometry and an insertion depth, the model predicts the contact pressure at each facet — turning a multi-minute simulation into a millisecond inference.

**Key properties:**

| Property | Detail |
|---|---|
| Input | Per-facet centroid (x, y, z), facet area, insertion depth |
| Output | Contact pressure (scalar per facet) |
| Model | Configurable MLP (depth, width, activation) |
| Tracking | MLflow — params, metrics, artifacts, model registry |
| Deployment | CLI script compatible with Azure ML |
| Licenses | Apache 2.0 / BSD / MIT only — commercial-use safe |

---

## Getting Started

### Prerequisites

- Python 3.10+
- `make` (optional but recommended)
- A CUDA-capable GPU (optional — falls back to CPU automatically)

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd surrogate-lab

# Create a virtual environment and install dependencies
make env

# Or manually:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For notebook support:

```bash
make env-full
# or: pip install jupyter ipykernel
```

### Prepare Your Data

Data must come from the `../xplt-parser/` pipeline. Each simulation produces one CSV file with the following columns:

| Column | Type | Description |
|---|---|---|
| `facet_id` | int | Facet identifier |
| `centroid_x` | float | Facet centroid X coordinate |
| `centroid_y` | float | Facet centroid Y coordinate |
| `centroid_z` | float | Facet centroid Z coordinate |
| `facet_area` | float | Surface area of the facet |
| `contact_pressure` | float | Target: contact pressure at this facet |
| `time_step` | float | Simulation time step (auto-converted to insertion depth) |

Place your CSV files in `data/simulations/`:

```bash
cp /path/to/your/sim_*.csv data/simulations/
```

> **Note:** If your CSV already has an `insertion_depth` column, the time-step conversion is skipped automatically.

---

## Quick Start

```bash
# Activate your environment
source .venv/bin/activate

# Train with default config
make train
# or: python scripts/train.py

# View results in MLflow UI
make mlflow
# open http://localhost:5000
```

---

## Tutorial: Train Your First Surrogate Model

This tutorial walks through the full pipeline end-to-end using the Python API directly. The same steps are available interactively in `notebooks/training.ipynb`.

### Step 1: Configure the pipeline

Everything is controlled by `configs/config.yaml`. Open it and verify the paths and hyperparameters:

```yaml
data:
  source: "data/simulations/"
  file_pattern: "*.csv"
  time_to_depth:
    method: linear
    scale: 1.0            # 1 mm per time unit — adjust to your simulation

features:
  inputs:
    - centroid_x
    - centroid_y
    - centroid_z
    - facet_area
    - insertion_depth
  target: contact_pressure
  normalization:
    method: standard      # 'standard' (z-score) or 'minmax'

model:
  type: MLP
  layers: [128, 128, 64]  # Hidden layer widths
  activation: relu
  dropout: 0.0

training:
  optimizer: adam
  lr: 0.001
  epochs: 200
  batch_size: 256
  early_stopping:
    enabled: true
    patience: 20

mlflow:
  experiment_name: surrogate-lab
  tracking_uri: mlruns/
```

No changes to Python code are needed — the pipeline reads this file at runtime.

### Step 2: Load and explore data

```python
from src.utils.config import load_config
from src.data.loader import load_simulation_data

cfg = load_config("configs/config.yaml")

# Loads all CSVs in data/simulations/, converts time_step → insertion_depth,
# validates that all required columns are present, returns a single DataFrame
df = load_simulation_data(cfg)

print(df.shape)          # (n_facets × n_timesteps, n_columns)
print(df.describe())
print(df["contact_pressure"].describe())
```

Example output:
```
(48000, 8)
       facet_id  centroid_x  ...  contact_pressure
count   48000.0     48000.0  ...           48000.0
mean      150.0       0.023  ...               1.4
std        86.6       2.841  ...               3.2
min         1.0      -8.120  ...               0.0
max       300.0       9.440  ...              18.6
```

### Step 3: Engineer features and split data

```python
from src.features.engineer import FeaturePipeline, build_xy
from src.features.splitter import split_data

# Extract feature matrix X and target vector y
X, y = build_xy(df, cfg)
print(f"X shape: {X.shape}")   # (48000, 5)
print(f"y shape: {y.shape}")   # (48000,)

# Split into train / val / test (70 / 15 / 15)
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, cfg)
print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

# Fit scalers on training data only (prevents data leakage)
pipeline = FeaturePipeline(cfg)
X_train_s, y_train_s = pipeline.fit_transform(X_train, y_train)
X_val_s,   y_val_s   = pipeline.transform(X_val,   y_val)
X_test_s,  y_test_s  = pipeline.transform(X_test,  y_test)

# Save scalers for later inference
pipeline.save("artifacts/scalers/")
```

### Step 4: Build and train the model

```python
import torch
from src.models.factory import build_model
from src.training.trainer import train

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

# Build MLP from config (input_dim inferred from feature count)
model = build_model(input_dim=X_train_s.shape[1], cfg=cfg)
print(model)
# MLP(
#   (layers): Sequential(
#     (0): Linear(5, 128)  (1): ReLU()
#     (2): Linear(128, 128) (3): ReLU()
#     (4): Linear(128, 64)  (5): ReLU()
#     (6): Linear(64, 1)
#   )
# )

# Train with early stopping and MLflow logging
model = train(
    model, X_train_s, y_train_s,
    X_val_s, y_val_s,
    cfg=cfg,
    run_name="tutorial-run-01"
)
```

Training output:
```
Epoch  10/200  train_loss=0.04821  val_loss=0.04612
Epoch  20/200  train_loss=0.02934  val_loss=0.02801
...
Epoch  87/200  train_loss=0.00312  val_loss=0.00298
Early stopping triggered (patience=20 exceeded).
Restored best model from checkpoints/best_model.pt (val_loss=0.00282)
```

### Step 5: Evaluate and visualize results

```python
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualization import (
    plot_predicted_vs_actual,
    plot_residuals,
    plot_training_curves,
)

# Compute RMSE, MAE, R² in original (physical) units
metrics = evaluate_model(model, X_test_s, y_test_s, pipeline, device)
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE:  {metrics['mae']:.4f}")
print(f"R²:   {metrics['r2']:.4f}")
```

Example output:
```
RMSE: 0.2134
MAE:  0.1401
R²:   0.9821
```

```python
# Get raw predictions for plotting
import numpy as np
model.eval()
with torch.no_grad():
    y_pred_s = model(torch.tensor(X_test_s).to(device)).cpu().numpy()

y_pred = pipeline.inverse_transform_y(y_pred_s)
y_true = pipeline.inverse_transform_y(y_test_s)

plot_predicted_vs_actual(y_true, y_pred, save_path="artifacts/pred_vs_actual.png")
plot_residuals(y_true, y_pred, save_path="artifacts/residuals.png")
```

This produces two diagnostic plots:

- **Predicted vs Actual** — scatter plot along the ideal diagonal; spread indicates error
- **Residuals** — residuals vs predicted values + histogram; ideally centered at zero

### Step 6: Inspect with MLflow

```bash
make mlflow
# or: mlflow ui --backend-store-uri mlruns/
```

Open [http://localhost:5000](http://localhost:5000) in your browser. You'll see:

- **Parameters**: model architecture, learning rate, batch size, optimizer, normalization method
- **Metrics**: `train_loss` and `val_loss` per epoch, final `rmse`, `mae`, `r2` on test set
- **Artifacts**: best model checkpoint, prediction plots, scaler files
- **Runs**: compare experiments side-by-side (e.g., different layer configs, activations)

To promote a model to the registry:

```yaml
# config.yaml
mlflow:
  register_model: true
  model_name: contact_pressure_surrogate
```

Then use the MLflow UI to transition from `Staging` to `Production`.

---

## Project Structure

```
surrogate-lab/
├── configs/
│   └── config.yaml          # Single source of truth — edit this, not code
│
├── src/
│   ├── data/
│   │   ├── loader.py        # load_simulation_data(): CSV → DataFrame
│   │   └── schema.py        # Column validation
│   ├── features/
│   │   ├── engineer.py      # FeaturePipeline, build_xy()
│   │   └── splitter.py      # split_data()
│   ├── models/
│   │   ├── mlp.py           # MLP(nn.Module)
│   │   └── factory.py       # build_model(), register_model()
│   ├── training/
│   │   └── trainer.py       # train(), EarlyStopping
│   ├── evaluation/
│   │   ├── metrics.py       # compute_metrics(), evaluate_model()
│   │   └── visualization.py # Prediction and residual plots
│   └── utils/
│       ├── config.py        # load_config(), get_feature_names()
│       └── logging_utils.py # get_logger()
│
├── scripts/
│   └── train.py             # CLI training entrypoint
│
├── notebooks/
│   └── training.ipynb       # Interactive walkthrough of full pipeline
│
├── tests/                   # pytest test suite
├── data/simulations/        # Place your CSV files here
├── artifacts/               # Scalers, plots (auto-created)
├── checkpoints/             # Best model weights (auto-created)
├── mlruns/                  # MLflow tracking (auto-created)
├── requirements.txt
└── Makefile
```

---

## Configuration Reference

All pipeline behaviour is controlled by `configs/config.yaml`:

```yaml
data:
  source: "data/simulations/"   # Directory containing CSV files
  file_pattern: "*.csv"         # Glob pattern to select files
  time_to_depth:
    method: linear              # Conversion method for time_step → insertion_depth
    scale: 1.0                  # Multiplier (e.g., 1.0 mm per time unit)

features:
  inputs:                       # Add/remove feature names here only
    - centroid_x
    - centroid_y
    - centroid_z
    - facet_area
    - insertion_depth
  target: contact_pressure      # Change to 'force', 'stress', etc. for other targets
  normalization:
    method: standard            # 'standard' (z-score) or 'minmax' (0–1)

split:
  train: 0.70
  val: 0.15
  test: 0.15
  random_seed: 42

model:
  type: MLP                     # Change to a registered custom model type
  layers: [128, 128, 64]        # Hidden layer widths (any depth)
  activation: relu              # relu | tanh | elu | gelu
  dropout: 0.0                  # Dropout probability (0.0 = disabled)

training:
  optimizer: adam
  lr: 0.001
  loss: mse
  epochs: 200
  batch_size: 256
  early_stopping:
    enabled: true
    patience: 20
    min_delta: 1.0e-6
  checkpoint:
    enabled: true
    dir: checkpoints/
    save_best: true

evaluation:
  metrics: [rmse, mae, r2]
  kfold:
    enabled: false
    n_splits: 5

mlflow:
  experiment_name: surrogate-lab
  tracking_uri: mlruns/
  log_artifacts: true
  register_model: false
  model_name: contact_pressure_surrogate
```

---

## Extending the Pipeline

### Add a new input feature

1. Ensure your CSV has the new column (e.g., `insertion_speed`)
2. Add it to `config.yaml`:
   ```yaml
   features:
     inputs:
       - ...
       - insertion_speed   # new
   ```
3. Re-run training. No code changes required.

### Change the prediction target

```yaml
features:
  target: von_mises_stress   # was: contact_pressure
```

Ensure that column exists in your CSV. The pipeline adapts automatically.

### Add a custom model

```python
# src/models/my_model.py
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        # ... define layers using cfg['model'] params

    def forward(self, x):
        # ... return shape (batch,)
```

Register it before training:

```python
from src.models.factory import register_model
from src.models.my_model import MyModel

register_model("MyModel", MyModel)
```

Switch to it in `config.yaml`:

```yaml
model:
  type: MyModel
```

### Add a new evaluation metric

Add your function to `src/evaluation/metrics.py`:

```python
def mean_relative_error(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))))
```

Then call it in `compute_metrics()`.

---

## Running Tests

```bash
# Run full test suite
make test

# Verbose output
pytest -v

# Single module
pytest tests/test_models.py -v

# With coverage report
pytest --cov src --cov-report term-missing
```

---

## CLI Reference

```bash
# Train with default config
python scripts/train.py

# Override config file
python scripts/train.py --config experiments/high_lr.yaml

# Override data source (single file or directory)
python scripts/train.py --data data/simulations/sim_01.csv

# Tag the MLflow run
python scripts/train.py --run-name ablation-no-dropout
```

### Makefile shortcuts

| Target | Description |
|---|---|
| `make env` | Create `.venv` and install core dependencies |
| `make env-dev` | + pytest and pytest-cov |
| `make env-full` | + jupyter and ipykernel |
| `make train` | Run `scripts/train.py` with default config |
| `make test` | Run pytest |
| `make mlflow` | Launch MLflow UI at localhost:5000 |
| `make clean` | Remove `.venv`, checkpoints, artifacts, mlruns |

---

## License

Apache 2.0 — see [LICENSE](LICENSE). All dependencies use MIT, BSD, or Apache 2.0 licenses. Safe for commercial use and resale.
