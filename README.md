# P2 — House Prices Regression (Scikit-Learn)

This repository contains an end-to-end **tabular regression** project built with **scikit-learn**.  
Goal: train a reproducible model for predicting house prices using a strict ML protocol:
- fixed train/test split (saved indices),
- all preprocessing inside a `Pipeline`/`ColumnTransformer`,
- cross-validation for model selection,
- saved model artifacts for inference.

> Note: the **final test evaluation** is done in a later session (Session 30).  
> In this repo, the training script focuses on **CV evaluation on the training split** and saving artifacts.

---

## Repository structure

data/
raw/ # raw datasets (not modified by code)
splits/ # saved split indices (train_indices.npy, test_indices.npy)

models/
model.joblib # trained model (Pipeline / TransformedTargetRegressor)
model_meta.json # metadata for reproducibility (params, metrics, versions)

reports/
figures/ # saved plots (residuals, learning curve, etc.)
metrics.md # training logs (appended each run)
metrics_models.csv # summary table of model CV metrics
top_errors.csv # optional, created with --save-oof

src/
data.py # load dataset + split indices
features.py # feature groups + ColumnTransformer
model.py # model definition (HGB + optional log1p target)
eval.py # CV evaluation + plots/tables
train.py # main training entrypoint
predict.py # inference script (CSV/Parquet -> predictions CSV)


---

## Setup

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate

2) Install dependencies

Using requirements.txt:
pip install -r requirements.txt
(Recommended) upgrade pip first:
pip install --upgrade pip

---

Data
Place the training dataset here:

data/raw/house_prices_train.csv
This project expects the target column to be:

SalePrice

Also make sure you have saved split indices here:

data/splits/train_indices.npy
data/splits/test_indices.npy

The training script never creates splits automatically.
It uses the pre-saved indices to keep evaluation reproducible.

---

Reproduce training (CV + saving artifacts)

From the repository root, run:

python -m src.train \
  --data-path data/raw/house_prices_train.csv \
  --splits-dir data/splits \
  --target-col SalePrice \
  --target-transform log1p \
  --model-name HGB_tuned \
  --save-oof \
  --save-learning-curve

What this command does

Loads the dataset from data/raw/

Loads the fixed train/test split indices from data/splits/

Builds a preprocessing ColumnTransformer:

numeric: median imputation + standard scaling

categorical: most-frequent imputation + OrdinalEncoder (unknown -> -1)

Builds the model:

HistGradientBoostingRegressor inside a Pipeline

optionally wraps it with TransformedTargetRegressor(log1p/expm1) when --target-transform log1p

Runs KFold cross-validation on the training split and writes metrics to reports

Fits the final model on the full training split and saves artifacts

This run evaluates CV performance on training data.
Final test evaluation is performed separately (Session 30).

---

Outputs (artifacts and reports)

After a successful run you will get:

Model artifacts

models/model.joblib
The trained model saved with joblib.
This includes preprocessing + model (end-to-end object).

models/model_meta.json
Metadata for reproducibility:

data paths, target name

feature groups (numeric/categorical column lists)

model hyperparameters

CV metrics summary

library versions, timestamp, random_state

Reports

reports/metrics_models.csv
Table with CV metrics for the model name you pass (--model-name).

reports/metrics.md
A human-readable log appended after each training run.

Optional (if you use flags):

reports/figures/<MODEL>_oof_hist.png

reports/figures/<MODEL>_oof_pred_vs_resid.png
Residual analysis plots using OOF predictions (--save-oof)

reports/top_errors.csv
Top absolute errors table (--save-oof)

reports/figures/learning_curve.png
Learning curve plot (--save-learning-curve)

---

Inference

Once you have models/model.joblib, you can generate predictions on any dataset
(with the same feature columns as training).

Example:

python -m src.predict \
  --model-path models/model.joblib \
  --input-path data/raw/house_prices_train.csv \
  --output-path predictions/preds_on_train.csv \
  --pred-col SalePrice

---


Reproducibility notes

No preprocessing is done before splitting.

All preprocessing happens inside Pipeline / ColumnTransformer.

The split is fixed by saved indices in data/splits/.

Randomness is controlled via --random-state.