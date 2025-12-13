# P2 — House Prices Regression (Scikit-Learn)

This repository contains an end-to-end **tabular regression** project built with **scikit-learn**.

**Goal:** Train a reproducible model for predicting house prices using a strict ML protocol:

- Fixed train/test split (saved indices)
- All preprocessing inside a `Pipeline` / `ColumnTransformer`
- Cross-validation for model selection
- Saved model artifacts for inference

> In this repo, the training script focuses on **cross-validation (CV) evaluation on the training split** and saving artifacts.

---

## Repository structure

```text
.
├── README.md
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   ├── raw
│   │   └── house_prices_train.csv
│   └── splits
│       ├── test_indices.npy
│       └── train_indices.npy
├── models
│   ├── model.joblib
│   └── model_meta.json
├── notebooks
│   ├── 01_eda_and_split.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_ridge_lasso_alpha_search.ipynb
│   ├── 04_random_forest_baseline.ipynb
│   ├── 05_hist_gradient_boosting.ipynb
│   ├── 06_target_transformations.ipynb
│   ├── 07_leader_hyperparam_search.ipynb
│   ├── 08_learning_curve.ipynb
│   ├── 09_interpretation_pdp.ipynb
│   └── 10_error_analysis_and_features.ipynb
├── reports
│   ├── cv_results_leader.csv
│   ├── error_analysis_df.csv
│   ├── error_analysis_with_features.csv
│   ├── figures
│   │   ├── HGB_tuned_oof_hist.png
│   │   ├── HGB_tuned_oof_pred_vs_resid.png
│   │   ├── lasso_alpha_curve.png
│   │   ├── learning_curve.png
│   │   ├── pdp_GarageCars.png
│   │   ├── pdp_GrLivArea.png
│   │   ├── pdp_OverallQual.png
│   │   └── ridge_alpha_curve.png
│   ├── hgb_permutation_importances.csv
│   ├── lasso_alpha_search.csv
│   ├── metrics.md
│   ├── metrics_baseline.csv
│   ├── metrics_models.csv
│   ├── rf_feature_importances.csv
│   ├── ridge_alpha_search.csv
│   └── top_errors.csv
├── requirements.txt
├── src
│   ├── data.py          # load dataset + split indices
│   ├── eval.py          # CV evaluation + plots/tables
│   ├── features.py      # feature groups + ColumnTransformer
│   ├── house_prices
│   │   └── __init__.py
│   ├── model.py         # model definition (HGB + optional log1p target)
│   ├── predict.py       # inference script (CSV/Parquet -> predictions CSV)
│   └── train.py         # main training entrypoint
```

Key components:

- `data/` — datasets and fixed train/test split indices
- `models/` — saved model and metadata
- `notebooks/` — EDA, baseline models, hyperparameter search, interpretation, and error analysis
- `reports/` — metrics, CV results, error analysis tables, and figures
- `src/` — all code for data loading, features, training, evaluation, and prediction

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

### 2. Install dependencies

Using `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Data

Place the training dataset here:

```text
data/raw/house_prices_train.csv
```

This project expects the target column to be:

```text
SalePrice
```

The fixed split indices must be present at:

```text
data/splits/train_indices.npy
data/splits/test_indices.npy
```

The training script **never creates splits automatically**.  
It uses the **pre-saved indices** to keep evaluation reproducible.

---

## Reproduce training (CV + saving artifacts)

From the repository root, run:

```bash
python -m src.train \
  --data-path data/raw/house_prices_train.csv \
  --splits-dir data/splits \
  --target-col SalePrice \
  --target-transform log1p \
  --model-name HGB_tuned \
  --save-oof \
  --save-learning-curve
```

### What this command does

1. **Loads the dataset** from `data/raw/`.
2. **Loads the fixed train/test split indices** from `data/splits/`.
3. **Builds a preprocessing `ColumnTransformer`:**
   - Numeric: median imputation + standard scaling
   - Categorical: most-frequent imputation + `OrdinalEncoder` (unknown → `-1`)
4. **Builds the model:**
   - `HistGradientBoostingRegressor` inside a `Pipeline`
   - Optionally wraps it with `TransformedTargetRegressor(log1p/expm1)` when `--target-transform log1p`
5. **Runs K-fold cross-validation** on the training split and writes metrics to `reports/`.
6. **Fits the final model** on the full training split and saves artifacts into `models/`.

This run evaluates **CV performance on training data**.  
**Final test evaluation** is performed separately (Session 30).

---

## Outputs (artifacts and reports)

After a successful run you will get:

### Model artifacts

- `models/model.joblib`  
  The trained model saved with `joblib`.  
  This includes **preprocessing + model** (end-to-end object).

- `models/model_meta.json`  
  Metadata for reproducibility:
  - Data paths, target name  
  - Feature groups (numeric/categorical column lists)  
  - Model hyperparameters  
  - CV metrics summary  
  - Library versions, timestamp, `random_state`

### Reports

- `reports/metrics_models.csv`  
  Table with CV metrics for the model name you pass via `--model-name`.

- `reports/metrics.md`  
  A human-readable log appended after each training run.

Additional CSVs and plots in `reports/` come from experiments in the notebooks, such as:

- Baseline metrics
- Alpha searches for Ridge/Lasso
- Random forest and HGB feature/permutation importances
- Error analysis tables

### Optional (if you use flags)

If you pass `--save-oof`:

- `reports/figures/<MODEL>_oof_hist.png`  
- `reports/figures/<MODEL>_oof_pred_vs_resid.png`  
  - Residual analysis plots using OOF predictions

- `reports/top_errors.csv`  
  - Top absolute errors table

If you pass `--save-learning-curve`:

- `reports/figures/learning_curve.png`  
  - Learning curve plot

---

## Inference

Once you have `models/model.joblib`, you can generate predictions on any dataset  
(with the **same feature columns** as training).

Example:

```bash
python -m src.predict \
  --model-path models/model.joblib \
  --input-path data/raw/house_prices_train.csv \
  --output-path predictions/preds_on_train.csv \
  --pred-col SalePrice
```

This will save a CSV of predictions to the given `--output-path`.

---

## Reproducibility notes

- No preprocessing is done **before** splitting.
- All preprocessing happens inside `Pipeline` / `ColumnTransformer`.
- The split is fixed by saved indices in `data/splits/`.
- Randomness is controlled via `--random-state` (propagated through training and evaluation).