# House Prices Regression (Scikit-Learn)

End-to-end tabular regression project for predicting house prices with a **reproducible, protocol-driven** workflow using scikit-learn.

The project demonstrates:

- Fixed train/test split with **saved indices**
- All preprocessing inside a `Pipeline` / `ColumnTransformer`
- **Cross-validation** for model selection and diagnostics
- **Saved model artifacts** for inference
- **One-time** final evaluation on the held-out test set (Session 30)

---

## Repository structure

```text
.
├── README.md
├── data
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
│   ├── 10_error_analysis_and_features.ipynb
│   └── 11_final_test_evaluation.ipynb
├── reports
│   ├── metrics.md
│   ├── metrics_models.csv
│   ├── cv_results_leader.csv
│   ├── test_metrics.md
│   ├── test_top_errors.csv
│   └── figures
│       ├── learning_curve.png
│       ├── pdp_OverallQual.png
│       ├── pdp_GrLivArea.png
│       ├── pdp_GarageCars.png
│       ├── test_y_true_vs_y_pred.png
│       ├── test_residuals_hist.png
│       └── test_residuals_pred_vs_resid.png
├── src
│   ├── data.py          # load dataset + split indices
│   ├── eval.py          # CV evaluation + plots/tables
│   ├── features.py      # feature groups + ColumnTransformer
│   ├── model.py         # model definition (HGB + optional log1p target)
│   ├── predict.py       # inference script (CSV/Parquet -> predictions CSV)
│   └── train.py         # main training entrypoint
└── requirements.txt
```

Key components:

- `data/` — dataset + fixed train/test split indices  
- `models/` — saved model and metadata  
- `notebooks/` — EDA, baselines, tuning, interpretation, error analysis, and final test evaluation  
- `reports/` — metrics, CV results, test report, and figures  
- `src/` — all core code for loading, preprocessing, training, evaluation, and prediction  

> Note: If you see an extra folder `repor/` at the repository root, it is not used by the project and can be safely removed.

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Data

Place the main dataset at:

```text
data/raw/house_prices_train.csv
```

Expected target column:

```text
SalePrice
```

Fixed split indices (precomputed) must be present at:

```text
data/splits/train_indices.npy
data/splits/test_indices.npy
```

The training script **never** creates splits automatically.  
It **always** uses pre-saved indices to ensure fully reproducible evaluation.

---

## Training and cross-validation

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

1. **Loads data**
   - Reads the dataset from `data/raw/house_prices_train.csv`.
   - Loads fixed split indices from `data/splits/`.

2. **Builds preprocessing (`ColumnTransformer`)**
   - Numeric features: median imputation + standard scaling  
   - Categorical features: imputation + encoder  
   - Exact configuration is implemented in `src/features.py`.

3. **Builds the model**
   - `HistGradientBoostingRegressor` inside an end-to-end `Pipeline`.
   - Optional `TransformedTargetRegressor(log1p/expm1)` when `--target-transform log1p`.

4. **Runs cross-validation**
   - K-fold CV on the **training split only**.
   - Writes CV metrics and diagnostics to `reports/`.

5. **Fits final model and saves artifacts**
   - Trains the final model on the full **training split**.
   - Saves artifacts under `models/`.

This run evaluates **CV performance on training data** and produces a **reusable trained model artifact**.

---

## Final test evaluation (Session 30)

**Protocol rule:**  
The test split is evaluated **exactly once**, after all modeling choices are fixed (hyperparameters, target transform, feature set, etc.).

Final test evaluation artifacts are stored in:

- `reports/test_metrics.md`
- `reports/figures/test_y_true_vs_y_pred.png`
- `reports/figures/test_residuals_hist.png`
- `reports/figures/test_residuals_pred_vs_resid.png`
- `reports/test_top_errors.csv`

The corresponding notebook is:

- `notebooks/11_final_test_evaluation.ipynb`

---

## Final model results

**Final model:**  
`HGB_tuned` — tuned `HistGradientBoostingRegressor` with `log1p` target transform.

**Cross-validation (5-fold, train split):**

- RMSE (mean ± std): **28,566 ± 4,796**
- MAE (mean ± std): **17,169 ± 1,311**

**Test (held-out split, one-time):**

- RMSE: **26,345**
- MAE: **16,247**
- R²: **0.9095**

**Interpretation**

- Test performance is slightly **better** than the CV mean.
- Test metrics lie **within normal CV variability**.
- No clear sign of overfitting to the training data.

---

## Artifacts and reports

### Model artifacts

- `models/model.joblib`  
  - Trained end-to-end model saved with `joblib`.  
  - Includes **preprocessing + model** in a single object.

- `models/model_meta.json`  
  Contains metadata for full reproducibility, including:

  - Dataset paths and target column
  - Split information (train/test indices paths or parameters)
  - Feature groups / preprocessing configuration
  - Model hyperparameters
  - CV metrics summary
  - Timestamp, library versions, `random_state`

### Reports

- `reports/metrics_models.csv`  
  - CV metrics summary table (one row per `--model-name`).

- `reports/metrics.md`  
  - Human-readable log of experiments and conclusions.

Other files in `reports/` (e.g. `cv_results_leader.csv`, figures) are produced during:

- Baseline models
- Alpha searches (Ridge/Lasso)
- Hyperparameter tuning
- Interpretation (PDPs)
- Error analysis

---

## Inference

Once `models/model.joblib` has been created, you can generate predictions on any dataset that has the **same feature columns** as the training data.

Example:

```bash
python -m src.predict \
  --model-path models/model.joblib \
  --input-path data/raw/house_prices_train.csv \
  --output-path predictions/preds_on_train.csv \
  --pred-col SalePrice
```

This will:

- Load the saved pipeline from `models/model.joblib`.
- Apply the **same preprocessing** used in training.
- Produce predictions for `SalePrice`.
- Save them to `predictions/preds_on_train.csv`.

---
