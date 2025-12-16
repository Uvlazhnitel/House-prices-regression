# Test metrics

**Target:** `SalePrice`  
**Model artifact:** `models/model.joblib`

## Metrics (test)
- RMSE: **26344.8763**
- MAE: **16246.5627**
- R2: **0.9095**

## Figures
- y_true vs y_pred: `reports/figures/test_y_true_vs_y_pred.png`
- residual histogram: `reports/figures/test_residuals_hist.png`
- y_pred vs residual: `reports/figures/test_residuals_pred_vs_resid.png`

## Tables
- top errors: `reports/test_top_errors.csv`

### Model meta
```json
{
  "timestamp_utc": "2025-12-15T15:26:28Z",
  "python_version": "3.13.7",
  "platform": "macOS-26.1-arm64-arm-64bit-Mach-O",
  "numpy_version": "2.3.5",
  "pandas_version": "2.3.3",
  "sklearn_version": "1.7.2",
  "data_path": "data/raw/house_prices_train.csv",
  "splits_dir": "data/splits",
  "target_col": "SalePrice",
  "index_kind": "positional",
  "target_transform": "raw",
  "random_state": 42,
  "cv_splits": 5,
  "main_metric": "rmse",
  "secondary_metric": "mae",
  "model_name": "HGB_tuned",
  "feature_groups": {
    "numeric_cols": [
      "1stFlrSF",
      "2ndFlrSF",
      "3SsnPorch",
      "BedroomAbvGr",
      "BsmtFinSF1",
      "BsmtFinSF2",
      "BsmtFullBath",
      "BsmtHalfBath",
      "BsmtUnfSF",
      "EnclosedPorch",
      "Fireplaces",
      "FullBath",
      "GarageArea",
      "GarageCars",
      "GarageYrBlt",
      "GrLivArea",
      "HalfBath",
      "KitchenAbvGr",
      "LotArea",
      "LotFrontage",
      "LowQualFinSF",
      "MSSubClass",
      "MasVnrArea",
      "MiscVal",
      "MoSold",
      "OpenPorchSF",
      "OverallCond",
      "OverallQual",
      "PoolArea",
      "ScreenPorch",
      "TotRmsAbvGrd",
      "TotalBsmtSF",
      "WoodDeckSF",
      "YearBuilt",
      "YearRemodAdd",
      "YrSold"
    ],
    "categorical_cols": [
      "Alley",
      "BldgType",
      "BsmtCond",
      "BsmtExposure",
      "BsmtFinType1",
      "BsmtFinType2",
      "BsmtQual",
      "CentralAir",
      "Condition1",
      "Condition2",
      "Electrical",
      "ExterCond",
      "ExterQual",
      "Exterior1st",
      "Exterior2nd",
      "Fence",
      "FireplaceQu",
      "Foundation",
      "Functional",
      "GarageCond",
      "GarageFinish",
      "GarageQual",
      "GarageType",
      "Heating",
      "HeatingQC",
      "HouseStyle",
      "KitchenQual",
      "LandContour",
      "LandSlope",
      "LotConfig",
      "LotShape",
      "MSZoning",
      "MasVnrType",
      "MiscFeature",
      "Neighborhood",
      "PavedDrive",
      "PoolQC",
      "RoofMatl",
      "RoofStyle",
      "SaleCondition",
      "SaleType",
      "Street",
      "Utilities"
    ]
  },
  "hgb_params": {
    "max_iter": 120,
    "learning_rate": 0.1254335218612733,
    "max_depth": null,
    "random_state": 42,
    "min_samples_leaf": 11,
    "max_leaf_nodes": 23,
    "early_stopping": true,
    "validation_fraction": 0.1
  },
  "cv_summary": {
    "main_metric": "rmse",
    "main_mean": 28566.109638302063,
    "main_std": 4796.254567862704,
    "secondary_metric": "mae",
    "secondary_mean": 17169.06673586387,
    "secondary_std": 1310.5576533390956,
    "n_splits": 5
  },
  "artifacts": {
    "model_joblib": "models/model.joblib",
    "metrics_table": "reports/metrics_models.csv"
  }
}
```

