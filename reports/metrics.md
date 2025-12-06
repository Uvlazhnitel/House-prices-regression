
## Baseline Metrics

### Main Metric: **RMSE** (Root Mean Squared Error)

#### Reasons for Choosing RMSE:
- RMSE is measured in the same units as the target variable (monetary units).
- Larger errors (especially for expensive items) are penalized more heavily, which is important for the business.

### Model Comparison (5-fold CV, train set):
- Metrics are reported as mean values over 5-fold cross-validation on the training set.

- **DummyRegressor (mean)**:
  - RMSE: 77077.66
  - MAE: 56319.67
- **LinearRegression**:
  - RMSE: 38539.27
  - MAE: 19757.67
- **Ridge (alpha=1.0)**:
  - RMSE: 35665.77
  - MAE: 19296.88
- **Lasso (alpha=0.1)**:
  - RMSE: 36470.49
  - MAE: 19316.63

### Conclusion:
- All linear models significantly outperform the dummy model in terms of RMSE and MAE.
- Ridge is tested with alpha=1.0 and Lasso with alpha=0.1; further tuning of the `alpha` parameter will be performed in a separate session.
- Among the tested models, **Ridge (alpha=1.0)** shows the best performance with the lowest RMSE and MAE, making it the most promising candidate for further optimization.

## Ridge & Lasso regularization

Main metric: RMSE (neg_root_mean_squared_error in CV).

Ridge:
- Alpha values tested: [0.001, 0.01, 0.1, 1, 10, 100, 1000]
- Best alpha: 10.0
- CV RMSE (mean ± std): 31556 ± 7217

Lasso:
- Alpha values tested: [0.001, 0.01, 0.1, 1, 10, 100, 1000]
- Best alpha: 100.0
- CV RMSE (mean ± std): 31597 ± 7633

Conclusion:
- Regularization significantly improved RMSE compared to very small alpha values (~36k → ~31.6k).
- Best Ridge and best Lasso perform almost identically on CV.
- For the next comparisons, we will use Ridge(alpha=10.0) and Lasso(alpha=100.0) as tuned linear baselines.

## RandomForestRegressor — feature importances (top 10)

Top features by importance:

1. OverallQual (0.55) — overall material and finish quality.
2. GrLivArea (0.12) — above ground living area.
3. TotalBsmtSF (0.03) — total basement area.
4. 2ndFlrSF (0.03) — second floor area.
5. BsmtFinSF1 (0.028) — finished basement area.
6. 1stFlrSF (0.027) — first floor area.
7. LotArea (0.018) — lot size.
8. GarageArea (0.016) — garage area.
9. GarageCars (0.014) — car capacity of the garage.
10. YearBuilt (0.012) — year the house was built.

Interpretation:

- OverallQual is by far the most important feature: model heavily relies on the overall quality rating to explain house prices.
- Different types of area-related features (living area, basement, floors, lot, garage) collectively form the second most important group of predictors.
- YearBuilt also contributes: newer houses tend to be more expensive on average.

This ranking is consistent with domain intuition: higher quality, more living space, larger lot/garage and newer construction all drive higher prices.

## Model comparison (cross-validation on train)

**Setup**

- Metric: RMSE (lower is better), secondary metric: MAE.
- Evaluation: 5-fold cross-validation on the training set using the same preprocessing and CV splits for all models.

**Results**

| Model                           | RMSE (mean ± std)      | MAE (mean ± std)       |
|---------------------------------|------------------------|------------------------|
| DummyRegressor (mean)           | 77,160 ± 3,673         | 56,318 ± 2,612         |
| LinearRegression                | 36,397 ± 7,545         | 19,622 ± 806           |
| Ridge (alpha = 1.0)             | 32,814 ± 7,333         | 18,676 ± 1,117         |
| Lasso (alpha = 0.1)             | 36,039 ± 7,401         | 19,445 ± 899           |
| Ridge (alpha = 10.0, tuned)     | 31,556 ± 7,217         | 18,001 ± 581           |
| Lasso (alpha = 100.0, tuned)    | 31,597 ± 7,633         | 20,613 ± 858           |
| RandomForestRegressor           | 30,188 ± 4,925         | 18,172 ± 1,124         |
| HistGradientBoostingRegressor   | **28,749 ± 4,849**     | **17,058 ± 995**       |

**Summary**

- The **DummyRegressor** baseline performs very poorly (RMSE ≈ 77k), as expected.
- Switching to **LinearRegression** dramatically improves performance (RMSE drops to ~36k).
- Adding **L2/L1 regularization** helps:
  - **Ridge (alpha = 10.0)** is the best linear model with RMSE ≈ 31.6k.
  - **Lasso (alpha = 100.0)** achieves a similar RMSE but worse MAE, so Ridge is preferred among linear models.
- **Tree-based models** further improve performance:
  - **RandomForestRegressor** reduces RMSE to ~30.2k.
  - **HistGradientBoostingRegressor** is currently the best model with RMSE ≈ 28.7k and MAE ≈ 17.1k.
- **Current leader:** HistGradientBoostingRegressor (baseline hyperparameters). This model will be treated as the main candidate for further tuning and analysis.

## Target transformation (raw vs log1p)

Model: HistGradientBoostingRegressor (baseline config)

CV setup: 5-fold KFold, shuffle=True, random_state=42  
Main metric: RMSE (neg_root_mean_squared_error in sklearn, converted to positive values)

Results (evaluated in original price space):

- HGB + raw target (SalePrice):
  - RMSE: 28748.83 ± 4848.51

- HGB + log1p target (y_log = log1p(SalePrice), predictions mapped back via expm1):
  - RMSE: 28161.12 ± 4554.00

Conclusion:
- log1p target reduced CV RMSE by ~587 (~2%) and slightly decreased the std across folds.
- Final choice: **use log1p(SalePrice) as the training target** for HGB (and future boosted models), with metrics always evaluated in the original price space.

## Hyperparameter tuning for HGB (log1p target)

**Model**

- `TransformedTargetRegressor(func=log1p, inverse_func=expm1)`
- Base regressor: `Pipeline(preprocessor → HistGradientBoostingRegressor)`

Target:

- The model is trained on `log1p(SalePrice)` and predictions are transformed back with `expm1`, so all metrics below are in the original price scale.

**Search setup**

- Method: `RandomizedSearchCV`
- Estimator: `log_target_model` (log1p wrapper around HGB pipeline)
- CV: 5-fold `KFold(shuffle=True, random_state=RANDOM_STATE)`
- Scoring:
  - primary: `neg_root_mean_squared_error` (used for `refit`)
  - secondary: `neg_mean_absolute_error`
- Number of sampled configurations: `n_iter = 40`
- Parameters tuned (HGB inside the pipeline):

  - `learning_rate`
  - `max_leaf_nodes`
  - `min_samples_leaf`
  - `max_iter`

**Best hyperparameters**

- `learning_rate = 0.1254335218612733`
- `max_leaf_nodes = 23`
- `min_samples_leaf = 11`
- `max_iter = 120`

**CV performance (5-fold, original price scale)**

- RMSE (mean ± std): **27 743 ± 4 617**
- MAE  (mean ± std): **17 094 ± 1 378**

**Conclusion**

- HistGradientBoostingRegressor with a log1p-transformed target and the hyperparameters above is selected as the tuned leader model for the regression task.
- This configuration will be used as the main model for further evaluation on the test set and for packaging (`predict.py`, model artifact in `models/`).
