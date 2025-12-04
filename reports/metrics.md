
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
