
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