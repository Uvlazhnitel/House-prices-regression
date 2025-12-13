
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

## Learning curve (leader model)

Model: **HistGradientBoostingRegressor (tuned), target = log1p(SalePrice)**  
Figure: `reports/figures/learning_curve.png`  

Learning curve data (5-fold CV, RMSE on original target scale):

| Train size | Train RMSE (mean) | Validation RMSE (mean) |
|-----------:|------------------:|------------------------:|
|        93  | 22,493.18         | 40,735.50               |
|       261  | 13,682.07         | 33,599.04               |
|       429  | 14,130.36         | 31,433.37               |
|       597  | 11,966.90         | 30,543.44               |
|       765  | 14,323.08         | 30,400.23               |
|       934  | 12,781.68         | 27,998.61               |

Observations:

- For small training sizes (93–261 samples), the model achieves very low training error (≈22.5k → 13.7k RMSE) but much higher validation error (≈40.7k → 33.6k RMSE), indicating strong overfitting on small data.
- As the training size increases from 429 to 934 samples, validation RMSE steadily improves from ≈31.4k down to ≈28.0k, while training RMSE remains in the ≈12–14k range.
- At the largest training size (934 samples), there is still a large gap between train and validation errors (train ≈12.8k vs. val ≈28.0k), which points to substantial overfitting (high variance).
- The validation curve is still slowly decreasing with more data, so additional data would likely help, but the marginal gains are already diminishing.

Conclusion:

- The current leader model is **overfitting**: it fits the training data very well but generalizes noticeably worse on validation folds.  
- Further improvements should focus on **stronger regularization** (e.g., limiting tree depth, increasing `min_samples_leaf`, adding `l2_regularization`) and/or **better feature engineering**, rather than increasing model complexity.

## Model interpretation (PDP & permutation importance)

### 1. Permutation importance

I computed permutation importance for the tuned leader model on the validation data.  
The numbers below show the **increase in RMSE (in the original target units)** when each feature is randomly permuted, averaged over several repetitions (higher = more important):

Top features by permutation importance

- **OverallQual**: +34,021 (±1,079)
- **GrLivArea**: +25,896 (±693)
- **GarageCars**: +7,199 (±380)
- **BsmtFinSF1**: +6,868 (±329)
- **TotalBsmtSF**: +6,614 (±420)
- **Neighborhood**: +5,890 (±887)
- **LotArea**: +4,383 (±256)
- **1stFlrSF**: +3,770 (±194)
- **YearBuilt**: +3,468 (±187)
- **YearRemodAdd**: +2,789 (±255)

Interpretation:

- **OverallQual** and **GrLivArea** are by far the most critical predictors: shuffling either of them increases RMSE by ~26k–34k, which is a very large degradation compared to other features.
- **GarageCars**, basement-related square footage, and location (**Neighborhood**) form the second tier of important predictors.
- Many remaining features have relatively small importance values; permuting them only slightly worsens RMSE, which suggests that the model can compensate for them using other correlated variables.

These results are consistent with domain intuition: overall quality, living area, parking capacity, basement size and neighborhood should strongly affect house prices.

---

### 2. PDP — OverallQual

File: `reports/figures/pdp_OverallQual.png`

The PDP for **OverallQual** (overall material and finish quality, 1–10) shows a clear, strongly **increasing** relationship:

- For low quality levels (≈1–3) the predicted price is in a relatively low band and changes very little.
- From **3 to 6** the predicted price starts to increase more noticeably.
- There is a **sharp jump** in predicted price between **6 and 8**: moving from “average” to “high” quality has a large positive effect.
- Between **8 and 10** the curve continues to increase but flattens slightly, indicating **diminishing returns** at the highest quality levels.

Interpretation:

> The model considers overall quality to be a major driver of price.  
> Upgrading a house from low/average quality to high quality has a strong positive impact on the predicted sale price, while further improvements at the very top end still help but with smaller marginal gains.

This matches expectations: going from poor to good quality matters more than fine-tuning already premium houses.

---

### 3. PDP — GrLivArea

File: `reports/figures/pdp_GrLivArea.png`

The PDP for **GrLivArea** (above-ground living area) shows an almost **monotonic increasing** relationship:

- In the lower range (≈850–1,300 sq ft) the predicted price grows steadily as living area increases.
- Around **1,400–1,600 sq ft** there is a steeper region, where additional square footage gives a more visible jump in predicted price.
- From roughly **1,600 up to 2,400+ sq ft**, the curve keeps increasing but the slope is a bit smoother — the model still rewards extra space, but each additional square foot contributes slightly less than in the mid-range.

Interpretation:

> The model sees “more living area” as consistently increasing the predicted price, with some indication of **diminishing returns** at larger sizes.  
> Increasing a very small house to a medium-sized one has a stronger effect than adding the same amount of space to an already large house.

Again this is intuitive: going from cramped to comfortable has bigger impact than expanding an already spacious house.

---

### 4. PDP — GarageCars

File: `reports/figures/pdp_GarageCars.png`

The PDP for **GarageCars** (number of cars the garage can hold) has a **stepwise** shape:

- Moving from **0 → 1 → 2 cars** increases the predicted price gradually.
- The jump from **2 to 3 cars** adds another clear increase.
- Between **3 and 4 cars** the curve is almost flat, suggesting that beyond three parking spots, the model does **not** expect a substantial additional gain in price.

Interpretation:

> The model treats having a garage, and especially moving from 1–2 to **3 car spaces**, as valuable.  
> However, once a house already has a 3-car garage, adding even more capacity yields little additional benefit in the prediction.

This is a reasonable pattern: going from no garage to a 2–3 car garage is a strong positive signal; more than that is niche and less impactful.

---

### 5. Overall conclusion

- Permutation importance and PDPs agree that **overall quality**, **living area**, and **garage capacity** are key drivers for the model’s price predictions.
- For all three features, the model captures **non-linear effects and diminishing returns**:
  - quality and area strongly increase price up to a point, then the effect saturates;
  - garage capacity is most beneficial up to 2–3 spaces, with little gain beyond that.
- The learned relationships are consistent with domain knowledge, which increases confidence that the model is not relying on spurious artifacts but on meaningful housing characteristics.

## Error analysis & feature hypotheses

### 1. Experimental setup

* Model: current **leader pipeline** (`log_target_model`) with preprocessing and regressor, trained on the fixed train split.
* To get honest error estimates on the training data, I computed **OOF (out-of-fold) predictions** using `cross_val_predict`:

  * same `cv` object as in the main evaluation (same folds, same `random_state`),
  * for each training sample, the prediction comes from a model that **did not see this sample** during training.
* Based on OOF predictions I computed:

  * `residual = y_true − y_pred`,
  * `abs_error = |y_true − y_pred|`,
  * `relative_error = residual / y_true`.
* Saved artifacts:

  * per-sample error metrics only: `reports/error_analysis_df.csv`,
  * full table with features + errors (via concatenation with `X_train`): `reports/error_analysis_with_features.csv`.

---

### 2. Summary statistics of errors

On the training set (1168 samples):

**Target & predictions:**

* mean `y_true` ≈ **181,441**, mean `y_pred` ≈ **179,561** → global level of predictions is very close to the true average price.
* median `y_true` = **165,000**, median `y_pred` ≈ **164,340** → medians are also very close.

**Residuals (`residual = y_true − y_pred`):**

* mean residual ≈ **+1,880** → on average, the model **slightly underestimates** prices (true price is ~1.9k higher than predicted).
* std residual ≈ **28,082**.
* quantiles:

  * 25% ≈ **−10,258**
  * 50% ≈ **+649**
  * 75% ≈ **+12,237**
    → for **50% of houses**, residual lies roughly in **[−10k; +12k]**.

**Absolute error (`abs_error`):**

* mean abs_error ≈ **17,096**,
* median abs_error ≈ **10,964**,
* 75% quantile ≈ **20,852**.

Given the median price ≈ 165k:

* for **50% of houses**, absolute error ≈ **≤ 11k** (~6–7% of median price),
* for **75% of houses**, absolute error ≈ **≤ 21k**.

**Tails:**

* min residual ≈ **−215,733**, max residual ≈ **+284,694**,
* max abs_error ≈ **284,694**.

→ There is a **small number of extreme outliers** with errors of order 200–280k, which heavily influence RMSE.

**Conclusion:**
For the majority of samples the model error is reasonable (tens of thousands), with almost no global bias. RMSE is strongly driven by a few problematic houses with very large errors.

---

### 3. Residual distribution (histogram)

* The residual histogram is **centered near zero**:

  * main mass around 0,
  * approximate symmetry.
* Most residuals lie in a relatively narrow band near 0; only a few points form long tails towards large positive/negative values (|residual| > 100k).

**Conclusion:**
The model does **not** exhibit strong systematic bias in one direction. Most predictions are close to the true values; a few extreme outliers create long tails in the residual distribution.

---

### 4. Residuals vs predictions / true values

#### 4.1. `y_true vs residual`

* For true prices up to ~**300k**, residuals are scattered symmetrically around zero:

  * both under- and over-estimations present,
  * no clear trend, moderate spread.
* For **expensive houses** (true price ≳ ~350k):

  * many residuals are **positive** (`y_true > y_pred`) → strong **underestimation**,
  * error magnitude grows significantly.

**Conclusion:**
In the **main price segment** (~70k–300k) the model is roughly unbiased. In the **upper price segment**, the model tends to **underestimate** expensive houses and makes much larger errors there.

#### 4.2. `y_pred vs residual`

* For predicted prices up to ~**300k**:

  * cloud of points around residual = 0,
  * no obvious trend, residuals look like noise.
* For higher predicted prices (~**> 300–350k**):

  * vertical spread of residuals increases strongly,
  * residuals reach up to ±200–300k.

**Conclusion:**
For “normal” predictions (≤ 300k) the model behaves stably and without strong systematic drift. In the **high predicted price segment**, error variance grows markedly (**heteroscedasticity**): the model is much less reliable there and can both heavily over- and under-estimate individual houses.

---

### 5. Top-N errors (sorted by `abs_error`)

I concatenated `X_train` with error metrics to obtain `analysis_df`, then sorted rows by `abs_error` in descending order and examined the **top 10** rows. For each, I inspected:

* `y_true`, `y_pred`, `abs_error`, `residual`,
* key features: `GrLivArea`, `OverallQual`, `GarageCars`, `BsmtFinSF1`, `TotalBsmtSF`,
* and context features: `Neighborhood`, `LotArea`, `YearBuilt`.

**Observations:**

1. The top-10 most erroneous cases include both **very expensive** and **relatively cheap** houses:

   * expensive, strongly **underestimated**:

     * ~392k, 375k, 466.5k, 538k, 582.9k, 745k,
   * cheaper, strongly **overestimated**:

     * ~147k, 160k, 184.8k, 235k.

2. **Overestimated cheap houses**:

   * have **large living areas** (e.g. 1262–2198 sqft),
   * **above-average quality** (OverallQual 6–8),
   * normal or large garages and basements.
   * Example: a house at 147k with GrLivArea≈2198 and OverallQual=8 is predicted around 312k.
     → According to structural features, these homes look like “expensive” ones, but the transaction price is much lower. The model **overprices** such houses.

3. **Underestimated expensive houses**:

   * do **not** always have extreme size or quality:

     * GrLivArea often ~1362–2090 (even 1077 in one case),
     * OverallQual ranges between 5 and 7.
   * Example: a 745k house in `CollgCr` (1710 sqft, OverallQual=7, built in 2003) is predicted ~460k.
     → Based on basic numerical features, these homes look more like “upper-middle” segment, but the actual price is much higher. The model **underestimates** them.

4. Neighborhoods in top-10 are varied:

   * `CollgCr`, `Veenker`, `NoRidge`, `Somerst`, `Crawfor`, `NWAmes`, `OldTown`, `BrkSide`, `Mitchel`, etc.
   * There is no single dominating neighborhood in this small sample, but several **typically more expensive** neighborhoods appear (`NoRidge`, `Somerst`, `Crawfor`, some `CollgCr`/`Veenker` cases).

**Qualitative conclusion on top errors:**

> * The model tends to **pull prices towards a “middle/upper-middle” band**:
>
>   * truly expensive houses with moderate size/quality are **underestimated** (predictions too low),
>   * relatively cheap but large, high-quality houses are **overestimated** (predictions too high).
> * This suggests the model:
>
>   1. heavily relies on structural features (size, quality, garage, basement),
>   2. **does not fully account for neighborhood / location effects and other context factors**,
>      which leads to:
>
>      * price underestimation in premium segments,
>      * and overestimation of “suspiciously cheap” large/quality houses.

---

### 6. Initial feature hypotheses

Based on the error analysis above, I formulated several **feature engineering hypotheses** (to be tested later):

1. **Neighborhood “price level” features**

   * Observation: some high-error cases are in neighborhoods that are typically considered more expensive (`NoRidge`, `Somerst`, `Veenker`, `Crawfor`, etc.), where the model underestimates the price.
   * Hypothesis: the model does not explicitly know the **price class** of a neighborhood.
   * Potential features:

     * `neighborhood_price_rank`: numeric rank of each neighborhood based on median/mean SalePrice (e.g. 1 = cheap, 2 = mid, 3 = expensive).
     * `is_premium_neighborhood`: binary flag for neighborhoods in a top-N group by median SalePrice.
   * Expected effect: better handling of premium areas → reduced underestimation of expensive houses and reduced overestimation of cheap houses in high-end locations.

2. **Interaction between size and quality**

   * Observation: top errors include:

     * cheap but **large and high-quality** houses (overestimated),
     * expensive houses that are not extreme in either size or quality (underestimated).
   * Hypothesis: the price impact of living area depends on overall quality (and vice versa); very large + high quality houses form a special “premium” segment which is not well captured by simple additive effects.
   * Potential features:

     * Interaction feature `GrLivArea * OverallQual`.
     * Binary flag `is_big_and_high_quality` (e.g. living area above some percentile AND OverallQual ≥ 7).
   * Expected effect: allow the model to better differentiate the “big & high-quality” segment and adjust predictions accordingly.

3. **Age / year-built grouping**

   * Observation: some top-error houses are very old but expensive (e.g. `Crawfor` 1915, `BrkSide` 1939), while others in newer neighborhoods also behave unusually.
   * Hypothesis: the model does not sufficiently distinguish age groups (old vs mid vs new stock), especially in combination with neighborhood.
   * Potential features:

     * `house_age` = (year_of_sale − `YearBuilt`).
     * Categorical `age_group` (e.g. old / mid / new) based on `YearBuilt`.
   * Goal: enable different pricing behavior for old houses in cheap vs premium neighborhoods.

## Feature engineering experiments and results

This session focused on error analysis and testing supervised feature engineering ideas on top of the current best model (tuned `HistGradientBoostingRegressor` with log-target setup).

Baseline CV performance (5-fold, RMSE on original target scale):

- **Baseline HGB model (no extra supervised features):**  
  `RMSE = 27,743.10 ± 4,616.72`

Below are the two main hypotheses I tested and their results.

---

### Hypothesis 1 — Neighborhood “price level” features

**Idea**

From the error analysis, several large errors occurred in neighborhoods that are typically perceived as more expensive (e.g. `NoRidge`, `Somerst`, `Veenker`, `Crawfor`, etc.). The intuition is that the model may not explicitly understand the **price level** of each neighborhood and thus:

- underestimates some expensive houses in premium areas,
- overestimates some relatively cheap houses that “look good” structurally but are located in less desirable areas.

To address this, I implemented a supervised transformer `NeighborhoodPriceLevelAdder` that:

- computes the **median SalePrice per Neighborhood** on the training fold,
- bins neighborhoods into `n_bins = 3` groups using `qcut`:
  - 0 = cheaper neighborhoods,
  - 1 = mid-range,
  - 2 = most expensive,
- adds two new features:
  - `neighborhood_price_rank` (0, 1, 2),
  - `is_premium_neighborhood` (1 if in the highest-price bin, else 0).

These features were appended to the numeric columns and passed through the standard numeric preprocessing (median imputation + scaling) together with the tuned HGB model.

**CV comparison (5-fold, RMSE):**

- Baseline HGB:  
  `RMSE = 27,743.10 ± 4,616.72`
- HGB + `neighborhood_price_rank` + `is_premium_neighborhood`:  
  `RMSE = 27,924.55 ± 4,765.20`

The mean RMSE slightly increased by ~181, which is very small relative to the fold-to-fold standard deviation (~4.7k). There is no consistent or meaningful improvement.

**Conclusion**

The neighborhood price-level features **do not improve** the model in this form. The current model already encodes `Neighborhood` via categorical encoding, and the coarse 3-bin aggregation does not add useful signal.  
**Hypothesis 1 is not supported**, and I keep the baseline HGB model (without these extra features) as the main version.

---

### Hypothesis 2 — Size–quality interaction features

**Idea**

From the top-error cases:

- some **cheap houses** are **large and high-quality** (big `GrLivArea`, high `OverallQual`) but sold relatively cheaply → the model overestimates them;
- some **expensive houses** are not extreme in either size or quality but still very expensive → the model underestimates them.

This suggests that the interaction between **size and quality** might be important:  
“big + high quality” houses may behave like a separate premium segment.

To capture this, I implemented a transformer `SizeQualityInteractionAdder` that adds:

- `area_quality_interaction = GrLivArea * OverallQual`,
- `is_big_high_quality`:
  - computes a `big_area_threshold` as the 80th percentile of `GrLivArea` on the training fold,
  - sets `is_big_high_quality = 1` if `GrLivArea >= big_area_threshold` **and** `OverallQual ≥ 7`, else 0.

Again, these features were appended to the numeric columns and passed through the same preprocessing and tuned HGB model.

**CV comparison (5-fold, RMSE):**

- Baseline HGB:  
  `RMSE = 27,743.10 ± 4,616.72`
- HGB + `area_quality_interaction` + `is_big_high_quality`:  
  `RMSE = 28,994.69 ± 5,107.70`

The mean RMSE increased by ~1,250, and the standard deviation also grew. This is a clear degradation, not explainable by random CV noise.

**Conclusion**

The size–quality interaction features **hurt** performance in this setup.  
The `HistGradientBoostingRegressor` already captures nonlinear interactions between `GrLivArea` and `OverallQual` via tree splits, so manually adding these interaction features is redundant or noisy for this model.  
**Hypothesis 2 is rejected**, and the baseline HGB model without these additional features remains the preferred version.

---

### Overall conclusion

- I performed a detailed error analysis (residual distribution, residuals vs predictions, top-N error cases) and formulated multiple hypotheses about neighborhood and size–quality effects.
- I implemented and tested two concrete feature-engineering hypotheses inside a proper `Pipeline` with cross-validation.
- Neither hypothesis led to a consistent improvement in CV RMSE; in fact, Hypothesis 2 significantly worsened performance.

Therefore, I keep the **baseline tuned HGB model without these extra supervised features** as the final candidate model for this project, and I treat these experiments as documented negative results in the feature engineering phase.
## Train run: HGB_tuned
- UTC time: 2025-12-13T13:47:00Z
- Target: `SalePrice` (transform: `log1p`)
- CV: KFold(n_splits=5, shuffle=True, random_state=42)
- Main metric: **rmse** = 27737.518431 ± 4621.740442
- Secondary metric: **mae** = 17100.219495 ± 1384.889558
- Saved model: `models/model.joblib`
- Saved meta: `models/model_meta.json`

## Train run: HGB_tuned
- UTC time: 2025-12-13T13:48:23Z
- Target: `SalePrice` (transform: `log1p`)
- CV: KFold(n_splits=5, shuffle=True, random_state=42)
- Main metric: **rmse** = 27737.518431 ± 4621.740442
- Secondary metric: **mae** = 17100.219495 ± 1384.889558
- Saved model: `models/model.joblib`
- Saved meta: `models/model_meta.json`

## Train run: HGB_tuned
- UTC time: 2025-12-13T13:55:02Z
- Target: `SalePrice` (transform: `log1p`)
- CV: KFold(n_splits=5, shuffle=True, random_state=42)
- Main metric: **rmse** = 27737.518431 ± 4621.740442
- Secondary metric: **mae** = 17100.219495 ± 1384.889558
- Saved model: `models/model.joblib`
- Saved meta: `models/model_meta.json`

