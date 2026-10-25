"""
03_modeling_linear.py

Linear regression on the feature-selection teammate's chosen predictors
(feature_selection/selected_features.csv).

Reads:
    data/processed/03_X_train.csv, 03_y_train.csv
    data/processed/03_X_test.csv,  03_y_test.csv
    feature_selection/selected_features.csv

Does:
    1. Subsets X_train/X_test to the 29 selected columns, then drops
       country_Unknown as the reference level. All 12 country_* columns
       always sum to 1 per row, so one of them is fully predictable from
       the other 11 (the "dummy-variable trap") - keeping all 12 makes the
       individual country coefficients meaningless. Dropping one fixes it;
       the remaining country coefficients are then read as "difference from
       an unknown-country respondent".
    2. 5-fold cross-validation on TRAIN ONLY (R2, RMSE, MAE). This is a
       practice-exam check used to judge the modelling choice - the test
       set is not touched here.
    3. Fits sklearn LinearRegression on the full training set.
    4. Fits statsmodels OLS on the same data for p-values / 95% confidence
       intervals (inferential stats for the slide).
    5. Scores the final model on TEST exactly once (R2, RMSE, MAE) and
       compares it to the baseline of always predicting the training mean
       (README: this baseline gives R2 = 0 by definition).
    6. Saves a coefficients table, a metrics summary, and two diagnostic
       plots (predicted vs. actual, residuals vs. predicted).

Reflection questions to answer on the slide (not computed here):
    - What problem did this step solve?
    - What other approaches were considered, and why this one?
    - What would have gone wrong if this step were skipped?
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
FEATURE_DIR = ROOT / "feature_selection"
FIGURES = ROOT / "figures"
RANDOM_STATE = 42
REFERENCE_COUNTRY = "country_Unknown"  # dropped to avoid the dummy-variable trap

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
X_train_full = pd.read_csv(PROCESSED / "03_X_train.csv")
y_train = pd.read_csv(PROCESSED / "03_y_train.csv").iloc[:, 0]
X_test_full = pd.read_csv(PROCESSED / "03_X_test.csv")
y_test = pd.read_csv(PROCESSED / "03_y_test.csv").iloc[:, 0]

selected = pd.read_csv(FEATURE_DIR / "selected_features.csv")["selected_feature"].tolist()

missing_in_train = [c for c in selected if c not in X_train_full.columns]
if missing_in_train:
    raise ValueError(f"Selected features missing from X_train: {missing_in_train}")

if REFERENCE_COUNTRY not in selected:
    raise ValueError(f"{REFERENCE_COUNTRY} not found in selected_features.csv - check the reference column name")

model_features = [c for c in selected if c != REFERENCE_COUNTRY]

X_train = X_train_full[model_features].copy()
X_test = X_test_full[model_features].copy()

print(f"Using {len(model_features)} features "
      f"({len(selected)} selected - 1 reference country column '{REFERENCE_COUNTRY}' dropped).\n")

# ---------------------------------------------------------------------------
# 2. 5-fold cross-validation on TRAIN ONLY
# ---------------------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
lr_cv = LinearRegression()

cv_r2 = cross_val_score(lr_cv, X_train, y_train, cv=kf, scoring="r2")
cv_rmse = -cross_val_score(lr_cv, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
cv_mae = -cross_val_score(lr_cv, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")

print("5-fold CV on train (practice exams - test set untouched):")
print(f"  R2:   {cv_r2.mean():.3f}  (folds: {np.round(cv_r2, 3)})")
print(f"  RMSE: {cv_rmse.mean():.3f}  (folds: {np.round(cv_rmse, 3)})")
print(f"  MAE:  {cv_mae.mean():.3f}  (folds: {np.round(cv_mae, 3)})")

# ---------------------------------------------------------------------------
# 3. Fit the final model on the FULL training set
# ---------------------------------------------------------------------------
final_model = LinearRegression()
final_model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 4. statsmodels OLS on the same data - p-values / confidence intervals
# ---------------------------------------------------------------------------
X_train_sm = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_sm).fit()

coef_table = pd.DataFrame({
    "feature": ["const"] + model_features,
    "coefficient": ols_model.params.values,
    "std_err": ols_model.bse.values,
    "p_value": ols_model.pvalues.values,
    "ci_lower_95": ols_model.conf_int()[0].values,
    "ci_upper_95": ols_model.conf_int()[1].values,
})
coef_table["significant_at_0.05"] = coef_table["p_value"] < 0.05

# ---------------------------------------------------------------------------
# 5. Score on TEST exactly once
# ---------------------------------------------------------------------------
y_pred_test = final_model.predict(X_test)

test_r2 = r2_score(y_test, y_pred_test)
test_rmse = mean_squared_error(y_test, y_pred_test) ** 0.5
test_mae = mean_absolute_error(y_test, y_pred_test)

baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
baseline_r2 = r2_score(y_test, baseline_pred)
baseline_rmse = mean_squared_error(y_test, baseline_pred) ** 0.5
baseline_mae = mean_absolute_error(y_test, baseline_pred)

print("\nFinal evaluation on TEST (done exactly once):")
print(f"  Linear regression -> R2: {test_r2:.3f}, RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}")
print(f"  Mean baseline      -> R2: {baseline_r2:.3f}, RMSE: {baseline_rmse:.3f}, MAE: {baseline_mae:.3f}")

# ---------------------------------------------------------------------------
# 6. Save outputs
# ---------------------------------------------------------------------------
PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

coef_table.to_csv(PROCESSED / "04_linear_regression_coefficients.csv", index=False)

metrics = pd.DataFrame([
    {"stage": "5-fold CV (train)", "model": "LinearRegression", "r2": cv_r2.mean(), "rmse": cv_rmse.mean(), "mae": cv_mae.mean()},
    {"stage": "Final test (once)", "model": "LinearRegression", "r2": test_r2, "rmse": test_rmse, "mae": test_mae},
    {"stage": "Final test (once)", "model": "Mean baseline", "r2": baseline_r2, "rmse": baseline_rmse, "mae": baseline_mae},
])
metrics.to_csv(PROCESSED / "04_linear_regression_metrics.csv", index=False)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test, alpha=0.15, s=10)
lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
plt.plot(lims, lims, "r--", linewidth=1)
plt.xlabel("Actual AI_Sentiment_Score (test)")
plt.ylabel("Predicted AI_Sentiment_Score (test)")
plt.title("Linear regression: predicted vs. actual (test set)")
plt.tight_layout()
plt.savefig(FIGURES / "04_linear_regression_pred_vs_actual.png", dpi=150)
plt.close()

residuals = y_test - y_pred_test
plt.figure(figsize=(6, 4))
plt.scatter(y_pred_test, residuals, alpha=0.15, s=10)
plt.axhline(0, color="r", linestyle="--", linewidth=1)
plt.xlabel("Predicted AI_Sentiment_Score (test)")
plt.ylabel("Residual (actual - predicted)")
plt.title("Linear regression: residuals vs. predicted (test set)")
plt.tight_layout()
plt.savefig(FIGURES / "04_linear_regression_residuals.png", dpi=150)
plt.close()

print("\nSaved:")
print(f"  {PROCESSED / '04_linear_regression_coefficients.csv'}")
print(f"  {PROCESSED / '04_linear_regression_metrics.csv'}")
print(f"  {FIGURES / '04_linear_regression_pred_vs_actual.png'}")
print(f"  {FIGURES / '04_linear_regression_residuals.png'}")

print("\nFull statsmodels OLS summary:\n")
print(ols_model.summary())
