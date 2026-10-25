"""
03_modeling_forest.py

Random forest regression on the same predictors used for linear regression
(feature_selection/selected_features.csv, minus the country_Unknown
reference column - kept identical to 03_modeling_linear.py so the two
models are compared on exactly the same inputs).

Reads:
    data/processed/03_X_train.csv, 03_y_train.csv
    data/processed/03_X_test.csv,  03_y_test.csv
    feature_selection/selected_features.csv

Does:
    1. Subsets X_train/X_test to the same 28 columns used for linear
       regression. Random forests don't suffer from the dummy-variable
       trap (they split on one column at a time, not solve an equation),
       so dropping country_Unknown isn't strictly required here - it's
       kept only so the two models are compared on identical inputs, not
       for any random-forest-specific reason.
    2. 5-fold cross-validation on TRAIN ONLY (R2, RMSE, MAE) - same
       practice-exam idea as the linear regression script, test set never
       touched here.
    3. Fits the final RandomForestRegressor on the full training set.
    4. Scores it on TEST exactly once and compares it to both the mean
       baseline and the linear regression result (read back from
       04_linear_regression_metrics.csv if present).
    5. Saves a feature-importance table, a metrics summary, and the same
       two diagnostic plots as the linear regression script, plus a
       feature-importance bar chart.

Reflection questions to answer on the slide (not computed here):
    - What problem did this step solve? (a non-linear cross-check on the
      linear regression - trees can pick up curves/interactions a straight
      line can't.)
    - What other approaches were considered? (gradient boosting, tuning
      depth/leaf-size via grid search - skipped here to keep this a fair,
      simple cross-check rather than a second optimisation project.)
    - What would have gone wrong if this step were skipped? We would not
      know whether the modest R2 from linear regression is a limit of the
      *data* or a limit of assuming straight-line relationships.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
FEATURE_DIR = ROOT / "feature_selection"
FIGURES = ROOT / "figures"
RANDOM_STATE = 42
REFERENCE_COUNTRY = "country_Unknown"  # dropped only to match the linear regression's inputs

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
X_train_full = pd.read_csv(PROCESSED / "03_X_train.csv")
y_train = pd.read_csv(PROCESSED / "03_y_train.csv").iloc[:, 0]
X_test_full = pd.read_csv(PROCESSED / "03_X_test.csv")
y_test = pd.read_csv(PROCESSED / "03_y_test.csv").iloc[:, 0]

selected = pd.read_csv(FEATURE_DIR / "selected_features.csv")["selected_feature"].tolist()
model_features = [c for c in selected if c != REFERENCE_COUNTRY]

X_train = X_train_full[model_features].copy()
X_test = X_test_full[model_features].copy()

print(f"Using {len(model_features)} features (same set as 03_modeling_linear.py).\n")

# ---------------------------------------------------------------------------
# 2. 5-fold cross-validation on TRAIN ONLY
# ---------------------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rf_cv = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)

cv_r2 = cross_val_score(rf_cv, X_train, y_train, cv=kf, scoring="r2")
cv_rmse = -cross_val_score(rf_cv, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")
cv_mae = -cross_val_score(rf_cv, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")

print("5-fold CV on train (practice exams - test set untouched):")
print(f"  R2:   {cv_r2.mean():.3f}  (folds: {np.round(cv_r2, 3)})")
print(f"  RMSE: {cv_rmse.mean():.3f}  (folds: {np.round(cv_rmse, 3)})")
print(f"  MAE:  {cv_mae.mean():.3f}  (folds: {np.round(cv_mae, 3)})")

# ---------------------------------------------------------------------------
# 3. Fit the final model on the FULL training set
# ---------------------------------------------------------------------------
final_model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
final_model.fit(X_train, y_train)

importance_table = pd.DataFrame({
    "feature": model_features,
    "importance": final_model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 4. Score on TEST exactly once
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
print(f"  Random forest -> R2: {test_r2:.3f}, RMSE: {test_rmse:.3f}, MAE: {test_mae:.3f}")
print(f"  Mean baseline -> R2: {baseline_r2:.3f}, RMSE: {baseline_rmse:.3f}, MAE: {baseline_mae:.3f}")

lr_metrics_path = PROCESSED / "04_linear_regression_metrics.csv"
if lr_metrics_path.exists():
    lr_metrics = pd.read_csv(lr_metrics_path)
    lr_test_row = lr_metrics[(lr_metrics["stage"] == "Final test (once)") & (lr_metrics["model"] == "LinearRegression")]
    if not lr_test_row.empty:
        row = lr_test_row.iloc[0]
        print(f"  Linear regression (for comparison) -> R2: {row['r2']:.3f}, RMSE: {row['rmse']:.3f}, MAE: {row['mae']:.3f}")

# ---------------------------------------------------------------------------
# 5. Save outputs
# ---------------------------------------------------------------------------
PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

importance_table.to_csv(PROCESSED / "05_random_forest_feature_importance.csv", index=False)

metrics = pd.DataFrame([
    {"stage": "5-fold CV (train)", "model": "RandomForest", "r2": cv_r2.mean(), "rmse": cv_rmse.mean(), "mae": cv_mae.mean()},
    {"stage": "Final test (once)", "model": "RandomForest", "r2": test_r2, "rmse": test_rmse, "mae": test_mae},
    {"stage": "Final test (once)", "model": "Mean baseline", "r2": baseline_r2, "rmse": baseline_rmse, "mae": baseline_mae},
])
metrics.to_csv(PROCESSED / "05_random_forest_metrics.csv", index=False)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test, alpha=0.15, s=10)
lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
plt.plot(lims, lims, "r--", linewidth=1)
plt.xlabel("Actual AI_Sentiment_Score (test)")
plt.ylabel("Predicted AI_Sentiment_Score (test)")
plt.title("Random forest: predicted vs. actual (test set)")
plt.tight_layout()
plt.savefig(FIGURES / "05_random_forest_pred_vs_actual.png", dpi=150)
plt.close()

residuals = y_test - y_pred_test
plt.figure(figsize=(6, 4))
plt.scatter(y_pred_test, residuals, alpha=0.15, s=10)
plt.axhline(0, color="r", linestyle="--", linewidth=1)
plt.xlabel("Predicted AI_Sentiment_Score (test)")
plt.ylabel("Residual (actual - predicted)")
plt.title("Random forest: residuals vs. predicted (test set)")
plt.tight_layout()
plt.savefig(FIGURES / "05_random_forest_residuals.png", dpi=150)
plt.close()

top_n = importance_table.head(15)
plt.figure(figsize=(7, 6))
plt.barh(top_n["feature"][::-1], top_n["importance"][::-1])
plt.xlabel("Importance")
plt.title("Random forest: top 15 feature importances")
plt.tight_layout()
plt.savefig(FIGURES / "05_random_forest_feature_importance.png", dpi=150)
plt.close()

print("\nSaved:")
print(f"  {PROCESSED / '05_random_forest_feature_importance.csv'}")
print(f"  {PROCESSED / '05_random_forest_metrics.csv'}")
print(f"  {FIGURES / '05_random_forest_pred_vs_actual.png'}")
print(f"  {FIGURES / '05_random_forest_residuals.png'}")
print(f"  {FIGURES / '05_random_forest_feature_importance.png'}")

print("\nTop 10 feature importances:")
print(importance_table.head(10).to_string(index=False))
