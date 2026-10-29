"""
03_modeling_ordinal.py

Ordinal regression (ordered logit) on the same 28 features used by
03_modeling_linear.py and 03_modeling_forest.py.

Why this model:
    AI_Sentiment_Score only takes 5 values (0, 25, 50, 75, 100), and they are
    ORDERED (very unfavourable -> very favourable). Linear regression treats
    the score as a smooth number, so it can predict impossible values like
    63.7 or 104. Ordinal regression is built for exactly this kind of target:
    it predicts the chance of each of the 5 levels while respecting their
    order. It was listed as a "next step" on the limitations slide.

How it works (plain words):
    The model computes one score per person from the features (like linear
    regression, but with no intercept), then uses 4 cut-points on that score
    to split people into the 5 levels. A higher score pushes a person past
    more cut-points, toward a more favourable level. Each coefficient is in
    log-odds; exp(coefficient) is the odds ratio = how much the odds of being
    in a HIGHER level multiply when that feature goes up by one.

Reads:
    data/processed/03_X_train.csv, 03_y_train.csv
    data/processed/03_X_test.csv,  03_y_test.csv
    feature_selection/selected_features.csv
    data/processed/04_linear_regression_metrics.csv   (optional, for comparison)
    data/processed/05_random_forest_metrics.csv       (optional, for comparison)

Does:
    1. Same 28 features as the other models (29 selected minus country_Unknown).
    2. 5-fold CV on TRAIN ONLY (same folds as the other scripts).
    3. Fits the ordered logit on the full training set (statsmodels OrderedModel).
    4. Scores TEST exactly once, two ways:
         - expected score  (sum of level x its probability) -> R2, RMSE, MAE,
           directly comparable with linear regression and random forest
         - most likely level (one of 0/25/50/75/100)        -> exact-level
           accuracy and "within one level" accuracy
       Linear regression's predictions are also rounded to the nearest level
       so its accuracy can be compared fairly.
    5. Saves coefficients (with odds ratios, p-values), metrics, a comparison
       table, and a confusion-matrix plot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from statsmodels.miscmodels.ordinal_model import OrderedModel

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
FEATURE_DIR = ROOT / "feature_selection"
FIGURES = ROOT / "figures"
RANDOM_STATE = 42
REFERENCE_COUNTRY = "country_Unknown"
LEVELS = np.array([0, 25, 50, 75, 100])

# ---------------------------------------------------------------------------
# 1. Load data (identical to the linear / forest scripts)
# ---------------------------------------------------------------------------
X_train_full = pd.read_csv(PROCESSED / "03_X_train.csv")
y_train = pd.read_csv(PROCESSED / "03_y_train.csv").iloc[:, 0]
X_test_full = pd.read_csv(PROCESSED / "03_X_test.csv")
y_test = pd.read_csv(PROCESSED / "03_y_test.csv").iloc[:, 0]

selected = pd.read_csv(FEATURE_DIR / "selected_features.csv")["selected_feature"].tolist()
model_features = [c for c in selected if c != REFERENCE_COUNTRY]
X_train = X_train_full[model_features].copy()
X_test = X_test_full[model_features].copy()

assert set(y_train.unique()) <= set(LEVELS), "target has values outside 0/25/50/75/100"
print(f"Using {len(model_features)} features (same as linear regression and random forest).\n")


def to_ordered(y):
    """0/25/50/75/100 as an ordered category, which OrderedModel needs."""
    return pd.Series(pd.Categorical(y, categories=LEVELS, ordered=True), index=y.index)


def fit_ordinal(X, y):
    model = OrderedModel(to_ordered(y), X, distr="logit")
    return model.fit(method="bfgs", maxiter=2000, disp=False)


def predict_both(result, X):
    """Return (expected score, most likely level) for each row."""
    probs = np.asarray(result.model.predict(result.params, exog=X))
    expected = probs @ LEVELS
    most_likely = LEVELS[probs.argmax(axis=1)]
    return expected, most_likely


def round_to_level(pred):
    return LEVELS[np.abs(np.clip(pred, 0, 100)[:, None] - LEVELS[None, :]).argmin(axis=1)]


def level_scores(y_true, y_level):
    steps = np.abs(np.asarray(y_true) - y_level) / 25
    return (steps == 0).mean(), (steps <= 1).mean()


# ---------------------------------------------------------------------------
# 2. 5-fold CV on TRAIN ONLY (same folds as the other scripts)
# ---------------------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for fold, (tr, va) in enumerate(kf.split(X_train), start=1):
    res = fit_ordinal(X_train.iloc[tr], y_train.iloc[tr])
    exp_va, lvl_va = predict_both(res, X_train.iloc[va])
    y_va = y_train.iloc[va]
    exact, within1 = level_scores(y_va, lvl_va)
    cv_rows.append({
        "r2": r2_score(y_va, exp_va),
        "rmse": mean_squared_error(y_va, exp_va) ** 0.5,
        "mae": mean_absolute_error(y_va, exp_va),
        "exact_level": exact, "within_one_level": within1,
    })
    print(f"  fold {fold}: R2 {cv_rows[-1]['r2']:.3f}, exact level {exact:.3f}")
cv = pd.DataFrame(cv_rows).mean()
print("\n5-fold CV on train (expected score):")
print(f"  R2 {cv['r2']:.3f} | RMSE {cv['rmse']:.3f} | MAE {cv['mae']:.3f} | "
      f"exact level {cv['exact_level']:.3f} | within one level {cv['within_one_level']:.3f}")

# ---------------------------------------------------------------------------
# 3. Fit on the full training set
# ---------------------------------------------------------------------------
final = fit_ordinal(X_train, y_train)
if not final.mle_retvals.get("converged", True):
    print("WARNING: the optimiser did not report convergence - treat results with care.")

n_feat = len(model_features)
conf = final.conf_int()
coef_table = pd.DataFrame({
    "feature": model_features,
    "coefficient_log_odds": final.params.values[:n_feat],
    "odds_ratio": np.exp(final.params.values[:n_feat]),
    "std_err": final.bse.values[:n_feat],
    "p_value": final.pvalues.values[:n_feat],
    "ci_lower_95": conf.iloc[:n_feat, 0].values,
    "ci_upper_95": conf.iloc[:n_feat, 1].values,
})
coef_table["significant_at_0.05"] = coef_table["p_value"] < 0.05
coef_table = coef_table.sort_values("coefficient_log_odds", ascending=False)

# Cut-points: statsmodels stores the first one directly, the rest as log-gaps.
raw = final.params.values[n_feat:]
cutpoints = np.concatenate([[raw[0]], raw[0] + np.cumsum(np.exp(raw[1:]))])
print("\nCut-points between levels (on the model's score):")
for (lo, hi), c in zip(zip(LEVELS[:-1], LEVELS[1:]), cutpoints):
    print(f"  {lo:>3} | {hi:<3}: {c:.2f}")

# ---------------------------------------------------------------------------
# 4. Score on TEST exactly once
# ---------------------------------------------------------------------------
exp_test, lvl_test = predict_both(final, X_test)
ord_exact, ord_within1 = level_scores(y_test, lvl_test)

# Linear regression on the same data, rounded to the nearest level, for a fair accuracy comparison
lin = LinearRegression().fit(X_train, y_train)
lin_pred = lin.predict(X_test)
lin_exact, lin_within1 = level_scores(y_test, round_to_level(lin_pred))

# Baselines: always the training mean (for R2/RMSE/MAE), always the most common level (for accuracy)
mode_level = int(y_train.mode()[0])
base_exact, base_within1 = level_scores(y_test, np.full(len(y_test), mode_level))
base_pred = np.full(len(y_test), y_train.mean())

rows = [
    {"stage": "5-fold CV (train)", "model": "Ordinal regression (expected score)",
     "r2": cv["r2"], "rmse": cv["rmse"], "mae": cv["mae"],
     "exact_level": cv["exact_level"], "within_one_level": cv["within_one_level"]},
    {"stage": "Final test (once)", "model": "Ordinal regression (expected score)",
     "r2": r2_score(y_test, exp_test), "rmse": mean_squared_error(y_test, exp_test) ** 0.5,
     "mae": mean_absolute_error(y_test, exp_test),
     "exact_level": ord_exact, "within_one_level": ord_within1},
    {"stage": "Final test (once)", "model": "Ordinal regression (most likely level)",
     "r2": r2_score(y_test, lvl_test), "rmse": mean_squared_error(y_test, lvl_test) ** 0.5,
     "mae": mean_absolute_error(y_test, lvl_test),
     "exact_level": ord_exact, "within_one_level": ord_within1},
    {"stage": "Final test (once)", "model": "Linear regression",
     "r2": r2_score(y_test, lin_pred), "rmse": mean_squared_error(y_test, lin_pred) ** 0.5,
     "mae": mean_absolute_error(y_test, lin_pred),
     "exact_level": lin_exact, "within_one_level": lin_within1},
    {"stage": "Final test (once)", "model": f"Baseline (mean for R2/RMSE/MAE; always {mode_level} for accuracy)",
     "r2": r2_score(y_test, base_pred), "rmse": mean_squared_error(y_test, base_pred) ** 0.5,
     "mae": mean_absolute_error(y_test, base_pred),
     "exact_level": base_exact, "within_one_level": base_within1},
]
metrics = pd.DataFrame(rows)

# Random forest test scores from its own script, if available (no accuracy columns there)
rf_path = PROCESSED / "05_random_forest_metrics.csv"
if rf_path.exists():
    rf = pd.read_csv(rf_path)
    rf = rf[(rf["model"] == "RandomForest") & rf["stage"].str.startswith("Final")]
    if len(rf):
        r = rf.iloc[0]
        metrics.loc[len(metrics)] = {"stage": "Final test (once)", "model": "Random forest (from 05_random_forest_metrics.csv)",
                                     "r2": r["r2"], "rmse": r["rmse"], "mae": r["mae"],
                                     "exact_level": np.nan, "within_one_level": np.nan}

pd.set_option("display.width", 160)
print("\nFinal evaluation on TEST (done exactly once):")
print(metrics.round(3).to_string(index=False))

print("\nHow often each level is predicted (test):")
print(pd.crosstab(pd.Series(lvl_test, name="predicted"), columns="count").T.to_string())

# ---------------------------------------------------------------------------
# 5. Save outputs
# ---------------------------------------------------------------------------
PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
coef_table.to_csv(PROCESSED / "06_ordinal_regression_coefficients.csv", index=False)
metrics.to_csv(PROCESSED / "06_ordinal_regression_metrics.csv", index=False)
pd.DataFrame({"between_levels": [f"{lo}|{hi}" for lo, hi in zip(LEVELS[:-1], LEVELS[1:])],
              "cutpoint": cutpoints}).to_csv(PROCESSED / "06_ordinal_regression_cutpoints.csv", index=False)

# Confusion matrix: rows = true level, columns = predicted level, shown as % of each true level
cm = confusion_matrix(y_test, lvl_test, labels=LEVELS)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{cm_pct[i, j]:.0f}%\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                color="white" if cm_pct[i, j] > 55 else "#1f1f1f")
ax.set_xticks(range(5), LEVELS)
ax.set_yticks(range(5), LEVELS)
ax.set_xlabel("Predicted level (most likely)")
ax.set_ylabel("True level")
ax.set_title("Ordinal regression: true vs. predicted level (test set)\n% of each true level")
fig.colorbar(im, ax=ax, label="% of row")
fig.tight_layout()
fig.savefig(FIGURES / "06_ordinal_regression_confusion_matrix.png", dpi=150)
plt.close(fig)

print("\nTop coefficients (odds ratio = how the odds of a HIGHER level multiply):")
print(coef_table[["feature", "coefficient_log_odds", "odds_ratio", "p_value", "significant_at_0.05"]]
      .round(4).head(10).to_string(index=False))
print("\nSaved:")
for p in ["06_ordinal_regression_coefficients.csv", "06_ordinal_regression_metrics.csv",
          "06_ordinal_regression_cutpoints.csv"]:
    print(f"  {PROCESSED / p}")
print(f"  {FIGURES / '06_ordinal_regression_confusion_matrix.png'}")
