# Updates log

A running record of decisions made on this project, in plain language, so
anyone picking this up later knows *what* was decided and *why* - not just
what the code does.

---

## Modeling stage (linear regression)

### 1. Which features go into the model
**Decision:** use the full list of 29 features already chosen by the
feature-selection teammate (`feature_selection/selected_features.csv`), not
a smaller "top 10 by correlation" list.

**Why:** the top-10-by-raw-correlation list actually included two columns
(`AI_Benefit_Breadth`, `AI_Tool_Count`) that were already deliberately
dropped earlier, because they're just the sum of other flag columns already
in the list - keeping them would double-count the same information. Taking
a shortcut-top-10 would also silently throw away `country`, even though the
team already found (with evidence, R2 = 0.014) that it's worth keeping.
Going smaller would need its own justification, not just "fewer columns is
simpler" - so for this pass we kept the full, already-agreed-on set.

### 2. The dummy-variable trap (country columns)
**What it is:** `country` was turned into 12 columns (`country_Brazil`,
`country_Canada`, ... `country_Unknown`), each 0 or 1, and for every person
exactly one of them is 1. That means if you know 11 of the 12 columns, you
can always figure out the 12th one for free - it carries no new
information once the other 11 are known. A plain linear regression doesn't
like that: it needs every input to carry at least some information the
others don't, and with all 12 included the equation has many "equally
correct" answers instead of one, so the individual country coefficients
become meaningless (predictions still work, but you can't trust what any
single country's number says).

**Decision:** drop one country column and use it as the baseline everyone
else is compared to. Dropped `country_Unknown` (not a real country, so the
least meaningful one to lose). Final model uses 28 columns (29 selected -
1 dropped). Every remaining country coefficient now reads as "difference
from an unknown-country respondent", which is interpretable.

### 3. 5-fold cross-validation on train, before touching test
**What it is:** split the 30,028 training rows into 5 chunks, train on 4,
check the score on the 5th, rotate 5 times, average the 5 scores. No test
data involved at all.

**Why now, before the final check:** this is a "practice exam" used to
decide things (which features, which model) safely - all on data the model
could in principle use for training. The test set is the one-time "real
exam" that can only be opened once, after every decision is already locked
in. If we used the real test set to compare options instead, we'd be
reusing it over and over to make decisions - the same test-set-leakage
mistake the project already fixed once during feature selection. CV result
on train (R2 = 0.339) came out very close to the final test result
(R2 = 0.328), which is a good sign - the model isn't overfitting to the
training folds.

### 4. Inferential stats alongside the predictive model
**Decision:** fit the same 28 features twice - once with `sklearn`
(for CV scores and the test prediction) and once with `statsmodels` OLS
(for p-values and 95% confidence intervals).

**Why:** `sklearn`'s LinearRegression is built for prediction and doesn't
report p-values by default. For the slide, we also want to say which
features are *statistically reliable* predictors (not just which ones have
the biggest coefficient), so `statsmodels` was added purely for that
reporting layer - it doesn't change the predictions.

### 5. One script per model
**Decision:** linear regression and random forest go in separate plain
`.py` files (no `# %%` cells), e.g. `03_modeling_linear.py` and a later
`03_modeling_forest.py` (or similar), rather than one combined script.

**Why:** keeps each model's logic, outputs and any model-specific choices
independent and easier to read/debug on their own, and matches the
project's existing one-file-per-stage pattern (`01_data_prep.py`,
`02_viz_and_feature_selection.ipynb`).

### 6. Baseline comparison
**Decision:** always report the model's test score next to a "dumb"
baseline that just predicts the training mean (74.2) for every row.

**Why:** a bare R2 or RMSE number means little on its own. The baseline
(R2 = 0 by definition) is the "did we learn anything at all" floor - the
model has to clear that bar before any other claim about it means
anything. Linear regression cleared it clearly (R2 = 0.328 vs 0.000).

---

## Results so far (linear regression, 28 features)

| Stage | R2 | RMSE | MAE |
|---|---|---|---|
| 5-fold CV (train) | 0.339 | 17.32 | 13.60 |
| Test (once) | 0.328 | 17.46 | 13.61 |
| Mean baseline (test) | 0.000 | 21.30 | 14.63 |

In plain terms: predictions are off by about 13-17 points on average (on a
0-100 scale with only 5 possible values), versus 21 points for just
guessing the average every time. The model explains roughly a third of why
sentiment varies - a real improvement over guessing, but most of the
variation in sentiment still isn't captured by these 28 features. That's a
normal result for survey/attitude data, not a sign of a bug.

Files: `03_modeling_linear.py`, `data/processed/04_linear_regression_coefficients.csv`,
`data/processed/04_linear_regression_metrics.csv`,
`figures/04_linear_regression_pred_vs_actual.png`,
`figures/04_linear_regression_residuals.png`.

---

## Modeling stage (random forest)

### 7. Same feature set as linear regression
**Decision:** use the identical 28 columns (29 selected minus
`country_Unknown`) for the random forest, even though trees don't actually
need the dummy-variable fix (a tree splits on one column at a time, so it
doesn't care that the 12 country columns add up to 1 - there's no shared
equation to confuse).

**Why:** the point of building a second model is to compare it fairly
against the first one. If the two models saw different inputs, a
difference in score could be "different model" or "different data" - we
wouldn't know which. Keeping the inputs identical isolates the comparison
to just "does a non-linear model do better here."

### 8. Default-ish settings, no tuning
**Decision:** `RandomForestRegressor(n_estimators=300, random_state=42)`,
otherwise default settings (no grid search over tree depth, leaf size,
etc.).

**Why:** the goal here was a reasonable cross-check on the linear
regression, not a second full optimisation project. 300 trees is enough to
get a stable average without taking long to run. Deeper tuning could
likely close some of the gap below, but that's a possible next step, not
something done here.

### 9. Result: random forest did *worse* than linear regression here
| Stage | Model | R2 | RMSE | MAE |
|---|---|---|---|---|
| 5-fold CV (train) | Linear regression | 0.339 | 17.32 | 13.60 |
| 5-fold CV (train) | Random forest | 0.224 | 18.76 | 14.55 |
| Test (once) | Linear regression | 0.328 | 17.46 | 13.61 |
| Test (once) | Random forest | 0.234 | 18.64 | 14.36 |
| Test (once) | Mean baseline | 0.000 | 21.30 | 14.63 |

**What this tells us, in plain language:** both models beat the baseline,
but the straight-line model (linear regression) actually did noticeably
better than the random forest (R2 0.328 vs 0.234). That's not a mistake -
it's a real finding worth saying out loud on the slide: it suggests the
relationships in this data are mostly simple and additive (each benefit
flag, tool flag, or trust point adds roughly the same fixed amount to
sentiment, regardless of what else is true about the person), rather than
full of curves or "it depends" interactions that a tree-based model is
built to catch. A random forest with careful tuning might close some of
that gap, but out of the box, the simpler model wins here - which is a
legitimate and useful result, not a failure of the random forest script.

CV and test scores for the random forest are close to each other
(0.224 vs 0.234), so it isn't overfitting - it's just a genuinely weaker
fit for this data than the linear model.

Files: `03_modeling_forest.py`,
`data/processed/05_random_forest_feature_importance.csv`,
`data/processed/05_random_forest_metrics.csv`,
`figures/05_random_forest_pred_vs_actual.png`,
`figures/05_random_forest_residuals.png`,
`figures/05_random_forest_feature_importance.png`.
