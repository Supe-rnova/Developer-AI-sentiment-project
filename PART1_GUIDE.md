# Part 1 guide - your share: cleaning, manipulation, pre-processing

Written for a beginner. Read top to bottom once, then use sections 5-7 when you prepare slides and talk to teammates.

---

## 1. Your understanding, checked

**What you said:** *"For regression you take the whole dataset, make a heatmap, look at correlation coefficients and RMSE, and keep only the features that predict y well."*

**What is right**
- You should pick x from **evidence**, not from a hunch. Your v1 picked 8 columns by feel - you were right to call that a mistake.
- A correlation table / heatmap against y is a normal first screen for a regression project.

**What needs correcting**

| # | Your version | Correct version | Why it matters |
|---|---|---|---|
| 1 | Use the **entire** dataset | Split first, then use the **training set only** | If the test rows help decide *which columns to keep*, the test is no longer a fair exam. The final score looks better than it really is (data leakage). This is the most common beginner trap. |
| 2 | Use correlation **and RMSE** to pick columns | Correlation (and the heatmap) screens columns **before** a model exists. RMSE only exists **after** a model makes predictions. You can use it to compare two *candidate feature sets* (with cross-validation **on the training set**), and you report RMSE on the test set **once**, at the end. | RMSE is not a number you read off a heatmap. |
| 3 | Everything is eligible for the correlation check | Remove columns first **by reasoning about what they are**: IDs, columns built from y (leakage), exact duplicates, constants, metadata. | Leaked columns have the *best* correlation with y, so a heatmap would pick exactly the wrong ones. Example in this data: `ai_overall_sentiment` is the raw answer `AI_Sentiment_Score` is calculated from. |
| 4 | High correlation with y = good feature | It is a good *first* screen, but (a) it only sees **straight-line** relationships (the random forest can see more), (b) two x's that are strongly correlated with each other carry the same information, so keep one, (c) correlation is not cause. | Otherwise you keep 5 copies of one idea. |

**Two sentences in your team's debriefs need fixing** (tell the owner, @HaMan):
1. *"In a first check, the four AI variables did nearly all of the work…"* - that check was done by looking at data, and the 8 columns were chosen first. Remove it; replace it with the real result after the teammate's correlation step.
2. The methodology order is "Reshape (keep y and 8 predictors) → Explore (heatmap) → Pre-process (split)". The split must come **before** the heatmap that is used to choose x. New order: **Clean → Manipulate → Split → (fit fill/encode/scale on train) → correlation on train → choose x → model → test once.**
(Also, the debrief says the y was chosen because "nothing else predicts" job-threat concern. That is fine for a course project, but state it plainly as "we switched target because…", don't present it as planned from the start.)

---

## 2. What I changed in your project

Your v1 pipeline was already built (the split-before-fill order, the recode, the reflections). The one structural mistake was the `KEEP = [...]` list that hard-coded 8 predictors and threw the other columns away. Now:

| | v1 | v2 (now) |
|---|---|---|
| Predictors going to the next teammate | 8, hand-picked | **26 candidates → 37 model-ready columns** (every legitimate column) |
| Why a column is dropped | "not in the plan" | only for what it **is**: id, leakage, duplicate, constant, metadata - each **proved in code** and listed in `02_column_audit.csv` |
| Who picks x | you (by guess) | teammate, by correlation on **train only** |
| Boolean tool/benefit flags | dropped | kept as 0/1 (19 columns), renamed (`tool_writing_code`) |
| Four unlabeled category-code columns | kept as predictors | removed: no code book, so they cannot be interpreted |
| New files | | `02_column_audit.csv`, `01_cleaning_report.csv`, `02_missing_report.csv`, `03_feature_map.csv`, `03_train_readable.csv`, `03_test_readable.csv` |

Files changed: `01_data_prep.py`, `README.md`, new `PART1_GUIDE.md`, and everything in `data/processed/` (same names, new contents). Git keeps your old version.

Key numbers (all from the run): 65,437 raw rows → 45,873 with a target → **37,536 current AI users** → train 30,028 / test 7,508 (80/20, stratified, `random_state=42`) → 30,510 rows have no gap at all.

---

## 3. The three stages in plain words (keep them distinct - marks depend on it)

| Stage | The question | What we did here | What we did NOT do here |
|---|---|---|---|
| **Cleaning** - *is it correct?* | Fix what is **wrong** | dropped rows with no y; TRUE/FALSE → 0/1; **recoded `ai_complex_task_concern`** into true order; range checks; checked duplicates and count columns | no filtering, no scaling, no filling of gaps |
| **Manipulation** - *is it the shape I want?* | Reshape to fit the question | filtered to current AI users; top-10 countries + Other; audited and dropped 16 columns with reasons (incl. 4 with no code book); renamed flags; derived `sentiment_label` (plots only) | repairs; model-specific transforms |
| **Pre-processing** - *is it ready for the algorithm?* | Make it digestible | **split 80/20**; median fill + standardise numbers; "Unknown" + one-hot `country`; 0/1 flags untouched | choosing predictors (that is a *modelling-prep* step for the next teammate, on train only) |

**Skipped on purpose (say this on the slide):** merging tables (one table), creating new predictors (dataset already has engineered scores), outlier removal (all values pass the range checks and are real answers), PCA (would hide which original columns matter - that is our question). **Found nothing to fix, but checked:** duplicates, impossible values, count-vs-flag mismatches.

Gap-filling sits in pre-processing, not cleaning, on purpose: the median must be learned from the **training rows only**.

---

## 4. Things I found that you should know

- `AI_Sentiment_Score` is built from `ai_overall_sentiment` (raw codes → 0/25/50/75/100, with "indifferent" and "unsure" both = 50). That raw column would be pure leakage.
- `Dependency_Signal` = exactly `50 + AI_Trust_Score/2`. It is the same column in disguise; keeping both would double-count.
- `ai_accuracy_trust` and `ai_job_threat_concern` are the raw alphabetical codes of `AI_Trust_Score` and `AI_Job_Integration_Comfort`; the ordered versions are kept.
- `AI_Tool_Count` and `AI_Benefit_Breadth` are exactly the sums of the 12 / 7 flags, so each count is redundant with its flags (flagged in the README).
- The recode `ai_complex_task_concern` is justified by the **wording of the answers**. Average sentiment going 57 → 70 → 75 → 82 → 91 across ratings 1-5 is only a typo check. Never adjust a recode to make a correlation look better.
- **Removed for lack of a code book:** `age_range`, `education_level`, `organization_size`, `developer_type`. The Kaggle file gives only numeric codes and no labels, so we cannot say what any code means, interpret a result about them, or explain them on a slide. We removed them rather than guess (role `no_codebook` in `02_column_audit.csv`). If you find the Stack Overflow code book, they can be added back as labelled categories.

---

## 5. Slide material (3 stages × Execution + Reflection)

| Slide | Execution evidence (files) | Reflection (already written in the script's `Reflection` cells) |
|---|---|---|
| Cleaning | `01_cleaning_report.csv` (what was checked / found / done); before-after of `ai_complex_task_concern` from `02_summary_by_complex_rating.csv` | problem solved · alternatives (delete vs set-to-missing; fill now vs later) · what breaks if skipped |
| Manipulation | `step_log.csv` (65,437 → 45,873 → 37,536 rows; 44 → 29 columns); `02_column_audit.csv` (why 16 columns left) | same three questions |
| Pre-processing | `03_feature_map.csv` (26 → 37 columns); first rows of `03_X_train.csv`; the split sizes | same three questions |

In the talk, be ready to say, in one sentence each: *why split before filling* (test must not influence anything) and *why we did not pick predictors by hand* (needs evidence, on train only).

---

## 6. Message to paste to the feature-selection teammate

> Data is ready (pull the latest repo). Please choose x like this:
> 1. Load `03_X_train.csv` and `03_y_train.csv` **only**. Do not open `03_X_test.csv` / `03_y_test.csv` until the final model is scored.
> 2. Compute each column's correlation with `AI_Sentiment_Score` (a sorted bar chart is clearer than a 37-column heatmap), then a heatmap of the strongest ones to spot x-vs-x duplicates.
> 3. Write your rule down **before** running it (e.g. keep |r| ≥ 0.10 with y; for any pair of x's with |r| ≥ 0.80 keep one).
> 4. Careful: `AI_Tool_Count` = sum of the `tool_*` flags and `AI_Benefit_Breadth` = sum of the `ben_*` flags - don't use a count with its own flags. One-hot columns of one question (see `03_feature_map.csv`) should be kept or dropped together.
> 5. Apply the same column list to `X_train` and `X_test`, hand over the list, and the modeller fits on train and scores on test **once**.
> Correlation only sees straight-line links, so also let the random forest's importance cross-check it.

---

## 7. Your checklist

1. Read `01_data_prep.py` top to bottom once (it is commented for beginners). Click **Run Cell** on each `# %%` to see what each step prints.
2. Confirm you get the same numbers as section 2 on your machine (37,536 / 30,028 / 7,508 / 37 columns).
3. Decide with your team the one judgement call: keep or drop `AI_Job_Integration_Comfort`.
4. Tell @HaMan about the two debrief sentences in section 1.
5. `git add -A`, `git commit -m "Data prep v2: keep all candidate predictors, split-first"`, `git push` (see `GITHUB_SETUP.md`), then tell the teammates to `git pull`.
