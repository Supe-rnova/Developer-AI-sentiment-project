# Predicting Developer Sentiment Toward AI

Group project: predict `AI_Sentiment_Score` (0-100, how favourably a developer views AI tools) using the
*Global Developer Attitudes Toward AI 2024* dataset (Kaggle, from the 2024 Stack Overflow Developer Survey, ODbL licence).

> **Version 2 (data prep corrected).** v1 hand-picked 8 predictors before looking at any evidence. v2 keeps **all 26 legitimate
> candidate predictors** (37 model-ready columns). Choosing the final predictors (x) is now the **feature-selection teammate's** job,
> using correlation on the **training set only**. Pull the latest files before you start - names are the same, contents changed.

## Quick start
1. `pip install -r requirements.txt`
2. Put the Kaggle CSV at `data/raw/ai_developer_attitudes_2024.csv` (not stored in GitHub).
3. Open the project **folder** in VS Code, run `01_data_prep.py` (Run button, or `python 01_data_prep.py`). It rebuilds everything in `data/processed/`.
   Click **Run Cell** above any `# %%` line to run it in small pieces. Read `PART1_GUIDE.md` for the explanation.

## Who works on what
| File | Owner | Purpose |
|---|---|---|
| `01_data_prep.py` | data prep | raw CSV -> cleaned -> manipulated -> model-ready (**done**) |
| visualization + feature-selection script (new) | visualization + feature selection | plots, then correlation + heatmap on **train only** -> choose x |
| `03_modeling.py` | modeling | linear regression + random forest on the chosen x |
| `slides/` | presentation | the 5-minute deck |

## Which file should I use?
| File in `data/processed/` | What it is | Use it for |
|---|---|---|
| `02_manipulated.csv` | 37,536 current AI users, readable values, id + target + 26 candidates | **Descriptive plots only**: how y is distributed, counts, missingness |
| `03_train_readable.csv` / `03_test_readable.csv` | the same rows as `03_X_*`, readable (unscaled, with `respondent_id`, `sentiment_label`) | plots that help *choose* features (train only) |
| **`03_X_train.csv`, `03_y_train.csv`** | 30,028 rows x 37 columns, filled / one-hot / scaled | **Feature selection (correlation, heatmap) and model fitting** |
| **`03_X_test.csv`, `03_y_test.csv`** | 7,508 rows | **Do not touch** until the final model is scored |
| `03_feature_map.csv` | which model column came from which original column and its kind | grouping one-hot columns back to their question |
| `02_column_audit.csv` | all 44 raw columns: kept or dropped, and why | slides, questions about "where did column X go?" |
| `01_cleaning_report.csv`, `step_log.csv`, `02_missing_report.csv` | what was found and fixed; rows after each step; % missing | slides (cleaning / manipulation evidence) |

Rows in `03_X_train`, `03_y_train` and `03_train_readable` are in the **same order** (row *i* is the same person in all three).

## Rules that keep the results honest
1. **Feature selection uses TRAIN only** (`03_X_train` + `03_y_train`). Never compute correlations on `02_manipulated.csv` or on train+test combined to decide which x to keep.
2. **The test set is opened once**, at the very end, to report the final score.
3. **Do not choose columns by looking at test results.** Write down your selection rule (e.g. "|r| with y >= 0.10, and drop one of any pair of x's with |r| >= 0.80") *before* running it.
4. Plot from the readable files, not the scaled `03_X_*` (a value like -1.05 means nothing to a viewer).

## Data dictionary (`02_manipulated.csv`)
| Column(s) | Meaning |
|---|---|
| `respondent_id` | ID only - never a predictor |
| `AI_Sentiment_Score` | **target (y)**: 0, 25, 50, 75, 100 (very unfavourable ... very favourable; 50 mixes "indifferent" and "unsure") |
| `sentiment_label` | text version of y, for plots only - never a predictor |
| `AI_Trust_Score` | trust in AI accuracy, 0-100 |
| `ai_complex_task_concern` | how well AI handles complex tasks, **1 = very poor ... 5 = very well** (recoded by us from alphabetical raw codes) |
| `AI_Job_Integration_Comfort` | 10 = believes AI threatens their job, 50 = unsure, 100 = does not |
| `AI_Benefit_Breadth` | number of benefits seen (0-7) = sum of the 7 `ben_*` flags |
| `AI_Tool_Count` | number of tasks AI is used for (0-12) = sum of the 12 `tool_*` flags |
| `ben_*` (7), `tool_*` (12) | 0/1 flags: which benefits / tasks the person ticked |
| `years_professional_coding_num` | years of professional coding |
| `country` | top 10 countries + "Other" |

## Things to know before choosing x
- **Redundant columns on purpose:** `AI_Tool_Count` is exactly the sum of the `tool_*` flags, and `AI_Benefit_Breadth` the sum of the `ben_*` flags.
  Do not put a count **and** its own flags into the same linear regression. Pick one side.
- **One-hot columns:** `country` became 11 0/1 columns (10 countries + Other + Unknown) that add up to 1. Keep or drop them together (see `03_feature_map.csv`), and drop one level (or use `drop="first"`) when fitting a plain linear regression.
- **Missing values** were filled in `03_*` using the training set only (median for numbers, "Unknown" for categories). Gaps: `years_professional_coding_num` 17%, `country` 5%, `AI_Job_Integration_Comfort` 3%, `ai_complex_task_concern` 1%, `AI_Trust_Score` 1%. `30,510` rows have no gap at all if the team prefers dropping incomplete rows.
- **Judgement call - `AI_Job_Integration_Comfort`:** it is another opinion about AI (not computed from y), so it is kept as a candidate. If the team feels it is too close to "sentiment", drop it and say why on the slide.
- **Dropped on purpose (reasons proven in the script):** `ai_overall_sentiment` (raw version of y = leakage); `ai_tool_stance`, `AI_Adoption_Level`, `AI_Adoption_Score` (one value after filtering); `ai_accuracy_trust`, `Dependency_Signal`, `years_professional_coding`, `ai_job_threat_concern` (exact duplicates of kept columns); 4 constant metadata columns; **`developer_type`, `age_range`, `education_level`, `organization_size` (no code book: the Kaggle file gives only numeric codes, so we cannot say what they mean and removed them rather than guess; they can be added back if the Stack Overflow code book is found).**
- **Baseline** for the models: always predicting the average (74.2) gives R-squared 0.
- The split uses `random_state=42`, stratified on y, so everyone gets the same rows.
