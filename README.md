# Predicting Developer Sentiment Toward AI

Group project: predict `AI_Sentiment_Score` (0-100, how favourably a developer views AI tools) from 8 predictors, using the *Global Developer Attitudes Toward AI 2024* dataset (Kaggle, from the 2024 Stack Overflow Developer Survey, ODbL licence).

## Quick start
1. `pip install -r requirements.txt`
2. Download the CSV from Kaggle and save it as `data/raw/ai_developer_attitudes_2024.csv` (not stored in GitHub).
3. Open the project **folder** in VS Code (File -> Open Folder), open `01_data_prep.py`, and either press the Run button (top right) or run `python 01_data_prep.py` in the terminal. Click **Run Cell** above any `# %%` line to run it in small pieces. It rebuilds everything in `data/processed/`.
   (The processed files are already in the repo, so you only need to re-run if the prep logic changes.)

New to GitHub? See `GITHUB_SETUP.md`.

## Who works on what
| File | Owner | Purpose |
|---|---|---|
| `01_data_prep.py` | data prep (cleaning, manipulation, pre-processing) | raw CSV -> model-ready data |
| `02_visualization.py` | visualization | EDA plots |
| `03_modeling.py` | modeling | linear regression + random forest |
| `slides/` | presentation | the 5-minute deck |

## Which file should I use?
| File in `data/processed/` | Use it for |
|---|---|
| `01_cleaned.csv` | evidence for the cleaning slide (all 45,873 rows, all columns) |
| **`02_manipulated.csv`** | **Visualization.** 37,536 current AI users, 1 row each, readable (unscaled) values |
| `02_summary_by_country.csv`, `02_summary_by_complex_rating.csv` | ready-made tables for slides |
| **`03_X_train.csv`, `03_X_test.csv`, `03_y_train.csv`, `03_y_test.csv`** | **Modeling.** Filled, one-hot encoded and scaled. 30,028 train / 7,508 test rows, 37 feature columns |
| `03_preprocessor.joblib` | the fitted recipe (`joblib.load`) to apply to new data |
| `step_log.csv` | rows/columns after each step, for slides |

Plot from `02_manipulated.csv`, never from `03_*` (scaled values like -1.05 mean nothing to a viewer).

Load examples:
```python
import pandas as pd
viz = pd.read_csv("data/processed/02_manipulated.csv")
X_train = pd.read_csv("data/processed/03_X_train.csv"); y_train = pd.read_csv("data/processed/03_y_train.csv").squeeze()
X_test  = pd.read_csv("data/processed/03_X_test.csv");  y_test  = pd.read_csv("data/processed/03_y_test.csv").squeeze()
```

## Data dictionary (`02_manipulated.csv`)
| Column | Meaning |
|---|---|
| `respondent_id` | ID only, not a predictor |
| `AI_Sentiment_Score` | **target (y)**: 0, 25, 50, 75, 100 (very unfavourable ... very favourable) |
| `sentiment_label` | text version of the target, for plots only |
| `AI_Trust_Score` | trust in AI accuracy, 0-100 |
| `ai_complex_task_concern` | how well AI handles complex tasks, **1 = very poor ... 5 = very well** (recoded from raw alphabetical codes) |
| `AI_Benefit_Breadth` | number of benefits seen, 0-7 |
| `AI_Tool_Count` | number of tasks AI is used for, 0-12 |
| `years_professional_coding_num` | years of professional coding |
| `country` | top 10 countries + "Other" |
| `organization_size`, `education_level` | category codes (`code_1`, `code_2` ...), not numbers |

## Things to know
- **Missing values** remain in `02_manipulated.csv` on purpose (needed for honest plots). They are filled only in the `03_*` files, using the training set only. Most gaps are in `organization_size` (24%) and `years_professional_coding_num` (17%); `country` (5%), `education_level` (4%), `ai_complex_task_concern` (1.4%) and `AI_Trust_Score` (0.7%) have few. `28,064` rows have no missing predictor if the team prefers to drop incomplete rows instead.
- **Baseline** for the models: always predicting the average (74.2) -> R-squared of 0.
- **Dropped on purpose:** `ai_overall_sentiment`, `ai_tool_stance`, `ai_accuracy_trust`, `AI_Adoption_*`, `Dependency_Signal` (they are the raw or closely related versions of the target and would leak the answer).
- The train/test split uses `random_state=42` and is stratified on the target, so everyone gets the same split.
