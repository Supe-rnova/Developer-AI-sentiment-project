"""
01_data_prep.py - Data prep: Cleaning -> Manipulation -> Pre-processing
Project: Predicting developer sentiment toward AI (target: AI_Sentiment_Score)

HOW TO RUN (VS Code)
  * Whole file : open the project FOLDER in VS Code, then press the Run button (or: python 01_data_prep.py)
  * Step by step: click "Run Cell" above any '# %%' line (needs the Python + Jupyter extensions)
Needs: data/raw/ai_developer_attitudes_2024.csv  (download from Kaggle; not stored in GitHub)
"""

# %% [markdown]
# # 01 - Data Prep: Cleaning -> Manipulation -> Pre-processing
# **Project:** Predicting developer sentiment toward AI (target `AI_Sentiment_Score`)
# **Dataset:** Global Developer Attitudes Toward AI 2024 (Kaggle, from the 2024 Stack Overflow Developer Survey)
#
# This script takes the raw CSV to a **model-ready** dataset. It is split into three stages that follow the course handout.
# Each stage has a different goal, and each ends with a short **Reflection** (what problem it solved, what else we considered, what could go wrong if skipped).
#
# | Stage | Goal (the question it answers) | Output file |
# |---|---|---|
# | 1. Cleaning | Is the data **correct**? | `data/processed/01_cleaned.csv` |
# | 2. Manipulation | Is the data in the **shape** we need? | `data/processed/02_manipulated.csv` |
# | 3. Pre-processing | Is the data **ready for the algorithm**? | `data/processed/03_X_train.csv`, `03_X_test.csv`, `03_y_train.csv`, `03_y_test.csv` |
#
# **For teammates:** use `02_manipulated.csv` for visualization (values are still human-readable). Use the `03_*` files for modeling.
#
# **How to run:** put the Kaggle CSV at `data/raw/ai_developer_attitudes_2024.csv`, then run this file (see the top of the file).

# %%
import numpy as np
import pandas as pd
from pathlib import Path

# ---- Settings (change here only) ----
try:
    ROOT = Path(__file__).resolve().parent      # folder this file lives in
except NameError:                                # running as an interactive cell
    ROOT = Path.cwd()
RAW_PATH = ROOT / "data" / "raw" / "ai_developer_attitudes_2024.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42          # makes the train/test split repeatable for everyone

TARGET = "AI_Sentiment_Score"                       # y
NUMERIC_FEATURES = [                                # x (numbers)
    "AI_Trust_Score",
    "ai_complex_task_concern",
    "AI_Benefit_Breadth",
    "AI_Tool_Count",
    "years_professional_coding_num",
]
CATEGORICAL_FEATURES = ["country", "organization_size", "education_level"]   # x (categories)

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)

# Small log so the team can quote 'rows before / after' on slides
step_log = []
def log_step(stage, step, frame):
    step_log.append({"stage": stage, "step": step, "rows": len(frame), "columns": frame.shape[1]})

# %% [markdown]
# ## 0. Load and first look
# Before changing anything, look at what we have. We keep an untouched copy (`df_raw`) so we can always compare back.

# %%
df_raw = pd.read_csv(RAW_PATH)
df = df_raw.copy()
log_step("0 raw", "loaded raw file", df)

print("Shape (rows, columns):", df.shape)
print()
print(df.dtypes.value_counts())
print(df.head())

# %%
# Missing values in the columns we care about (as % of rows)
cols_of_interest = [TARGET] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
print((df[cols_of_interest].isna().mean() * 100).round(1).sort_values(ascending=False))

# %% [markdown]
# ---
# # STAGE 1 - DATA CLEANING
# **Goal: make the data *correct*.** We repair or remove things that are wrong. We do **not** reshape the data yet and we do **not** do model-specific transformations (scaling, encoding).
# (Handout analogy: *washing the vegetables and throwing out the rotten ones.*)
#
# ### 1.1 Duplicates

# %%
n_full = df.duplicated().sum()
n_id = df["respondent_id"].duplicated().sum()
print("Fully duplicated rows:", n_full)
print("Repeated respondent_id:", n_id)

df = df.drop_duplicates()                                   # exact duplicate rows
df = df.drop_duplicates(subset="respondent_id", keep="first")  # same person listed twice
log_step("1 cleaning", "removed duplicates", df)
print("Rows now:", len(df))
print("(0 duplicates means the data arrived clean on this point - we still checked.)")

# %% [markdown]
# ### 1.2 The target (y): missing and impossible values

# %%
print(df[TARGET].value_counts(dropna=False).sort_index())

# A model cannot learn from a blank answer, so rows with no target are dropped.
df = df[df[TARGET].notna()].copy()

# The score should only ever be one of these five values:
VALID_SCORES = [0, 25, 50, 75, 100]
bad = ~df[TARGET].isin(VALID_SCORES)
print("\nRows with an invalid target value:", int(bad.sum()))
df = df[~bad].copy()
log_step("1 cleaning", "dropped rows with missing/invalid target", df)
print("Rows now:", len(df))

# %% [markdown]
# ### 1.3 Wrong data types and messy text
# * `education_level` and `organization_size` are stored as **floats**, but they are really *category codes* (e.g. 3.0 is a label, not "three of something"). Averaging or scaling them would be meaningless, so we convert them to text categories.
# * Make sure the numeric predictors really are numeric.
# * Tidy text in `country` (stray spaces, empty strings).

# %%
# (a) coded categories: float -> text label such as "code_3"
CODED_COLS = ["education_level", "organization_size"]
for c in CODED_COLS:
    print(c, "unique codes:", sorted(df[c].dropna().unique()))
    df[c] = df[c].map(lambda v: f"code_{int(v)}" if pd.notna(v) else np.nan)

# (b) numeric predictors must be numeric; anything unparseable becomes NaN
for c in NUMERIC_FEATURES:
    before = df[c].isna().sum()
    df[c] = pd.to_numeric(df[c], errors="coerce")
    after = df[c].isna().sum()
    if after != before:
        print(f"{c}: {after - before} unparseable values turned into NaN")

# (c) text tidy-up for country
df["country"] = df["country"].astype("string").str.strip().replace("", pd.NA).astype(object)
df["country"] = df["country"].where(df["country"].notna(), np.nan)
print("\nDistinct countries:", df["country"].nunique())

# %% [markdown]
# ### 1.4 Recode `ai_complex_task_concern` into its true order
# The raw numbers are labels numbered in **alphabetical order of the answer text**, not in order of meaning. If we left them, a model would think "3" means "more" than "2" when it does not.
# We recode to a real scale: **1 = very poor ... 5 = very well**.
#
# The raw codes 1-5 follow the alphabetical order of the Stack Overflow answer texts:
# `1 Bad` | `2 Good, but not great` | `3 Neither good nor bad` | `4 Very poor` | `5 Very well`.
# So the true order is Very poor (1) < Bad (2) < Neither (3) < Good but not great (4) < Very well (5).
#
# > **Evidence the mapping is right:** the sanity check after the recode shows average sentiment rising steadily from rating 1 to 5. Before the recode it jumps up and down (70, 82, 75, 57, 91 for raw codes 1 to 5). The answer-text order is from the Stack Overflow survey; double-check it in the codebook if you want to be sure.

# %%
print("Raw values found:", sorted(df["ai_complex_task_concern"].dropna().unique()))
print("\nBEFORE recode - average sentiment per raw code (not in a sensible order):")
print(df.groupby("ai_complex_task_concern")[TARGET].mean().round(1).to_string())

RECODE_COMPLEX = {1.0: 2, 2.0: 4, 3.0: 3, 4.0: 1, 5.0: 5}   # raw code -> true rating (1..5)

found = set(df["ai_complex_task_concern"].dropna().unique())
if not found.issubset(RECODE_COMPLEX.keys()):
    raise ValueError(f"Unexpected codes {found - set(RECODE_COMPLEX)} - check RECODE_COMPLEX against the Kaggle/Stack Overflow codebook.")

df["ai_complex_task_concern"] = df["ai_complex_task_concern"].map(RECODE_COMPLEX)

# Sanity check: AFTER the recode, a better rating of AI on complex tasks should go with higher sentiment
print(df.groupby("ai_complex_task_concern")[TARGET].agg(["mean", "count"]).round(1))

# %% [markdown]
# ### 1.5 Impossible values (range checks)
# Each score has a known valid range. Values outside it are errors. We set them to *missing* (NaN) rather than deleting the whole row, so we keep the rest of that person's answers. The missing values are filled later, in pre-processing.

# %%
VALID_RANGES = {
    "AI_Trust_Score": (0, 100),
    "ai_complex_task_concern": (1, 5),
    "AI_Benefit_Breadth": (0, 7),
    "AI_Tool_Count": (0, 12),
    "years_professional_coding_num": (0, 60),   # survey tops out at "50+" (max in data is 51)
}
for col, (lo, hi) in VALID_RANGES.items():
    out = df[col].notna() & ~df[col].between(lo, hi)
    print(f"{col:32s} valid {lo}-{hi}: {int(out.sum())} impossible values")
    df.loc[out, col] = np.nan

print(df[NUMERIC_FEATURES].describe().round(2))

# %% [markdown]
# ### 1.6 Internal consistency
# `AI_Tool_Count` and `AI_Benefit_Breadth` are meant to be counts of the TRUE/FALSE columns `AIToolCurrently Using - ...` and `AIBen - ...`. We recompute them from the raw TRUE/FALSE columns and check that they agree.

# %%
tool_cols = [c for c in df.columns if c.startswith("AIToolCurrently Using - ")]
ben_cols = [c for c in df.columns if c.startswith("AIBen - ")]
print(len(tool_cols), "tool columns,", len(ben_cols), "benefit columns")

def count_true(frame, cols):
    return frame[cols].fillna(False).astype(bool).astype(int).sum(axis=1)

for target_col, cols in [("AI_Tool_Count", tool_cols), ("AI_Benefit_Breadth", ben_cols)]:
    recomputed = count_true(df, cols)
    mismatch = df[target_col].notna() & (df[target_col] != recomputed)
    print(f"{target_col}: {int(mismatch.sum())} rows disagree with the TRUE/FALSE columns")
    df.loc[mismatch, target_col] = recomputed[mismatch]     # the raw columns are the source of truth

# %% [markdown]
# ### 1.7 Missing values: report only (we do NOT fill them here)

# %%
print((df[cols_of_interest].isna().mean() * 100).round(1).sort_values(ascending=False).to_frame("% missing"))

# %%
df_clean = df.copy()
log_step("1 cleaning", "range/type/consistency fixes (no rows dropped)", df_clean)
df_clean.to_csv(OUT_DIR / "01_cleaned.csv", index=False)
print("Saved", OUT_DIR / "01_cleaned.csv", df_clean.shape)

# %% [markdown]
# ## Reflection - Cleaning
# | Question | Answer |
# |---|---|
# | **What problem did this solve?** | Removed duplicate people, rows with no target, impossible values and category codes that were stored as misleading numbers. The recode of `ai_complex_task_concern` gives the column its real meaning (1 = very poor ... 5 = very well). |
# | **What else did we consider?** | *Duplicates:* checked both whole rows and `respondent_id`. *Impossible values:* we could have deleted the rows; we set the value to missing instead so we keep the rest of that person's answers. *Missing predictor values:* we could fill them here, but filling uses statistics (median) that must be learned from the training set only, so it belongs in pre-processing. Dropping every incomplete row would also be possible (it loses roughly a quarter of the AI users) so we keep it as an option for the team. |
# | **What could go wrong if we skipped it?** | A model could be trained on duplicated people (inflating its score), on a target that is blank, and on a complex-task rating where "3" is not actually "more" than "2". Coefficients would then have the wrong sign or size, and the main question ("which factors push sentiment up or down?") would get a misleading answer. |

# %% [markdown]
# ---
# # STAGE 2 - DATA MANIPULATION
# **Goal: give the (now correct) data the *shape* our question needs.** Nothing is being repaired here: we filter, select, group and derive.
# (Handout analogy: *chopping and measuring into what the recipe calls for.*)
#
# ### 2.1 Filter: keep only developers who currently use AI tools
# The trust and complex-task questions were only asked of people who use AI, so non-users have nothing in those columns. Our predictors need those answers.
#
# `ai_tool_stance` holds the answer to "do you use AI tools?": **3 = yes, currently using**, 2 = not yet but plans to, 1 = no and does not plan to. (The same split appears in `AI_Adoption_Level`: 3 = current users.) Rule: keep `ai_tool_stance == 3`. This gives about **37,500** rows with a sentiment score, which matches the team debrief.

# %%
print("Answered trust question          :", df_clean["AI_Trust_Score"].notna().sum())
print("Answered complex-task question   :", df_clean["ai_complex_task_concern"].notna().sum())
print("Used AI for at least one task    :", (df_clean["AI_Tool_Count"] > 0).sum())
print("\nWho was asked the trust question? (share answered, by ai_tool_stance)")
print(df_clean.groupby("ai_tool_stance", dropna=False)["AI_Trust_Score"].apply(lambda s: round(s.notna().mean(), 3)).to_string())
print("\nai_tool_stance values (raw codes) among rows with a sentiment score:")
print(df_clean["ai_tool_stance"].value_counts(dropna=False).sort_index().to_string())

# %%
is_ai_user = df_clean["ai_tool_stance"] == 3        # 3 = currently uses AI tools
df_users = df_clean[is_ai_user].copy()
log_step("2 manipulation", "filtered to current AI users", df_users)
print("AI users kept:", len(df_users), "(team debrief expects ~37,500)")
print("Of those, answered trust question:", df_users["AI_Trust_Score"].notna().sum(),
      "| complex-task question:", df_users["ai_complex_task_concern"].notna().sum())

# %% [markdown]
# ### 2.2 Group countries: top 10 plus "Other"
# There are many countries and most have few people. One-hot encoding all of them would create dozens of near-empty columns. Missing country stays missing for now (filled in pre-processing).

# %%
top10 = df_users["country"].value_counts().head(10).index.tolist()
print("Top 10 countries:", top10)

orig = df_users["country"]
grouped = orig.where(orig.isin(top10), "Other")
df_users["country"] = grouped.where(orig.notna())

print(df_users["country"].value_counts(dropna=False))

# %% [markdown]
# ### 2.3 Select columns
# We keep the target, the 8 predictors from the debrief, and `respondent_id` (so any row can be traced back; it is **not** a predictor).
# We drop everything else. Two groups are dropped on purpose:
# * **Leakage risk:** `ai_overall_sentiment` is the raw answer that `AI_Sentiment_Score` is built from. `ai_tool_stance`, `ai_accuracy_trust`, `AI_Adoption_Level`, `AI_Adoption_Score` and `Dependency_Signal` are other AI-opinion scores that overlap with the target. Using the raw/derived version of the target would make the model look brilliant for the wrong reason.
# * **Not in the plan:** the individual TRUE/FALSE columns (already summarised by the two count columns), `developer_type`, `age_range`, etc.

# %%
# Metadata columns: confirm they are constant before dropping (cite them in the slides instead)
for c in ["data_source", "source_license", "source_url", "collection_year"]:
    print(c, "->", df_users[c].unique()[:3])

# %%
KEEP = ["respondent_id", TARGET] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
dropped = [c for c in df_users.columns if c not in KEEP]
print(f"Keeping {len(KEEP)} columns, dropping {len(dropped)}.")

df_model = df_users[KEEP].copy()
print(df_model.head())

# %% [markdown]
# ### 2.4 Derived column (for readable plots only)
# A text label for the 0/25/50/75/100 score, so charts say "Favourable" instead of "75". The meanings come from the team debrief; verify them against the Stack Overflow codebook. This column is **not** used by the model.

# %%
SENTIMENT_LABELS = {0: "Very unfavourable", 25: "Unfavourable", 50: "Indifferent/unsure",
                    75: "Favourable", 100: "Very favourable"}
df_model["sentiment_label"] = df_model[TARGET].map(SENTIMENT_LABELS)
print(df_model["sentiment_label"].value_counts().reindex(list(SENTIMENT_LABELS.values())))

# %% [markdown]
# ### 2.5 Group and aggregate: small summary tables for the slides and plots
# These answer simple questions and give the visualization teammates ready-made evidence.

# %%
by_country = (df_model.groupby("country")[TARGET]
              .agg(avg_sentiment="mean", n_developers="count")
              .round(1).sort_values("avg_sentiment", ascending=False))
by_complex = (df_model.groupby("ai_complex_task_concern")[TARGET]
              .agg(avg_sentiment="mean", n_developers="count").round(1))

print(by_country)
print(by_complex)

by_country.to_csv(OUT_DIR / "02_summary_by_country.csv")
print(by_complex.to_csv(OUT_DIR / "02_summary_by_complex_rating.csv"))

# %%
log_step("2 manipulation", "selected columns + grouped countries + label", df_model)
df_model.to_csv(OUT_DIR / "02_manipulated.csv", index=False)
print("Saved", OUT_DIR / "02_manipulated.csv", df_model.shape)
print("Baseline (always predict the average):", round(df_model[TARGET].mean(), 1))

# %% [markdown]
# ## Skipped on purpose (with reasons)
# * **Merging tables:** skipped, because everything we need is in one table.
# * **Creating new predictors:** skipped, because the dataset already contains engineered scores (`AI_Trust_Score`, `AI_Benefit_Breadth`, `AI_Tool_Count`) and the debrief fixes the 8 predictors. Only the plotting label above was derived.
# * **Removing outliers:** skipped, because after the range checks the remaining values are genuine survey answers, and random-forest/linear models can cope with them. We can revisit if the residual plot looks bad.
#
# ## Reflection - Manipulation
# | Question | Answer |
# |---|---|
# | **What problem did this solve?** | Reduced a 44-column survey of everyone to a focused table of AI users with just the target and 8 predictors, with a manageable country column. |
# | **What else did we consider?** | *Filter:* keep all rows and let missing values be filled; rejected, because for non-users trust and complex-task are blank by design, not by accident, so filling them would invent answers. We also considered 'answered the trust question' as the rule; `ai_tool_stance == 3` is cleaner because it is the survey's own definition of a current user. *Countries:* keep all (too many sparse columns) or top 5 (loses information); top 10 + Other is the balance in the debrief. |
# | **What could go wrong if we skipped it?** | The model would be trained on people who were never asked the key questions, with invented values for trust and complex-task. Keeping `ai_overall_sentiment` would leak the answer into the predictors and give a falsely high R-squared. |

# %% [markdown]
# ---
# # STAGE 3 - PRE-PROCESSING
# **Goal: make the data *ready for the algorithm*.** Models need complete numbers on a similar scale. These steps are model-specific.
# (Handout analogy: *the final prep right before it goes in the pan.*)
#
# ### 3.1 Split first (80% train / 20% test)
# **Order matters.** We split *before* filling or scaling, so the numbers used for filling and scaling come only from the training rows. If the test rows influenced them, the test score would be unfairly flattering (called *data leakage*).
# `stratify=y` keeps the five sentiment values in the same proportions in both sets.

# %%
from sklearn.model_selection import train_test_split

X = df_model[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print("Train:", X_train.shape, " Test:", X_test.shape)
print("Mean target - train:", round(y_train.mean(), 2), "| test:", round(y_test.mean(), 2))
print("Rows with no missing predictor at all (the 'drop incomplete rows' option):",
      int(X.notna().all(axis=1).sum()))

# %% [markdown]
# ### 3.2 Build the pre-processing recipe
# * **Numbers:** fill gaps with the **median** (robust to outliers), then **standardise** (mean 0, spread 1) so a variable measured 0-100 does not dominate one measured 1-5.
# * **Categories:** fill gaps with the label **"Unknown"**, then **one-hot encode** (each category becomes its own 0/1 column). `handle_unknown="ignore"` stops the model crashing if the test set contains a category the training set never saw.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_pipe = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    [("num", numeric_pipe, NUMERIC_FEATURES),
     ("cat", categorical_pipe, CATEGORICAL_FEATURES)],
    verbose_feature_names_out=False,
)
preprocessor.set_output(transform="pandas")

# %% [markdown]
# ### 3.3 Learn from the training set only, then apply to both

# %%
X_train_ready = preprocessor.fit_transform(X_train)   # learns medians / means / categories from TRAIN
X_test_ready = preprocessor.transform(X_test)         # only applies what was learned

print("Model-ready train:", X_train_ready.shape, "| test:", X_test_ready.shape)
print(X_train_ready.head())

# %% [markdown]
# ### 3.4 Check the result

# %%
print("Missing values left (train, test):", int(X_train_ready.isna().sum().sum()), int(X_test_ready.isna().sum().sum()))
print("All columns numeric:", all(pd.api.types.is_numeric_dtype(t) for t in X_train_ready.dtypes))

scaled = X_train_ready[NUMERIC_FEATURES].agg(["mean", "std"]).round(2)
print("\nScaled numeric columns on TRAIN (expect mean ~0, std ~1):")
print(scaled)
print("Columns that came from one-hot encoding:", X_train_ready.shape[1] - len(NUMERIC_FEATURES))

# %% [markdown]
# ### 3.5 Save everything the modeling teammate needs

# %%
import joblib

X_train_ready.to_csv(OUT_DIR / "03_X_train.csv", index=False)
X_test_ready.to_csv(OUT_DIR / "03_X_test.csv", index=False)
y_train.reset_index(drop=True).to_csv(OUT_DIR / "03_y_train.csv", index=False)
y_test.reset_index(drop=True).to_csv(OUT_DIR / "03_y_test.csv", index=False)
joblib.dump(preprocessor, OUT_DIR / "03_preprocessor.joblib")   # reuse on new data

log_step("3 preprocessing", "train rows", X_train_ready)
log_step("3 preprocessing", "test rows", X_test_ready)
pd.DataFrame(step_log).to_csv(OUT_DIR / "step_log.csv", index=False)
print(pd.DataFrame(step_log))

# %% [markdown]
# ## Reflection - Pre-processing
# | Question | Answer |
# |---|---|
# | **What problem did this solve?** | Linear regression and random forest need complete, numeric input. Imputation removes the blanks, one-hot encoding turns `country`, `organization_size` and `education_level` into 0/1 columns without implying an order, and scaling puts trust (0-100), tool count (0-12) and years of experience on a common scale so linear-regression coefficients can be compared. The test set is held out so the final score is honest. |
# | **What else did we consider?** | *Blanks:* drop incomplete rows (cleaner but loses data; row count is printed in 3.1 so the team can choose) vs. fill (keeps data; chosen). *Encoding:* label encoding (1, 2, 3...) rejected for unordered categories because it invents an order. *Scaling:* min-max scaling is an alternative; standardising is the usual choice for regression. A random forest does not need scaling, but one shared dataset keeps the comparison fair. *PCA:* skipped, because we only have 8 predictors and we want coefficients that are easy to explain. |
# | **What could go wrong if we skipped it?** | Models would crash on blanks and text. Without scaling, big-range variables would look more important than they are. Without splitting *first*, the test set would influence the fill/scale values (data leakage) and R-squared would be over-optimistic. |
#
# ---
# ## Hand-off to teammates
# | File in `data/processed/` | Use it for |
# |---|---|
# | `01_cleaned.csv` | evidence of the cleaning stage (all rows, all columns) |
# | `02_manipulated.csv` | **visualization** (readable values, 1 row per AI user, target + 8 predictors + `sentiment_label`) |
# | `02_summary_by_country.csv`, `02_summary_by_complex_rating.csv` | quick tables / charts for slides |
# | `03_X_train.csv`, `03_X_test.csv`, `03_y_train.csv`, `03_y_test.csv` | **modeling** (already imputed, encoded, scaled) |
# | `03_preprocessor.joblib` | apply the exact same recipe to new data |
# | `step_log.csv` | rows/columns after each step, for the slides |
