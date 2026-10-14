"""
01_data_prep.py - Data prep: Cleaning -> Manipulation -> Pre-processing   (VERSION 2)
Project: Predicting developer sentiment toward AI (target: AI_Sentiment_Score)

WHAT CHANGED IN VERSION 2
  Version 1 hand-picked 8 predictors (x) BEFORE any evidence, then threw every other column away.
  Version 2 keeps EVERY column that is a legitimate candidate predictor and hands the full set to the
  next teammate, who chooses x with a correlation analysis on the TRAINING set only.
  Columns are removed here ONLY for reasons about what the column IS (id, leakage, exact duplicate,
  constant, metadata) - never because of how well it correlates with y.

HOW TO RUN (VS Code)
  * Whole file : open the project FOLDER in VS Code, then press the Run button (or: python 01_data_prep.py)
  * Step by step: click "Run Cell" above any '# %%' line (needs the Python + Jupyter extensions)
Needs: data/raw/ai_developer_attitudes_2024.csv  (download from Kaggle; not stored in GitHub)
"""

# %% [markdown]
# # 01 - Data Prep: Cleaning -> Manipulation -> Pre-processing (v2)
# **Project:** Predicting developer sentiment toward AI (target `AI_Sentiment_Score`)
# **Dataset:** Global Developer Attitudes Toward AI 2024 (Kaggle, from the 2024 Stack Overflow Developer Survey)
#
# | Stage | Question it answers | Output |
# |---|---|---|
# | 1. Cleaning | Is the data **correct**? | `01_cleaning_report.csv` |
# | 2. Manipulation | Is the data in the **shape** we need? | `02_manipulated.csv`, `02_column_audit.csv` |
# | 3. Pre-processing | Is the data **ready for the algorithm**? | `03_X_train/test.csv`, `03_y_train/test.csv`, `03_feature_map.csv` |
#
# **The one rule that protects the whole project:** nothing in this file looks at how well a column predicts y.
# Choosing x by correlation is the NEXT teammate's job, and they must do it on the TRAINING rows only.

# %%
import numpy as np
import pandas as pd
from pathlib import Path

# ---- Settings (change here only) ----
try:
    ROOT = Path(__file__).resolve().parent
except NameError:                                # running as an interactive cell
    ROOT = Path.cwd()
RAW_PATH = ROOT / "data" / "raw" / "ai_developer_attitudes_2024.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42          # makes the train/test split repeatable for everyone
TARGET = "AI_Sentiment_Score"      # y
ID_COL = "respondent_id"

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 160)

step_log = []      # rows/columns after each step (for the slides)
def log_step(stage, step, frame):
    step_log.append({"stage": stage, "step": step, "rows": len(frame), "columns": frame.shape[1]})

report = []        # one line per problem we looked for: what, how many found, what we did
def log_check(stage, check, found, action):
    report.append({"stage": stage, "check": check, "rows_or_values_affected": int(found), "action": action})

# %% [markdown]
# ## 0. Load and first look
# We keep an untouched copy (`df_raw`) so we can always compare back.

# %%
df_raw = pd.read_csv(RAW_PATH)
df = df_raw.copy()
log_step("0 raw", "loaded raw file", df)
print("Shape (rows, columns):", df.shape)
print(df.dtypes.value_counts())
print("\nMissing values per column (count, only columns with gaps):")
miss = df.isna().sum()
print(miss[miss > 0].sort_values(ascending=False).to_string())

# %% [markdown]
# ---
# # STAGE 1 - DATA CLEANING
# **Goal: make the data *correct*.** Repair or remove what is wrong. No reshaping, no scaling/encoding.
# *(Handout analogy: washing the vegetables and throwing out the rotten ones.)*
#
# ### 1.1 Duplicates

# %%
n_full = int(df.duplicated().sum())
n_id = int(df[ID_COL].duplicated().sum())
print("Fully duplicated rows:", n_full, "| repeated respondent_id:", n_id)
df = df.drop_duplicates()
df = df.drop_duplicates(subset=ID_COL, keep="first")
log_check("1 cleaning", "duplicate rows / repeated respondent_id", n_full + n_id,
          "dropped (none found - data arrived clean on this point, but we checked)")
log_step("1 cleaning", "removed duplicates", df)

# %% [markdown]
# ### 1.2 The target (y): missing and impossible values
# A model cannot learn from a blank answer. The score may only be 0, 25, 50, 75 or 100.

# %%
print(df[TARGET].value_counts(dropna=False).sort_index())
n_missing_y = int(df[TARGET].isna().sum())
df = df[df[TARGET].notna()].copy()

VALID_SCORES = [0, 25, 50, 75, 100]
bad = ~df[TARGET].isin(VALID_SCORES)
n_bad_y = int(bad.sum())
df = df[~bad].copy()
log_check("1 cleaning", "target missing", n_missing_y, "rows dropped")
log_check("1 cleaning", "target not in {0,25,50,75,100}", n_bad_y, "rows dropped (none found)")
log_step("1 cleaning", "dropped rows with missing/invalid target", df)
print("Rows now:", len(df))

# %% [markdown]
# ### 1.3 Wrong data types
# * `developer_type`, `age_range`, `education_level`, `organization_size` are stored as **numbers** but they are
#   **category codes** with no code book in the Kaggle file (we do not know what code 3 means). Treating them as
#   numbers would be wrong, and we cannot label them, so they are **removed in stage 2** (see 2.3).
# * The TRUE/FALSE columns become 0/1 integers (models need numbers).
# * Numeric columns must really be numeric; stray spaces in `country` are removed.

# %%
NO_CODEBOOK = ["developer_type", "age_range", "education_level", "organization_size"]
for c in NO_CODEBOOK:
    print(f"{c:18s} unlabeled codes found:", sorted(int(v) for v in df[c].dropna().unique()))
log_check("1 cleaning", "category columns stored as unlabeled numeric codes (4 columns)", len(NO_CODEBOOK),
          "left as is here; removed in stage 2 (no code book)")

bool_cols = [c for c in df.columns if df[c].dtype == bool]
df[bool_cols] = df[bool_cols].astype(int)
log_check("1 cleaning", "TRUE/FALSE columns stored as bool", len(bool_cols), "converted to 0/1 integers")

NUMERIC_RAW = ["AI_Trust_Score", "ai_complex_task_concern", "AI_Benefit_Breadth", "AI_Tool_Count",
               "years_professional_coding_num", "AI_Job_Integration_Comfort"]
for c in NUMERIC_RAW:
    before = df[c].isna().sum()
    df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[c].isna().sum() != before:
        print(f"{c}: {df[c].isna().sum() - before} unparseable values turned into NaN")

df["country"] = df["country"].astype("string").str.strip().replace("", pd.NA).astype(object)
df["country"] = df["country"].where(df["country"].notna(), np.nan)
print("Distinct countries:", df["country"].nunique())

# %% [markdown]
# ### 1.4 Recode `ai_complex_task_concern` into its true order
# The raw numbers 1-5 are the **alphabetical order of the answer text**, not the order of meaning
# (Stack Overflow answer texts: `1 Bad` | `2 Good, but not great` | `3 Neither good nor poor` | `4 Very poor` | `5 Very well`).
# Recode to a real scale: **1 = very poor ... 5 = very well**.
#
# > **Why this recode is justified:** the answer wording. The average-sentiment table below is only a *sanity check*
# > that we did not make a typo - it must never be used to choose or tweak the mapping.
# > (The other score columns in this file - `AI_Trust_Score`, `AI_Job_Integration_Comfort` - are already
# > correctly ordered by the dataset's author; we verify that in stage 2.)

# %%
RECODE_COMPLEX = {1.0: 2, 2.0: 4, 3.0: 3, 4.0: 1, 5.0: 5}      # raw code -> true rating (1..5)
found = set(df["ai_complex_task_concern"].dropna().unique())
if not found.issubset(RECODE_COMPLEX):
    raise ValueError(f"Unexpected codes {found - set(RECODE_COMPLEX)} - check against the code book.")
n_recoded = int(df["ai_complex_task_concern"].notna().sum())
df["ai_complex_task_concern"] = df["ai_complex_task_concern"].map(RECODE_COMPLEX)
log_check("1 cleaning", "ai_complex_task_concern codes in alphabetical (wrong) order", n_recoded,
          "recoded to 1=very poor ... 5=very well")
print(df.groupby("ai_complex_task_concern")[TARGET].agg(["mean", "count"]).round(1))

# %% [markdown]
# ### 1.5 Impossible values (range checks)
# Out-of-range values are errors. We set them to *missing* (not delete the row) so the rest of that person's answers survive.

# %%
VALID_RANGES = {
    "AI_Trust_Score": (0, 100), "ai_complex_task_concern": (1, 5), "AI_Benefit_Breadth": (0, 7),
    "AI_Tool_Count": (0, 12), "years_professional_coding_num": (0, 60),
    "AI_Job_Integration_Comfort": (0, 100),
}
for col, (lo, hi) in VALID_RANGES.items():
    out = df[col].notna() & ~df[col].between(lo, hi)
    log_check("1 cleaning", f"{col} outside {lo}-{hi}", out.sum(), "set to missing (none found)" if out.sum() == 0 else "set to missing")
    df.loc[out, col] = np.nan
print(df[NUMERIC_RAW].describe().round(2))

# %% [markdown]
# ### 1.6 Internal consistency
# `AI_Tool_Count` / `AI_Benefit_Breadth` are meant to be the number of TRUE flags among the `AIToolCurrently Using - ...` /
# `AIBen - ...` columns. We recompute them from the flags (the flags are the source of truth).

# %%
tool_cols = [c for c in df.columns if c.startswith("AIToolCurrently Using - ")]
ben_cols = [c for c in df.columns if c.startswith("AIBen - ")]
print(len(tool_cols), "tool flags,", len(ben_cols), "benefit flags")
for target_col, cols in [("AI_Tool_Count", tool_cols), ("AI_Benefit_Breadth", ben_cols)]:
    recomputed = df[cols].sum(axis=1)
    mismatch = df[target_col].notna() & (df[target_col] != recomputed)
    log_check("1 cleaning", f"{target_col} disagrees with its TRUE/FALSE flags", mismatch.sum(),
              "recomputed from flags" if mismatch.sum() else "none found")
    df.loc[mismatch, target_col] = recomputed[mismatch]

# %% [markdown]
# ### 1.7 Missing values: REPORT only
# We do **not** fill gaps here. Filling uses a statistic (the median) that must be learned from the training rows only,
# so it belongs in pre-processing (stage 3).

# %%
df_clean = df.copy()
log_step("1 cleaning", "type/range/recode/consistency fixes (no rows dropped)", df_clean)
print("Cleaning done:", df_clean.shape)

# %% [markdown]
# ## Reflection - Cleaning
# | Question | Answer |
# |---|---|
# | **What problem did this solve?** | Removed rows with no target; put `ai_complex_task_concern` in its true order; confirmed no duplicates, no impossible values and that the two count columns match their flags. |
# | **What else did we consider?** | *Impossible values:* delete the row vs set to missing - we chose missing so we keep the person's other answers. *Missing predictors:* fill now vs later - later, because the median must come from training rows only. *Recode:* leave raw codes (a model would read "4" as more than "2") - rejected. |
# | **What could go wrong if skipped?** | A blank target breaks training; coded categories read as numbers give meaningless coefficients; a mis-ordered rating column can flip the sign of a coefficient and reverse the story. |

# %% [markdown]
# ---
# # STAGE 2 - DATA MANIPULATION
# **Goal: give the (now correct) data the *shape* our question needs.** Nothing is repaired here: filter, select, rename, group.
# *(Handout analogy: chopping and measuring.)*
#
# ### 2.1 Filter: keep only developers who currently use AI tools
# The trust and complex-task questions were only asked of current users. `ai_tool_stance` 3 = currently using.

# %%
print("Share who answered the trust question, by ai_tool_stance:")
print(df_clean.groupby("ai_tool_stance", dropna=False)["AI_Trust_Score"].apply(lambda s: round(s.notna().mean(), 3)).to_string())
df_users = df_clean[df_clean["ai_tool_stance"] == 3].copy()
log_step("2 manipulation", "filtered to current AI users (ai_tool_stance == 3)", df_users)
print("AI users kept:", len(df_users))

# %% [markdown]
# ### 2.2 Group countries: top 10 plus "Other"
# 185 countries would make ~185 near-empty one-hot columns. We keep the 10 most frequent. This uses only *how many people*
# are in each country, never their sentiment, so it does not leak information about y. Missing country stays missing (filled in stage 3).

# %%
top10 = df_users["country"].value_counts().head(10).index.tolist()
print("Top 10 countries:", top10)
orig = df_users["country"]
df_users["country"] = orig.where(orig.isin(top10), "Other").where(orig.notna())
print(df_users["country"].value_counts(dropna=False))

# %% [markdown]
# ### 2.3 Select columns: audit ALL 44 columns, with evidence
# We drop a column **only** if it is one of these (reasons about what the column *is*):
# `id` | `target` | `leakage` (made from y) | `duplicate` (same information as a column we keep) | `constant` (one value after filtering = nothing to learn) | `metadata` | `no_codebook` (numeric codes whose meaning we cannot look up, so we cannot interpret or report them).
# Every other column stays as a **candidate** (`candidate`) - the teammate decides with correlation on the training set.
#
# First we PROVE each claim instead of just asserting it.

# %%
def one_to_one(a, b, frame):
    """True if every value of a maps to exactly one value of b (and the reverse)."""
    pair = frame[[a, b]].dropna()
    return bool((pair.groupby(a)[b].nunique() == 1).all() and (pair.groupby(b)[a].nunique() == 1).all())

def many_to_one(a, b, frame):
    """True if every value of a maps to exactly one value of b (a is a recode of b)."""
    pair = frame[[a, b]].dropna()
    return bool((pair.groupby(a)[b].nunique() == 1).all())

u = df_users
evidence = {}
evidence["ai_overall_sentiment"] = many_to_one("ai_overall_sentiment", TARGET, u)
evidence["ai_accuracy_trust"] = one_to_one("ai_accuracy_trust", "AI_Trust_Score", u)
both = u[["Dependency_Signal", "AI_Trust_Score"]].dropna()
evidence["Dependency_Signal"] = bool(abs(both.corr().iloc[0, 1]) > 0.999)
evidence["years_professional_coding"] = bool((u["years_professional_coding"].fillna(-1) == u["years_professional_coding_num"].fillna(-1)).all())
evidence["ai_job_threat_concern"] = one_to_one("ai_job_threat_concern", "AI_Job_Integration_Comfort", u)
for c in ["ai_tool_stance", "AI_Adoption_Level", "AI_Adoption_Score",
          "data_source", "source_license", "source_url", "collection_year"]:
    evidence[c] = bool(u[c].nunique(dropna=False) == 1)
for k, v in evidence.items():
    print(f"{k:28s} claim verified: {v}")
assert all(evidence.values()), "A column-audit claim failed - re-check before dropping it."

# %%
AUDIT = {
    ID_COL: ("id", "Row label only; kept in the readable files so rows can be traced, never a predictor."),
    TARGET: ("target", "y - what we predict."),
    "ai_overall_sentiment": ("leakage", "The raw survey answer AI_Sentiment_Score is built from (every raw code maps to one score). Using it would hand the model the answer."),
    "ai_tool_stance": ("constant", "We filtered on it (=3), so it has a single value now."),
    "AI_Adoption_Level": ("constant", "Just a recode of ai_tool_stance -> single value after filtering."),
    "AI_Adoption_Score": ("constant", "Just a recode of ai_tool_stance -> single value after filtering."),
    "ai_accuracy_trust": ("duplicate", "Raw alphabetical codes of AI_Trust_Score (one-to-one). We keep the correctly ordered AI_Trust_Score."),
    "Dependency_Signal": ("duplicate", "Exactly 50 + AI_Trust_Score/2 (correlation 1.0 with AI_Trust_Score). Same information."),
    "years_professional_coding": ("duplicate", "Identical to years_professional_coding_num in every row."),
    "ai_job_threat_concern": ("duplicate", "Raw alphabetical codes of AI_Job_Integration_Comfort (one-to-one). We keep the ordered version."),
    "developer_type": ("no_codebook", "Numeric codes; the Kaggle file ships no code book, so we cannot say what the codes mean. Removed rather than guess."),
    "age_range": ("no_codebook", "Numeric codes; the Kaggle file ships no code book, so we cannot say what the codes mean. Removed rather than guess."),
    "education_level": ("no_codebook", "Numeric codes; the Kaggle file ships no code book, so we cannot say what the codes mean. Removed rather than guess."),
    "organization_size": ("no_codebook", "Numeric codes; the Kaggle file ships no code book, so we cannot say what the codes mean. Removed rather than guess."),
    "data_source": ("metadata", "Same value in every row."),
    "source_license": ("metadata", "Same value in every row."),
    "source_url": ("metadata", "Same value in every row."),
    "collection_year": ("metadata", "Same value in every row."),
}
NOTES = {
    "country": "top 10 + Other",
    "years_professional_coding_num": "years, number",
    "AI_Trust_Score": "0-100, ordered by the dataset author",
    "ai_complex_task_concern": "1 = very poor ... 5 = very well (recoded by us)",
    "AI_Job_Integration_Comfort": "10/50/100 = believes AI threatens job / unsure / does not (AI-opinion column; judgement call - see README)",
    "AI_Tool_Count": "= sum of the 12 tool_* flags (redundant with them)",
    "AI_Benefit_Breadth": "= sum of the 7 ben_* flags (redundant with them)",
}

def friendly(c):
    if c.startswith("AIBen - "):
        base, pre = c[len("AIBen - "):], "ben_"
    elif c.startswith("AIToolCurrently Using - "):
        base, pre = c[len("AIToolCurrently Using - "):], "tool_"
    else:
        return c
    base = base.replace("(please specify):", "").strip().lower()
    return pre + "".join(ch if ch.isalnum() else "_" for ch in base).strip("_").replace("__", "_")

RENAME = {c: friendly(c) for c in df_users.columns if c.startswith(("AIBen - ", "AIToolCurrently Using - "))}
rows = []
for c in df_raw.columns:
    if c in AUDIT:
        role, why = AUDIT[c]
    else:
        role, why = "candidate", NOTES.get(c, "0/1 flag (renamed to %s)" % RENAME.get(c, c))
    rows.append({"raw_column": c, "name_in_model_data": RENAME.get(c, c) if role in ("candidate", "id", "target") else "", "role": role, "reason": why})
audit = pd.DataFrame(rows)
audit.to_csv(OUT_DIR / "02_column_audit.csv", index=False)
print(audit["role"].value_counts().to_string())

# %% [markdown]
# ### 2.4 Apply the audit: keep ID + target + all candidates, rename the long flag names

# %%
keep_raw = [c for c in df_raw.columns if AUDIT.get(c, ("candidate",))[0] in ("id", "target", "candidate")]
df_model = df_users[keep_raw].rename(columns=RENAME).copy()
CANDIDATES = [c for c in df_model.columns if c not in (ID_COL, TARGET)]
print(f"Kept {len(keep_raw)} of {df_raw.shape[1]} columns: id + target + {len(CANDIDATES)} candidate predictors")
print(CANDIDATES)

# %% [markdown]
# ### 2.5 Derived column (for readable plots only)
# A text label for the 0/25/50/75/100 score. **Not** a model input. (Meanings follow the survey answer options; verify against the code book.
# Note the score 50 mixes "Indifferent" and "Unsure".)

# %%
SENTIMENT_LABELS = {0: "Very unfavourable", 25: "Unfavourable", 50: "Indifferent/unsure",
                    75: "Favourable", 100: "Very favourable"}
df_model["sentiment_label"] = df_model[TARGET].map(SENTIMENT_LABELS)
print(df_model["sentiment_label"].value_counts().reindex(list(SENTIMENT_LABELS.values())))

# Missing values among candidates (needed for the slides and for the fill decision in stage 3)
miss_report = (df_model[CANDIDATES].isna().mean() * 100).round(1)
miss_report = miss_report[miss_report > 0].sort_values(ascending=False)
print("\n% missing among AI users (candidates with gaps only):")
print(miss_report.to_string())
miss_report.rename("pct_missing").to_csv(OUT_DIR / "02_missing_report.csv")

# Descriptive summary table (whole dataset, evidence that the recode is sensible - NOT used to choose features)
df_model.groupby("ai_complex_task_concern")[TARGET].agg(avg_sentiment="mean", n_developers="count").round(1) \
    .to_csv(OUT_DIR / "02_summary_by_complex_rating.csv")

log_step("2 manipulation", "selected columns + grouped countries + renamed flags + label", df_model)
df_model.to_csv(OUT_DIR / "02_manipulated.csv", index=False)
print("Saved 02_manipulated.csv", df_model.shape, "| baseline (always predict the mean):", round(df_model[TARGET].mean(), 1))

# %% [markdown]
# ## Skipped on purpose (with reasons)
# * **Merging tables:** skipped - everything is in one table.
# * **Creating new predictors:** skipped - the dataset already contains the engineered scores/counts; adding more would be guesswork. (Only the plotting label was derived.)
# * **Removing outliers:** skipped - after the range checks the values are genuine survey answers.
# * **Choosing which predictors to keep by correlation:** deliberately NOT done here. It needs the training set only, so it is the next teammate's step.
#
# ## Reflection - Manipulation
# | Question | Answer |
# |---|---|
# | **What problem did this solve?** | Cut a 44-column survey of everyone to a table of current AI users with the target plus every legitimate candidate predictor, readable names and a manageable country column. Four category columns (developer_type, age_range, education_level, organization_size) were removed because the dataset has no code book, so their codes cannot be interpreted or reported. |
# | **What else did we consider?** | *Filter:* keep non-users and fill their trust/complex-task answers - rejected, those blanks are by design, filling would invent answers. *Unlabeled codes:* keep them as unordered categories vs remove - removed, because we cannot explain what the categories mean to the audience (if the Stack Overflow code book is found, they can be added back). *Columns:* hand-pick 8 (our v1) - rejected, it was a guess that skipped the evidence step; keep all and select on training data instead. *Countries:* all (too sparse) or top 5 (loses info) vs top 10 + Other. |
# | **What could go wrong if skipped?** | Non-users' invented answers; the raw answer ai_overall_sentiment would leak y and give a fake R-squared near 1; duplicate columns (e.g. Dependency_Signal = AI_Trust_Score) would double-count the same information. |

# %% [markdown]
# ---
# # STAGE 3 - PRE-PROCESSING
# **Goal: make the data *ready for the algorithm*.** Model-specific steps: split, fill, encode, scale.
# *(Handout analogy: the final prep right before it goes in the pan.)*
#
# ### 3.1 Split first (80% train / 20% test)
# **Order matters.** We split *before* filling or scaling, and the next teammate must do feature selection on the TRAIN rows only.
# If test rows influence ANY decision (a median, a mean, or which columns to keep) the final test score is flattering, not honest. This is **data leakage**.
# `stratify=y` keeps the five sentiment values in the same proportions in both sets.

# %%
from sklearn.model_selection import train_test_split

X = df_model[CANDIDATES]
y = df_model[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
print("Train:", X_train.shape, " Test:", X_test.shape)
print("Mean target - train:", round(y_train.mean(), 2), "| test:", round(y_test.mean(), 2))
print("Rows with no missing candidate at all (the 'drop incomplete rows' option):", int(X.notna().all(axis=1).sum()))

# %% [markdown]
# ### 3.2 Build the recipe: which column gets which treatment
# | Group | Columns | Fill gaps | Transform |
# |---|---|---|---|
# | Numbers | years, trust, complex-task rating, job-integration comfort, tool count, benefit breadth | **median** | **standardise** (mean 0, spread 1) |
# | 0/1 flags | `ben_*`, `tool_*` | none needed (no gaps) | none - already 0/1 |
# | Categories | country | label **"Unknown"** | **one-hot** (each level its own 0/1 column; no fake order) |

# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = ["years_professional_coding_num", "AI_Trust_Score", "ai_complex_task_concern",
                    "AI_Job_Integration_Comfort", "AI_Tool_Count", "AI_Benefit_Breadth"]
BINARY_FEATURES = [c for c in CANDIDATES if c.startswith(("ben_", "tool_"))]
CATEGORICAL_FEATURES = ["country"]
assert sorted(NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES) == sorted(CANDIDATES), "A candidate has no treatment!"

numeric_pipe = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
categorical_pipe = Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
                             ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
preprocessor = ColumnTransformer(
    [("num", numeric_pipe, NUMERIC_FEATURES),
     ("bin", "passthrough", BINARY_FEATURES),
     ("cat", categorical_pipe, CATEGORICAL_FEATURES)],
    verbose_feature_names_out=False)
preprocessor.set_output(transform="pandas")

# %% [markdown]
# ### 3.3 Learn from the training set only, then apply to both

# %%
X_train_ready = preprocessor.fit_transform(X_train)     # learns medians / means / categories from TRAIN
X_test_ready = preprocessor.transform(X_test)           # only applies what was learned
print("Model-ready train:", X_train_ready.shape, "| test:", X_test_ready.shape)

# %% [markdown]
# ### 3.4 Check the result (these checks must all pass)

# %%
assert X_train_ready.isna().sum().sum() == 0 and X_test_ready.isna().sum().sum() == 0, "gaps left"
assert list(X_train_ready.columns) == list(X_test_ready.columns), "train/test columns differ"
assert all(pd.api.types.is_numeric_dtype(t) for t in X_train_ready.dtypes), "non-numeric column left"
assert set(X_train.index).isdisjoint(X_test.index), "a row is in both train and test"
scaled = X_train_ready[NUMERIC_FEATURES].agg(["mean", "std"]).round(2)
assert (scaled.loc["mean"].abs() < 0.01).all() and ((scaled.loc["std"] - 1).abs() < 0.01).all()
print(scaled)

# feature map: which output column came from which original column
def source_of(col):
    if col in NUMERIC_FEATURES: return col, "numeric (scaled)"
    if col in BINARY_FEATURES: return col, "0/1 flag"
    for s in CATEGORICAL_FEATURES:
        if col.startswith(s + "_"): return s, "one-hot"
    raise ValueError(col)
fmap = pd.DataFrame([(c, *source_of(c)) for c in X_train_ready.columns], columns=["feature", "source_column", "kind"])
fmap.to_csv(OUT_DIR / "03_feature_map.csv", index=False)
# every original categorical column should be exactly one 1 per row after one-hot
for s in CATEGORICAL_FEATURES:
    cols = fmap.loc[fmap.source_column == s, "feature"]
    assert (X_train_ready[cols].sum(axis=1) == 1).all(), s
print("\nOutput columns by kind:\n", fmap["kind"].value_counts().to_string())
print("Total model-ready feature columns:", X_train_ready.shape[1])

# %% [markdown]
# ### 3.5 Save everything the next teammates need

# %%
X_train_ready.to_csv(OUT_DIR / "03_X_train.csv", index=False)
X_test_ready.to_csv(OUT_DIR / "03_X_test.csv", index=False)
y_train.reset_index(drop=True).to_csv(OUT_DIR / "03_y_train.csv", index=False)
y_test.reset_index(drop=True).to_csv(OUT_DIR / "03_y_test.csv", index=False)
# Same rows, same order, but readable (unscaled, with id + sentiment_label) -> for plots that help choose features
df_model.loc[X_train.index].reset_index(drop=True).to_csv(OUT_DIR / "03_train_readable.csv", index=False)
df_model.loc[X_test.index].reset_index(drop=True).to_csv(OUT_DIR / "03_test_readable.csv", index=False)

log_step("3 preprocessing", "train rows", X_train_ready)
log_step("3 preprocessing", "test rows", X_test_ready)
pd.DataFrame(step_log).to_csv(OUT_DIR / "step_log.csv", index=False)
pd.DataFrame(report).to_csv(OUT_DIR / "01_cleaning_report.csv", index=False)
print(pd.DataFrame(step_log).to_string(index=False))

# %% [markdown]
# ## Reflection - Pre-processing
# | Question | Answer |
# |---|---|
# | **What problem did this solve?** | Linear regression and random forest need complete numeric input. Median fill removes gaps; one-hot turns categories into 0/1 columns without implying an order; scaling puts 0-100 trust and 0-12 tool counts on one scale so coefficients are comparable; the held-out test set keeps the final score honest. |
# | **What else did we consider?** | *Gaps:* drop incomplete rows (loses ~19% of rows, count printed in 3.1) vs fill (chosen). *Encoding:* label 1,2,3.. rejected for unordered categories. *Scaling:* min-max is an alternative; standardising is usual for regression. *PCA:* skipped - it would hide which original columns matter, which is our question. *Where to select features:* before the split on all data (our v1 idea) - rejected, because it lets test rows steer the choice; after the split on train only (chosen). |
# | **What could go wrong if skipped?** | Models crash on blanks/text; big-range columns look more important than they are; splitting AFTER fill/scale/selection lets the test set leak in and R-squared is over-optimistic. |
#
# ---
# ## Hand-off to teammates (see README for details)
# | File in `data/processed/` | Use it for |
# |---|---|
# | `02_manipulated.csv` | **descriptive plots** (distribution of y, counts, missingness) - readable, ALL AI users |
# | `03_train_readable.csv` / `03_test_readable.csv` | readable copies of the same rows as `03_X_*` |
# | `03_X_train.csv`, `03_y_train.csv` | **feature selection** (correlation/heatmap) - TRAIN ONLY |
# | `03_X_test.csv`, `03_y_test.csv` | **do not touch** until the final model evaluation |
# | `03_feature_map.csv` | which model column came from which original column |
# | `02_column_audit.csv`, `01_cleaning_report.csv`, `step_log.csv`, `02_missing_report.csv` | evidence for the slides |
