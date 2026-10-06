# Figure notes: Visualization stage

Data: `data/processed/02_manipulated.csv` (37,536 current AI users). Code: `02_visualization.py`.
Target (y): `AI_Sentiment_Score` (0, 25, 50, 75, 100). All numbers below come from running the script.

## Ranked list for the slide (pick 3-4)

1. **fig_03_regplots.png**: shows the strength and the *shape* of every relationship. Best single chart.
2. **fig_02_correlation_heatmap.png**: ranks the predictors and shows which two overlap.
3. **fig_06_categorical_predictors.png**: tests the "background matters little" expectation.
4. **fig_01_target_distribution.png**: small, quick context for what we predict.

Backups (keep in the appendix, not on the slide): fig_04 pairplot (too busy for a slide), fig_05 boxplots, fig_00 missing values.

## Do the debrief's expectations hold?

| Expectation | Verdict | Evidence |
|---|---|---|
| More trust, more favourable | **Holds, strongest predictor** | r = 0.46. Average sentiment climbs 50 -> 65 -> 75 -> 82 -> 93 across trust levels 0 to 100 |
| Better complex-task rating, more favourable | **Holds** | r = 0.39. Average sentiment 57 (rating 1) to 91 (rating 5) |
| More benefits seen, more favourable | **Holds, but flattens** | r = 0.37. Rises from 53 to 88, then levels off at 6-7 benefits |
| More AI uses, more favourable | **Holds mostly, with a surprise** | r = 0.29. Rises overall, but dips at 0-1 uses (see Surprises) |
| Years of coding: no clear link | **Holds exactly** | r = 0.00, flat line |
| Country, organization size, education: small differences | **Mostly holds** | Org size and education: averages sit within about 70-76. Country is the exception: 68 (Poland) to 80 (Brazil), a 12-point spread |

## Surprises

1. **AI_Tool_Count dips at the low end.** Developers with 0 tools average 67.8, but those with 1 tool average 63.1. Then it rises steadily to about 89. So the relationship is not a straight line at the start. 6.6% of rows (2,480) have 0 tools even though the data is "current AI users". We do not know why, so we are not guessing. Worth asking the data-prep owner.
2. **Trust and complex-task rating overlap more than expected** (r = 0.56, the highest predictor-to-predictor value). They carry partly the same information.
3. **Country matters more than the other background columns.** Still small next to the spread of the target (standard deviation about 21), but not zero.

---

## fig_00_missing_values.png (backup)

**Execution:** horizontal bars of % missing per column. Code: `isna().mean()`.
**What it reveals:** `organization_size` is missing in 23.8% of rows and `years_professional_coding_num` in 17.2%. Trust (0.7%) and complex-task rating (1.4%) have almost none.
1. *Problem solved:* shows which charts use fewer rows (seaborn drops blanks plot by plot).
2. *Alternatives:* a table of percentages. A bar chart is faster to read.
3. *If skipped:* we might not notice that the years-of-coding chart uses 31,095 rows, not 37,536.

## fig_01_target_distribution.png

**Execution:** bar chart of the share of each score. Code: `value_counts(normalize=True)`.
**What it reveals:** 76.8% of developers score 75 or 100, only 5.7% score 0 or 25. The mean is 74.2 and the skew is -0.82 (tail to the left). This matches the debrief's numbers.
1. *Problem solved:* answers "is the target skewed, and does it need a transform?"
2. *Alternatives:* `histplot` or `kdeplot`. With only 5 values a bar chart is cleaner, and a smooth density curve would suggest values between the levels that do not exist. We rejected a log transform: the score includes 0 (log of 0 is undefined) and the skew comes from a ceiling at 100, not from a long tail.
3. *If skipped:* we would not know a "predict the average" baseline already scores 74, or that the model must beat it. We would also miss that predictions can drift above 100.

## fig_02_correlation_heatmap.png

**Execution:** `sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")` on the target and the 5 numeric predictors.
**What it reveals:** trust (0.46) > complex-task rating (0.39) > benefits (0.37) > tool count (0.29) > years of coding (0.00). The biggest overlap between two predictors is trust vs complex-task rating (0.56).
1. *Problem solved:* ranks predictors quickly and flags redundancy. Redundancy matters because linear regression gives each predictor its own coefficient, and overlapping predictors share credit.
2. *Alternatives:* Spearman correlation, which suits whole-number scales better. We checked it: same ranking (0.43, 0.38, 0.38, 0.30, 0.00), so Pearson is fine here.
3. *If skipped:* we would not know before modeling that trust and complex-task rating overlap, and could misread their individual coefficients.

## fig_03_regplots.png

**Execution:** `sns.regplot` of sentiment against each numeric predictor, with jitter so dots do not stack. Orange line is the straight-line fit. Dark dots are the real average sentiment at each predictor value (groups under 30 people skipped; years of coding is binned).
**What it reveals:** trust is almost perfectly straight. Complex-task rating is close to straight. Benefits flattens above 5. Tool count is straight above 1 but dips at 0-1. Years of coding is flat.
1. *Problem solved:* tests the assumption linear regression makes, that each relationship is a straight line.
2. *Alternatives:* `lmplot` (same idea, but we have no category to split by here) and plain scatter plots. Plain scatters of 5-level data are just stripes, so we added the dark average dots, which show the shape clearly.
3. *If skipped:* we would trust a straight line for tool count and benefits without knowing it fits worst at the ends, and could not explain the random forest's advantage if it appears.

Caution: benefits = 7 has only 49 people, so its small drop (86.2) is not reliable.

## fig_04_pairplot.png (backup)

**Execution:** `PairGrid` (no hue) on a 2,000-row sample, regression line in each scatter, histogram of the real values on the diagonal. Jitter is for display only.
**What it reveals:** the whole picture in one figure: the same rising lines for the four AI predictors, a flat line for years of coding, and the trust vs complex-task link.
1. *Problem solved:* every distribution and every pairwise trend at once, as the handbook describes.
2. *Alternatives:* `sns.pairplot(kind="reg")`. We built the grid by hand because the default histograms looked wrong on whole-number data.
3. *If skipped:* we would see relationships one at a time and might miss predictor-vs-predictor patterns.

Limit: a sample of 2,000, not all 37,536 rows, because the full set would be unreadable and slow.

## fig_05_boxplots.png (backup)

**Execution:** one boxplot per numeric predictor.
**What it reveals:** no impossible values (nothing below 0 or above the scale). Years of coding has a long upper tail (753 people above 30 years, maximum 51), which is plausible, not an error. The others have only a few mild outliers.
1. *Problem solved:* checks for extreme values left after cleaning.
2. *Alternatives:* `violinplot` (heavier to read) or counting values outside 1.5 x IQR (no picture).
3. *If skipped:* we would not know whether the long tail of years of coding is a data error or just experienced developers. Keep the values; consider scaling for the model.

## fig_06_categorical_predictors.png

**Execution:** horizontal bars of average sentiment per group with 95% confidence intervals and group sizes. Dashed line = overall mean (74.2). Missing answers are shown as their own orange bar.
**What it reveals:** country ranges from Poland (68.2) to Brazil (80.2). Organization size and education sit mostly within 70-76. People who skipped organization size average 74.3, about the same as everyone.
1. *Problem solved:* tests "do countries, organization sizes or education levels differ in sentiment?"
2. *Alternatives:* boxplots per group (cluttered with 10+ groups) and a table (less visual). Bars with error bars show both the gap and how certain it is.
3. *If skipped:* we could not back up the claim that background columns matter little, and would miss the country spread.

Cautions: `organization_size` and `education_level` are codes whose meanings are not in the dataset, so we do not guess them. The axis starts at 40, which makes differences look bigger than they are.

---

## Plots skipped (one line each)

- `residplot`: skipped, no model is fitted yet (belongs to the modeling teammate).
- `boxplot/violin by class`, `kdeplot hue=class`: skipped, these are classification plots and our target is a number.
- `lmplot` with `hue`: skipped, no categorical split needed; categories are covered in fig_06.
