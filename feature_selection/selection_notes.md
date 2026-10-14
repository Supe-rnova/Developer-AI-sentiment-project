# Feature selection notes

All numbers use the **training rows only** (`03_X_train.csv`, `03_y_train.csv`). The test files were never opened.

## Rules (fixed before looking at results)
- Keep a column if |r| with y is at least 0.1.
- If two kept columns overlap at |r| of 0.8 or more, drop the weaker one (pairs found: 0).
- Country (12 columns) is judged as one group: keep if it explains at least 0.01 of y.
- Counts vs flags: use the flags only if their 5-fold R2 beats the counts by at least 0.01.

## Results
- Columns passing the |r| cut: 19 of 37.
- Country as a group: R2 = 0.014 -> kept.
- Counts set (4 columns): 5-fold R2 = 0.305. Flags set (17 columns): 0.334. Decision: flags.

Ten strongest links with sentiment:
- AI_Trust_Score: r = +0.460 (numeric score or count)
- ai_complex_task_concern: r = +0.390 (numeric score or count)
- AI_Benefit_Breadth: r = +0.367 (numeric score or count)
- ben_increase_productivity: r = +0.296 (benefit flag)
- AI_Tool_Count: r = +0.291 (numeric score or count)
- ben_greater_efficiency: r = +0.249 (benefit flag)
- tool_writing_code: r = +0.227 (tool flag)
- ben_improve_accuracy_in_coding: r = +0.175 (benefit flag)
- ben_speed_up_learning: r = +0.171 (benefit flag)
- tool_debugging_and_getting_help: r = +0.169 (tool flag)

## Final x (29 columns)
- AI_Trust_Score
- ai_complex_task_concern
- ben_greater_efficiency
- ben_improve_accuracy_in_coding
- ben_improve_collaboration
- ben_increase_productivity
- ben_make_workload_more_manageable
- ben_speed_up_learning
- tool_committing_and_reviewing_code
- tool_debugging_and_getting_help
- tool_documenting_code
- tool_generating_content_or_synthetic_data
- tool_learning_about_a_codebase
- tool_project_planning
- tool_search_for_answers
- tool_testing_code
- tool_writing_code
- country_Brazil
- country_Canada
- country_France
- country_Germany
- country_India
- country_Netherlands
- country_Other
- country_Poland
- country_Ukraine
- country_United Kingdom of Great Britain and Northern Ireland
- country_United States of America
- country_Unknown

## Dropped (8 columns)
- years_professional_coding_num: |r| with y = 0.001, below the 0.1 cut
- AI_Job_Integration_Comfort: |r| with y = 0.008, below the 0.1 cut
- AI_Tool_Count: count replaced by its own flags (flags set 0.334 vs counts set 0.305 in 5-fold R2, gap at least 0.01)
- AI_Benefit_Breadth: count replaced by its own flags (flags set 0.334 vs counts set 0.305 in 5-fold R2, gap at least 0.01)
- ben_other: |r| with y = 0.092, below the 0.1 cut
- tool_deployment_and_monitoring: |r| with y = 0.078, below the 0.1 cut
- tool_other: |r| with y = 0.003, below the 0.1 cut
- tool_predictive_analytics: |r| with y = 0.067, below the 0.1 cut

## Notes per chart (problem solved / alternative considered / if skipped)
- **Target distribution** - Solves: shows whether y is skewed and what a 'guess the average' model scores. Alternative: histogram or density curve; with only 5 values a bar chart is clearer. If skipped: we would not know the baseline to beat.
- **Missing values** - Solves: shows which columns have gaps. Alternative: a table of percentages. If skipped: we might not notice that charts using years of coding or country use fewer rows.
- **Correlation with target (bar chart)** - Solves: ranks every candidate against y using training rows only. Alternatives: permutation importance or mutual information; correlation is the team's rule and the forest is a cross-check. If skipped: columns would be picked by gut feel.
- **Numeric heatmap with target** - Solves: shows links to y and overlap between the numeric x columns in one grid. Alternative: pairplot. If skipped: trust and complex-task rating overlapping would go unnoticed.
- **Tool and benefit flag heatmaps** - Solve: check that no two flags are near-copies. Alternative: a list of pairs above the threshold. If skipped: redundant flags could slip into the model.
- **Forest cross-check** - Solves: a second opinion on which columns matter, without using the test set. Alternative: linear coefficients. If skipped: the selection would rest on one method only.