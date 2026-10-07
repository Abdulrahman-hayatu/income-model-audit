# income-model-audit

Ensemble learning, hyperparameter optimization, and explainability on the Adult Income dataset (predict whether income is above $50K).

This repo is my submission for this week's deliverable for the Flexisaf internship. The brief: build at least two models using the advanced machine learning techniques listed in the learning outcome. I built 8 models (5 in the main comparison, 3 fairness-audit variants), covering ensemble learning, hyperparameter optimization, and explainable AI.

## Models built

| # | Model | Technique | Notebooks |
|---|---|---|---|
| 1 | Logistic regression | Baseline, not an advanced technique | 01, 04 |
| 2 | Random forest | Ensemble learning (bagging) | 01, 04 |
| 3 | XGBoost, default parameters | Ensemble learning (boosting) | 01, 04 |
| 4 | Stacking (random forest + XGBoost → logistic regression) | Ensemble learning (stacking) | 01, 04 |
| 5 | XGBoost, tuned with Optuna | Ensemble learning + hyperparameter optimization | 02, 04 |
| 6 | Model 5's parameters, without `sex` and `race` | Fairness audit variant | 05 |
| 7 | Same, also without `relationship` | Fairness audit variant | 05 |
| 8 | Same, also without `marital-status` | Fairness audit variant | 05 |

Explainable AI (SHAP, permutation importance, LIME) is applied to model 5 in notebook 03 and adds no model. I counted distinct model configurations: cross-validation refits and Optuna's 40 candidate parameter sets are not counted separately.

## Data

Adult Income v2 from OpenML (`fetch_openml`, no manual download). 48,842 rows, 13 features after dropping `fnlwgt` (a census sampling weight), 23.9% positive. 80/20 stratified split with seed 42: 39,073 train rows, 9,769 test rows. Missing categorical values become a `"Missing"` category. The test set was used once, in notebook 04, for models that were already fixed. All tuning and selection used cross-validation on the training set.

## How to run

Tested with Python 3.12.3.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# quick checks from the repo root
python -m src.data
python -m src.models
python -m src.explain
```

Then run notebooks 01 to 05 in order. Notebook 02 takes roughly 5-15 minutes and notebook 04 a few minutes (stacking). Notebooks 03 to 05 read `results/best_xgb_params.json`, which notebook 02 writes and which is already committed.

## Results

### Cross-validation on the training set (5-fold)

| Model | ROC-AUC | PR-AUC | Fit time per fold (s) |
|---|---|---|---|
| Logistic regression | 0.9064 ± 0.0043 | 0.7653 ± 0.0123 | 0.5 |
| Random forest (untuned) | 0.8910 ± 0.0058 | 0.7339 ± 0.0120 | 17.5 |
| XGBoost (defaults) | 0.9242 ± 0.0030 | 0.8191 ± 0.0070 | 1.0 |
| Stacking | 0.9239 ± 0.0031 | 0.8185 ± 0.0071 | 110.7 |

The classes are 76/24, so I use ROC-AUC and PR-AUC as the main metrics (always predicting "<=50K" gets about 76% accuracy). Per-fold F1 is in `results/`.

### Hyperparameter optimization

![Optuna history](figures/optuna_history.png)

Optuna (TPE), 40 trials, 8 XGBoost parameters, scored by cross-validated ROC-AUC. The best score was reached by about trial 2 and improved by roughly 0.0004 after that, so more trials would probably not have helped. The best trial's score is biased upward (it is the maximum of 40 noisy evaluations), so I re-scored default and tuned parameters on fresh folds (seed 123, 5 folds × 3 repeats):

| | ROC-AUC |
|---|---|
| Default XGBoost | 0.9240 ± 0.0034 |
| Tuned XGBoost | 0.9293 ± 0.0035 |
| Paired difference | +0.0053 ± 0.0007 (tuned better in 15/15 folds) |

The folds are repeats of the same data, so the 15 comparisons are not independent. Best parameters: `results/best_xgb_params.json`.

### Final evaluation on the test set

| Model | ROC-AUC | 95% CI | PR-AUC | F1 (0.5) |
|---|---|---|---|---|
| Logistic regression | 0.9055 | [0.8994, 0.9120] | 0.7665 | 0.6633 |
| Random forest | 0.8943 | [0.8873, 0.9012] | 0.7502 | 0.6693 |
| XGBoost (defaults) | 0.9270 | [0.9215, 0.9326] | 0.8281 | 0.7132 |
| XGBoost (tuned) | 0.9307 | [0.9255, 0.9362] | 0.8350 | 0.7175 |
| Stacking | 0.9269 | [0.9214, 0.9325] | 0.8285 | 0.7113 |

Paired bootstrap on the same resampled test rows (ROC-AUC difference, 95% CI):
- Tuned XGBoost − default XGBoost: +0.0038 [0.0021, 0.0056]
- Default XGBoost − logistic regression: +0.0215 [0.0179, 0.0251]
- Stacking − default XGBoost: −0.0001 [−0.0006, 0.0004]

The intervals only reflect noise in the test sample, not variation from training or seeds.

- Model choice mattered more than tuning: XGBoost gained about 0.02 AUC over logistic regression, and tuning added about 0.004 (less than the +0.0053 seen in cross-validation).
- Stacking gave no measurable gain over XGBoost and took about 110 times longer to fit. I did not test why.
- The random forest finished below logistic regression, but it was untuned, so this says nothing about bagging in general.

### Explainability

All explanations use the tuned XGBoost trained on the full training set. SHAP values come from a 2,000-row sample of the training data, so they describe behavior on training rows, in log-odds. They show what the model uses, not what causes income.

![SHAP global importance](figures/shap_global_bar.png)

SHAP and permutation importance agree closely (rank correlation about 0.96). The top five features are the same under both: `marital-status`, `age`, `capital-gain`, `education-num`, `occupation`. `race`, `native-country`, and `education` have permutation AUC drops of 0.0009 or less, so their order is noise. Two things to keep in mind:
- `education` and `education-num` carry the same information, so credit is split between them.
- `sex` has a mean |SHAP| of 0.152 but only a 0.0022 permutation AUC drop. SHAP measures how far a feature moves the log-odds, and AUC only reacts when the ranking changes, so the two need not agree.

**One borderline case, SHAP vs LIME.** I explained the training row closest to P = 0.5 (row 28766, P = 0.4998).

![SHAP vs LIME](figures/shap_vs_lime_row.png)

- LIME (10 seeds) fits this row only moderately: R² 0.63-0.67, predicting 0.32-0.37 where the model says 0.50. I treat its weights as a rough guide. It is stable across seeds (`capital-gain` was the top feature in 10 of 10 runs).
- SHAP and LIME agree on the sign for 11 of 13 features (rank correlation 0.80).
- Biggest gap: LIME ranks `capital-gain` first by a wide margin, SHAP ranks it third and puts `marital-status` first. A possible reason is that 91.8% of rows have zero capital gain, so LIME's resampled neighborhood contrasts this row with the rare non-zero cases. I did not test this.
- This is one row. It shows how the methods behave on a single case, not which is better in general.

### Fairness audit

The dataset has `sex` and `race` columns, so I checked how the tuned XGBoost behaves across groups and whether dropping those columns helps. Setup: out-of-fold predictions on the training set, the tuned parameters reused for every variant (not re-tuned), a fixed 0.5 threshold, no test set. Intervals are 1,000 bootstrap resamples and only reflect noise in the evaluation rows. Everything is one model and one seed.

The labels already differ by sex: 30.4% of men and 10.9% of women have income above $50K (a gap of 0.195), so a selection gap alone is not the model's own bias.

| All-features model | n | base rate | selection rate | TPR | FPR | AUC |
|---|---|---|---|---|---|---|
| Male | 26,140 | 0.304 | 0.259 | 0.663 | 0.082 | 0.910 |
| Female | 12,933 | 0.109 | 0.081 | 0.582 | 0.019 | 0.949 |

The selection gap (0.178) is slightly below the label gap (0.195) in absolute terms, but men are selected 3.2 times as often as women against a base-rate ratio of 2.8. Women have a lower TPR and a lower FPR at the 0.5 threshold, and a higher AUC, so the TPR/FPR differences come from where the fixed threshold lands. I did not test other thresholds.

Gaps are male minus female. The last column is how well sex can be predicted from the remaining features (logistic regression, 5-fold ROC-AUC; a lower bound).

| Variant | ROC-AUC | Selection gap | TPR gap | FPR gap | Sex predictable (AUC) |
|---|---|---|---|---|---|
| All features | 0.9289 | +0.178 [+0.171, +0.185] | +0.080 [+0.053, +0.107] | +0.063 [+0.059, +0.068] | 0.929 (all but sex) |
| Drop sex, race | 0.9282 | +0.174 [+0.168, +0.181] | +0.064 [+0.036, +0.091] | +0.062 [+0.057, +0.066] | 0.929 |
| Also drop relationship | 0.9271 | +0.191 [+0.185, +0.198] | +0.138 [+0.110, +0.165] | +0.070 [+0.066, +0.075] | 0.857 |
| Also drop marital-status | 0.8877 | +0.109 [+0.102, +0.115] | +0.070 [+0.042, +0.098] | +0.010 [+0.006, +0.015] | 0.791 |

- **Dropping `sex` and `race` changes very little:** AUC falls 0.0007, the selection gap shrinks by 0.004 and the TPR gap by 0.016 (paired intervals exclude 0, but the changes are small).
- **The information stays in the data.** Sex is still predictable at 0.929 without `sex` and `race`, 0.857 without `relationship`, and 0.791 without `marital-status` as well. `marital-status` was dropped after `relationship`, so I did not measure it alone, and I did not check which remaining features carry the signal.
- **Removing `relationship` raised the TPR gap** (+0.058 against all features) even though sex became less predictable. I do not know why. The TPR gap across the four variants (0.080, 0.064, 0.138, 0.070) shows no trend.
- **Also removing `marital-status` shrinks the selection and FPR gaps but costs accuracy:** AUC drops 0.041 and F1 falls from 0.709 to 0.617. Smaller gaps from a clearly worse model are not evidence of a fairer one, and I did not check how much of the shrinkage is the model selecting fewer people overall.
- **Race** (all-features model only, no intervals, `results/fairness_by_race_all_features.csv`): TPR is 0.656 for White, 0.549 for Black, and 0.679 for Asian-Pac-Islander. Amer-Indian-Eskimo and Other have about 40 positives each, so I do not interpret them.

I describe the gaps and do not call the model fair or unfair. With different base rates, equal selection rates, TPRs and FPRs cannot all hold at once unless predictions are perfect. I tested only dropping columns, not mitigation such as reweighting or group-specific thresholds.

## A bug I found

My first SHAP results were wrong. XGBoost was trained on a sparse matrix, but my SHAP code densified it first, so SHAP explained a different function than the pipeline (probabilities differed by up to 0.82 on 1,093 of 2,000 rows). It showed up as `native-country` ranking third and a waterfall that contradicted the row's predicted probability. I fixed it by making the preprocessor output dense (`sparse_threshold=0`) and adding a check in `explain_shap` that raises if SHAP values do not reproduce `pipeline.predict_proba`. Re-running cross-validation for default and tuned XGBoost gave identical per-fold scores (differences of 1e-16), so the reported metrics were not affected. I did not work out why training was unaffected, so I claim no mechanism.

## Limitations

- Adult is 1994 census data, so the labels reflect the income patterns of that period.
- Only XGBoost was tuned. The random forest and stacking models use defaults.
- Explanations cover one local case and training rows. The fairness audit is one model, one seed, with no mitigation tested.

## Repo layout

```
income-model-audit/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_baseline_and_ensembles.ipynb
│   ├── 02_hyperparameter_optimization.ipynb
│   ├── 03_explainability_shap_lime.ipynb
│   ├── 04_final_test_evaluation.ipynb
│   └── 05_fairness_audit.ipynb
├── src/
│   ├── data.py          # loading and splitting
│   ├── models.py        # baseline and ensemble pipelines
│   ├── tuning.py        # Optuna search
│   ├── explain.py       # SHAP with a check against the pipeline
│   ├── lime_utils.py    # LIME bridge to the pipeline
│   └── fairness.py      # group metrics and proxy-leakage check
├── results/             # CSVs behind every number above
└── figures/
```

## Author

**Abdulrahman Hayatu Usman**
BSc Computer Science — Ahmadu Bello University, Zaria

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/abdulrahman-hayatu)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?logo=github&logoColor=white)](https://github.com/Abdulrahman-Hayatu)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?logo=gmail&logoColor=white)](mailto:hayatuusmanabdulrahman@gmail.com)
