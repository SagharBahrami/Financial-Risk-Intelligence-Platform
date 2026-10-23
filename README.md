# Financial Risk Intelligence Platform

An end-to-end MLOps project with three machine learning use cases that share one infrastructure
(MLflow, Docker, CI/CD, monitoring).

| Use case | Dataset | Folder |
|---|---|---|
| Fraud detection | [Bank Account Fraud (NeurIPS 2022)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) | `src/risk_platform/fraud` |
| Credit risk | [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) | `src/risk_platform/credit` |
| Anti-money laundering | [IBM Transactions for AML](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) (Small files) | `src/risk_platform/aml` |

## Project structure

```
├── api/                  # FastAPI service (one app, one endpoint per model)
├── configs/              # settings per use case
├── data/                 # not in git
│   ├── raw/              # original downloads, never edited
│   ├── interim/          # cleaned data
│   └── processed/        # features ready for training
├── docker/               # Dockerfiles and docker-compose
├── notebooks/            # exploration (EDA) per use case
├── reports/figures/      # saved charts
├── src/risk_platform/
│   ├── common/           # code shared by all use cases
│   ├── fraud/
│   ├── credit/
│   └── aml/
└── tests/
```

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[eda,dev]"
pip install kaggle
```

Put your Kaggle API token in `%USERPROFILE%\.kaggle\access_token` (plain text, no file extension), then download the data:

```powershell
kaggle datasets download sgpjesus/bank-account-fraud-dataset-neurips-2022 -p data\raw\fraud_baf --unzip
kaggle competitions download home-credit-default-risk -p data\raw\credit_home_credit
kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv -p data\raw\aml_ibm
```

(Unzip the Home Credit archive, and repeat the last command for the other `*-Small_*` AML files.)

## Pipeline

### Fraud detection

| Step | Command | Output |
|---|---|---|
| Ingest + validate | `python -m risk_platform.fraud.ingest` | `data/interim/fraud_baf.parquet` |
| Time split | `python -m risk_platform.fraud.features` | `data/processed/fraud_{train,test}.parquet` |
| Tune (optional) | `python -m risk_platform.fraud.tune --model lightgbm` | `best_params.json`, nested MLflow runs |
| Train + calibrate + test | `python -m risk_platform.fraud.train --model lightgbm`<br>or `--tune-run-id <id>` | MLflow run in experiment `fraud` |
| Evaluate vs. champion | `python -m risk_platform.fraud.evaluate --run-id <id>` | comparison run `evaluate_<model>` |
| Register | `python -m risk_platform.fraud.register --run-id <id>` | new version of `fraud-detector` in the Model Registry |

- **Ingest** reads the raw CSV, drops constant columns, sets categorical types, validates the data with a
  [pandera](https://pandera.readthedocs.io/) schema (`src/risk_platform/fraud/schema.py`) and saves it as Parquet.
- **Time split** uses months 0-5 for training and months 6-7 for testing. Inside the training months,
  every step gets its own months, so no choice is made on data that is later used to judge it:

  | Months | Used for |
  |---|---|
  | 0-3 | fit models while tuning |
  | 4 | choose hyperparameters |
  | 0-4 | refit the final model |
  | 5 | fit the probability calibrator and choose the decision threshold |
  | 6-7 | final test of the frozen model only |

- **Tune** searches hyperparameters with Optuna, training on months 0-3 and validating on month 4,
  so the calibration month and the test months stay unseen. It combines a TPE sampler (Bayesian optimization) with a Hyperband
  pruner: every 100 boosting rounds a trial reports its validation score, and clearly weaker trials are
  stopped early. Every trial, pruned or not, is logged as a nested MLflow run. The best parameters are
  saved as `best_params.json`, and `train --tune-run-id <id>` refits the model with them.
- **Train** fits a full scikit-learn pipeline on months 0-4 and then freezes it. On month 5 it fits a
  probability calibrator and picks the decision threshold on the calibrated probabilities: the highest recall
  with at most 5% false positives. No calibration, sigmoid (Platt scaling) and isotonic calibration are compared by
  5-fold cross-validation on month 5, so the method can be chosen without looking at the test months
  (`training.calibration_method: auto` in the config). Auto takes the lowest Brier score among the methods that
  lose at most 0.005 recall at 5% FPR compared with no calibration (`training.max_recall_loss`), because isotonic
  calibration gives slightly better probabilities but can lose ranking quality through ties. A method with a
  negative Brier skill score is never chosen while another method beats the base rate. The frozen model is then scored once on
  months 6-7. MLflow gets the parameters, metrics before and after calibration, a precision-recall curve,
  a calibration curve and the model. Models: `logistic_regression`, `lightgbm`, `xgboost`
  (`src/risk_platform/fraud/models.py`).
- **Evaluate** loads a candidate model and the current champion the same way serving does, scores both on the
  test months and logs the comparison (metrics, deltas, overlaid PR curves) as its own MLflow run.
- **Register** registers a training run's model (or the best run, without `--run-id`) as a new version.
  It becomes `champion` only if it beats the current champion; otherwise it is marked `challenger`.

Each model is one pipeline from **raw application data** to a fraud probability: feature engineering
(negative "missing" codes → NaN + `<column>_missing` flags) and categorical encoding happen inside the
pipeline. Training and serving therefore run exactly the same code.

The logged model contains the pipeline, the calibrator and the decision threshold:
`predict_proba` returns the calibrated fraud probability and `predict` returns the 0/1 decision.
The threshold is also stored in the model metadata (`decision_threshold`).

Settings live in `configs/fraud.yaml`.

### Results (test set: months 6-7)

Metrics for imbalanced data: PR-AUC, and recall at 5% false-positive rate (as in the BAF paper).
Random guessing gives a PR-AUC of about 0.014 (the fraud rate). Brier skill score (BSS) compares the
Brier score with always predicting the fraud rate: 0 = no better, below 0 = worse.

Default hyperparameters, fit on months 0-4, calibration method chosen automatically on month 5
(sigmoid for logistic regression and LightGBM, none for XGBoost) and threshold from month 5:

| Model | PR-AUC | ROC-AUC | Recall @ 5% FPR | Brier (before → after) | BSS (before → after) |
|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.160 | 0.881 | 0.500 | 0.0970 → 0.0127 | -6.01 → 0.08 |
| LightGBM | 0.177 | 0.888 | **0.530** | 0.0127 → 0.0127 | 0.08 → 0.08 |
| XGBoost (no calibration) | 0.183 | 0.887 | 0.527 | 0.0126 → 0.0126 | 0.09 → 0.09 |

Decisions with the threshold chosen on month 5:

| Model | Threshold | Recall | Precision | F2 | FPR on test (months 6-7 pooled) |
|---|---|---|---|---|---|
| Logistic regression | 0.046 | 0.483 | 0.132 | 0.315 | 4.5% |
| LightGBM | 0.041 | 0.561 | 0.119 | 0.322 | 5.9% |
| XGBoost | 0.030 | 0.554 | 0.123 | 0.326 | 5.6% |

What I learned from these runs:
- Class weights (`class_weight: balanced`) made logistic regression's probabilities far too high
  (BSS -6). Sigmoid calibration fixed this without changing the ranking.
- LightGBM and XGBoost were already close to calibrated. Sigmoid calibration made XGBoost worse;
  the cross-validation on month 5 shows the same (isotonic 0.01057, uncalibrated 0.01060,
  sigmoid 0.01113), so this can be seen without the test months.
- Isotonic calibration had the best Brier score for all three models on month 5, but the gain was tiny
  (Brier skill 0.0006-0.0025 better than the next method) and it lost 0.85-1.1 points of recall at 5% FPR and
  about 0.01 PR-AUC out-of-fold, because it turns different scores into ties. That is why auto selection
  also checks the ranking instead of using the Brier score alone.
- The threshold for 5% FPR on month 5 gives 5.6-5.9% FPR on the two test months pooled for the tree models.
  For XGBoost this is an average of two different months: 6.65% in month 6 and 4.49% in month 7. Choosing the
  threshold adds almost no optimism inside month 5 (see the protocol below); the FPR moves because the score level
  of the later months differs. The monitoring section below looks at this month by month.
- Training on months 0-4 instead of 0-5 cost little: recall at 5% FPR went from 0.536 to 0.530 (LightGBM).

### Experiment: class weighting for LightGBM and XGBoost (not adopted)

I tested whether weighting the fraud class (`scale_pos_weight`) improves the boosted models.
The baseline is the pipeline above with no weighting. Only `scale_pos_weight` changed: 1 (baseline),
about 10 (square root of negatives/positives) and about 99 (full balancing, negatives/positives on months 0-4).
Data, months, hyperparameters, seed and threshold rule were the same. The runs are in the MLflow
experiment `fraud_class_weight` and can be repeated with `python -m risk_platform.fraud.class_weight_experiment`.

Recall @ 5% FPR before calibration, so calibration cannot affect it:

| Model | Weight | Month 5 | Test (months 6-7) |
|---|---|---|---|
| LightGBM | 1 (baseline) | **0.531** | **0.530** |
| LightGBM | 10 | 0.519 | 0.520 |
| LightGBM | 99 | 0.516 | 0.509 |
| XGBoost | 1 (baseline) | **0.534** | **0.527** |
| XGBoost | 10 | 0.528 | 0.522 |
| XGBoost | 99 | 0.520 | 0.484 |

- Ranking got worse as the weight grew, for both models, on month 5 and on the test months.
  ROC-AUC went down too. PR-AUC on month 5 was slightly higher at weight 10, but this did not carry over to recall.
- Raw Brier score got much worse (0.011 at weight 1, 0.017 at weight 10, 0.07 at weight 99). After calibration
  it was back to about 0.0106-0.0108, the same as the baseline, so weighting gained nothing for the probabilities.
- Weighting changed which calibration method the Brier-only rule picked (LightGBM chose sigmoid instead of isotonic).
  This makes the recall of the deployed models look better for LightGBM at weight 10, but it comes from the calibration
  method, not from the weighting. Comparing the uncalibrated ranking removes this.
- Limits: one seed, one calibration month, and I did not retune the hyperparameters for the weighted models.
  Differences of about 1 point are close to noise, but the direction was the same for both models and both datasets.

Decision: I keep the unweighted pipeline. Weighting adds a parameter and distorts the raw probabilities
without improving the ranking.

### Evaluation protocol and frozen final model

Rules I follow from now on:
1. Months 6-7 stay as an inspected holdout. I report their results and do not use them to choose between models,
   calibration methods, class weights or thresholds.
2. Every choice (model, calibration method, threshold, champion) uses month 5, with the same rule each time.
3. If new data becomes available, it is used once, for a final independent assessment.

How optimistic is the month-5 selection? The pipelines are frozen (fit on months 0-4), so only the threshold is
chosen on month 5. I chose it on 4/5 of month 5 and applied it to the other 1/5 (5-fold, repeated 5 times):

| Model | FPR in-sample | FPR out-of-fold | Recall in-sample | Recall out-of-fold (sd) |
|---|---|---|---|---|
| XGBoost | 4.97% | 4.98% | 0.534 | 0.534 (0.029) |
| LightGBM | 5.00% | 4.98% | 0.531 | 0.529 (0.028) |
| Logistic regression | 4.96% | 4.97% | 0.510 | 0.510 (0.023) |

Choosing the threshold on month 5 therefore adds almost no optimism inside month 5 (about 0.1 points of recall).
The FPR of 5.6-5.9% on months 6-7 is not caused by this; it comes from the score level of months 6 and 7 being
different from month 5. I looked at that month by month in the next section.

Model comparison on month 5 (paired bootstrap, 300 resamples of the rows, recall at 5% FPR):
- XGBoost - LightGBM: +0.004, 95% interval [-0.009, +0.016]. This is a tie; XGBoost was better in 71% of resamples.
  XGBoost is champion because of the selection rule, not because it is clearly better.
- XGBoost and LightGBM - logistic regression: about +0.02, intervals above zero. The tree models are better.

Frozen final model (registry `fraud-detector` v3, alias `champion`): XGBoost with the default hyperparameters in
`configs/fraud.yaml`, fit on months 0-4, no calibration (chosen on month 5), decision threshold 0.0297 (saved in the
model metadata), run `0b4ff926f3a14a08b30e14cd5b7e09c1`. I do not change it unless new validation data justifies a
new version; a retrained model is a new version and goes through the same month-5 rule.

### Monitoring: month-to-month changes at the frozen threshold

I checked how the frozen XGBoost v3 behaves in months 5, 6 and 7 with
`python -m risk_platform.fraud.monitor --swap` (code in `src/risk_platform/common/drift.py` and
`src/risk_platform/fraud/monitor.py`, tests in `tests/common/test_drift.py` and `tests/fraud/test_fraud_monitor.py`).
It only reads the champion and its threshold (0.0297). It retrains nothing, changes no threshold or registry entry,
and the months are used for reporting only, never to choose anything. Month 5 is the month the threshold was chosen
on, so its FPR is close to 5% by construction and is not an independent benchmark.

I keep three kinds of change apart, because they have different causes and different consequences:
- **Performance change**: what the decisions do (FPR, recall, precision). This needs labels.
- **Score drift**: the distribution of the model's scores. I report it twice. For **all applications** it needs no
  labels, so it is the version that can be watched in production; the share above the threshold is then the flag
  rate. For **legitimate applications only** it needs the labels (the outcomes are known late), and the share above
  the threshold is the FPR. FPR can only change through the scores of legitimate applications (or through the
  labels), so this version links performance and score drift.
- **Input drift**: the distribution of the features of all applications. It needs no labels either.

The subgroup analysis and the feature-swap diagnostic use the legitimate applications, so they need labels.

**Performance** (decisions at the threshold 0.0297; the 95% Wilson intervals assume independent rows, see below):

| Month | Legit | Fraud | Fraud rate | FPR | Recall | Precision | ROC-AUC | Recall @ 5% FPR (best threshold of that month) |
|---|---|---|---|---|---|---|---|---|
| 5 (threshold chosen here) | 117,912 | 1,411 | 1.18% | 4.97% [4.85, 5.09] | 53.4% | 11.4% | 0.893 | 53.4% |
| 6 | 106,718 | 1,450 | 1.34% | **6.65%** [6.50, 6.80] | 57.9% | 10.6% | 0.889 | 51.0% |
| 7 | 95,415 | 1,428 | 1.47% | **4.49%** [4.36, 4.62] | 52.9% | 15.0% | 0.889 | 55.5% |

Pooled over months 6-7 the FPR is 5.63%, which hides a swing of 2.2 points between the two months. The ranking of
the model (ROC-AUC, recall at the best 5% FPR threshold) stays similar. The decisions change because the same
threshold is applied to a different score level.

| FPR difference | Size | 95% interval | Variance inflation it survives |
|---|---|---|---|
| Month 6 - month 5 | +1.68 points | [+1.49, +1.88] | 75 times |
| Month 7 - month 5 | -0.48 points | [-0.66, -0.30] | 7 times |
| Month 7 - month 6 | -2.16 points | [-2.36, -1.97] | 118 times |

"Variance inflation it survives" is the factor by which the true variance would have to exceed the independent-rows
variance before the 95% interval reaches zero, which is (z / 1.96)^2. BAF has no dates or customer ids, so I could
not do a day-level or customer-level bootstrap, and applications from the same customer or the same burst of
traffic are certainly not independent. I do not know the real inflation. The option `--design-effect` (default 1,
independent rows) multiplies the variance of every probability statement in the report in the same way: the monthly
Wilson intervals, the FPR difference intervals and the subgroup tests. It does not change PSI or KS, which are not
intervals, and it does not change the "variance inflation it survives" column, which is defined from independent
rows. The tables here use the default. The month 6 differences are large enough to survive a lot of inflation. The difference between month 7 and month 5 survives only about 7 times, so I treat it as weak
evidence; the difference between month 6 and month 7 is the clearest.

**Score drift of all applications** (no labels needed; against month 5). Two random halves of month 5 give KS 0.0083
and PSI 0.00062 by chance.

| Month | Flag rate (share above the threshold) | Threshold sits at percentile | KS | PSI |
|---|---|---|---|---|
| 5 | 5.54% | 94.5 | - | - |
| 6 | 7.34% | 92.7 | 0.045 | 0.0120 |
| 7 | 5.20% | 94.8 | 0.047 | 0.0131 |

**Score drift of legitimate applications** (needs labels; against month 5). Two random halves of month 5 give KS
0.0076 and PSI 0.00056 by chance.

| Month | Share above the threshold (= FPR) | Threshold sits at percentile | KS | PSI | Median score of fraud |
|---|---|---|---|---|---|
| 5 | 4.97% | 95.0 | - | - | 0.0348 |
| 6 | 6.65% | 93.3 | 0.044 | 0.0119 | 0.0417 |
| 7 | 4.49% | 95.5 | 0.049 | 0.0137 | 0.0342 |

The scores are higher in month 6 and lower in month 7, for legitimate applications and for fraud. The label-free
version shows the same pattern as the legitimate-only version: the flag rate is 5.54%, 7.34% and 5.20%, and the PSI
is almost the same, so the movement can be seen without labels. The flag rate is higher than the FPR because the
fraud cases score higher, so the two are not the same number. By the usual rule of thumb (PSI below 0.1 is small) the
shifts are small, but the PSI is about 20 times the chance level. They matter for the decisions because the threshold
sits in the tail, where a small shift changes the share above it a lot.

**Input drift** (all applications, no labels needed; PSI against month 5; the chance level is below 0.005 for every
feature):

| Feature | Month 6 | Month 7 |
|---|---|---|
| velocity_4w | 0.55 | 4.21 |
| velocity_24h | 0.31 | 1.27 |
| velocity_6h | 0.16 | 0.39 |
| zip_count_4w | 0.03 | 0.49 |
| current_address_months_count | 0.04 | 0.24 |

The velocity features fall month after month: the mean of velocity_4w is 6189 in month 0, 4285 in month 5, 4119 in
month 6 and 3344 in month 7. In month 7, 91% of applications are below the 5th percentile of
velocity_4w in the training months 0-4 (48% in month 5). The values are still inside the training range (fewer than
0.02% are outside), so this is a trend in the inputs, not values the model has never seen. The share of missing
`prev_address_months_count` also falls from 83% to 75% to 65%. Preprocessing and prediction behave the same in all
months: the same columns and types, no unseen categories, and the scores do not depend on row order or batch size.

**Subgroups.** The FPR rose in nearly every subgroup in month 6 and fell in nearly every subgroup in month 7, so
this is a change of the score level, not of who applies. Splitting the change into composition and within-group
parts, composition explains little: the largest part is housing status in month 6 (+0.57 of the +1.68 points,
because group BA, which has an FPR of 20%, grew from 15.7% to 18.6% of applications). The monitoring script also
lists the subgroups that changed differently from the rest of the population; with 100,000 rows per month, many
small differences are statistically significant, so I read the sizes and rate ratios, not the p-values.

**Feature-swap diagnostic** (legitimate applications, so it needs labels). For each feature (or group of features) I
replaced its values in month 6 (or 7) with draws from month 5 and measured the change of the FPR in percentage points;
positive means the month's values go with a higher FPR. The draw is repeated five times with fixed seeds (0 to 4),
so the table is reproducible, and the numbers below are the means. The columns of a group (the three velocity
features) are drawn together from the same month 5 rows, so the relationships inside the group are kept; only their
link to the other features is broken.

| Feature | Month 6 | Month 7 |
|---|---|---|
| velocity features together | +0.8 | +0.2 |
| velocity_4w alone | +0.7 | -0.2 |
| prev_address_months_count | -0.4 | -0.6 |
| zip_count_4w | -0.1 | -0.5 |
| phone_home_valid | +0.4 | +0.1 |

The total changes are +1.68 and -0.48 points. This is a diagnostic association, not proof of what caused the change:
it changes one feature at a time on modified data with the other features left as they are, the effects do not add
up to the total, features that move together (the three velocity features) are only meaningful as a group, and
replacing a feature by random draws breaks its link to the features outside its group. The standard deviation over
the five draws is 0.01 to 0.06 points, so only the large and consistent effects mean anything.

What I conclude, and what I do not:
- The FPR at the frozen threshold changed from month to month (4.97%, 6.65%, 4.49%). The month 6 change and the
  difference between month 6 and 7 are not sampling noise. The change from month 5 to month 7 is weaker evidence.
- The score level of legitimate applications moved with the inputs, while the ranking quality stayed similar.
  Input drift exists (the velocity features have a steady trend), but the FPR went up and then down, so I do not
  claim that drift is steadily pushing the FPR up. Different features push in different directions in months 6 and 7.
- I cannot separate normal month-to-month variation from drift with three months. The training months cannot serve as
  a baseline because the model was fit on them, and BAF has no later months.
- A fixed score threshold does not keep a fixed FPR. In production I would watch the flag rate (the share of all
  applications above the threshold), the score PSI of all applications and the feature PSI every month; none of them
  needs labels. The FPR, the legitimate-only score drift and the subgroup analysis need the outcomes, so they can only
  be checked later. Then I would decide about recalibrating the threshold with the same month-based rule as before.
  I have not changed anything.

Run it with (it writes a Markdown report and CSV files to `reports/monitoring/fraud`, which is not tracked by git):

```powershell
python -m risk_platform.fraud.monitor --reference-month 5 --months 6 7 --swap
```

## Experiment tracking and model versioning

Runs are stored in a local SQLite database (`mlflow.db`). To browse them:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000.

Registered models live in the MLflow Model Registry. Every training, tuning and evaluation run is kept.

| Version | Model | Selected on | Recall @ 5% FPR | Alias |
|---|---|---|---|---|
| v1 | Logistic regression (baseline) | test months 6-7 | 0.507 (test) | |
| v2 | XGBoost, fit on months 0-5 | test months 6-7 | 0.528 (test) | historical |
| v3 | XGBoost, fit on months 0-4, no calibration | month 5 | 0.534 (month 5) | `champion` |

The two metrics are not comparable: v1 and v2 show recall on the test months, v3 on month 5.

Why the champion changed (2026-10-19): v2 was chosen with the test months, and month 5 was part of its fitting
data, so it cannot be compared on month 5. v3 follows the updated evaluation protocol: fit on months 0-4,
calibration method and decision threshold (0.0297) chosen on month 5, and selected by recall at 5% FPR on month 5.
The old champion had no month-5 metric, so I promoted v3 by hand (`register --promote`). v1 and v2 stay in the
registry as history, and the reason is written in their descriptions and tags.
XGBoost won on month 5 by only 0.36 points over LightGBM (0.5344 vs 0.5308), so this is a protocol-based choice,
not strong evidence that it is better. Before promoting, I checked the saved threshold, the calibration method
and the serving behavior: the metadata matches the run, the model loads through `mlflow.pyfunc`, JSON-style
input gives identical scores, and the saved threshold gives 4.97% false positives on month 5 (the month it was
chosen on, so this checks the behavior, not the performance).

No model was fitted on the test months 6-7, so there is no leakage into training. But I reused them: they were used
to compare and choose v1 and v2, and what I saw there influenced later design decisions (test-set reuse, a form of
selection bias). Their results are therefore somewhat optimistic and not an independent estimate of future
performance. Month 5 has the same problem on a smaller scale, because it was used to choose the calibration method,
the threshold and the champion. A stronger assessment needs new data from after month 7 or another dataset, which
the BAF files do not contain.

The alias `champion` always points to the version used in production, so serving code loads the model
without a version number:

```python
import mlflow

model = mlflow.pyfunc.load_model("models:/fraud-detector@champion")
proba = model.predict(applications_df)[:, 1]  # fraud probability per application (calibration method: see metadata)
is_fraud = proba >= model.metadata.metadata["decision_threshold"]
```

(Versions v1 and v2 were trained before calibration was added and have no `decision_threshold`;
the champion, v3, carries `decision_threshold` and `calibration_method` in its metadata.)

## Tests

```powershell
pytest
```
