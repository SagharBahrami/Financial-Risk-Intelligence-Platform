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

| Model | Threshold | Recall | Precision | F2 | FPR on test |
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
- The threshold for 5% FPR on month 5 gives 5.6-5.9% FPR on the test months for the tree models.
  The threshold is chosen on one month, so it is noisy, and the false-positive rate also changes between the
  two test months. I have not shown that this is drift; production monitoring would have to check it.
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
