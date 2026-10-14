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
| Tune (optional) | `python -m risk_platform.fraud.tune --model lightgbm` | best params, nested MLflow runs |
| Train + evaluate | `python -m risk_platform.fraud.train --model lightgbm` | MLflow run in experiment `fraud` |
| Evaluate vs. champion | `python -m risk_platform.fraud.evaluate --run-id <id>` | comparison run `evaluate_<model>` |
| Register | `python -m risk_platform.fraud.register --run-id <id>` | new version of `fraud-detector` in the Model Registry |

- **Ingest** reads the raw CSV, drops constant columns, sets categorical types, validates the data with a
  [pandera](https://pandera.readthedocs.io/) schema (`src/risk_platform/fraud/schema.py`) and saves it as Parquet.
- **Time split** uses months 0-5 for training and months 6-7 for testing.
- **Tune** searches hyperparameters with Optuna, training on months 0-4 and validating on month 5,
  so the test months stay unseen. It combines a TPE sampler (Bayesian optimization) with a Hyperband
  pruner: every 100 boosting rounds a trial reports its validation score, and clearly weaker trials are
  stopped early. Every trial, pruned or not, is logged as a nested MLflow run.
- **Train** fits a full scikit-learn pipeline and logs parameters, metrics, a precision-recall curve and the
  model to MLflow. Models: `logistic_regression`, `lightgbm`, `xgboost` (`src/risk_platform/fraud/models.py`).
- **Evaluate** loads a candidate model and the current champion the same way serving does, scores both on the
  test months and logs the comparison (metrics, deltas, overlaid PR curves) as its own MLflow run.
- **Register** registers a training run's model (or the best run, without `--run-id`) as a new version.
  It becomes `champion` only if it beats the current champion; otherwise it is marked `challenger`.

Each model is one pipeline from **raw application data** to a fraud probability: feature engineering
(negative "missing" codes → NaN + `<column>_missing` flags) and categorical encoding happen inside the
pipeline. Training and serving therefore run exactly the same code.

Settings live in `configs/fraud.yaml`.

### Results (test set: months 6-7)

Metrics for imbalanced data: PR-AUC, and recall at 5% false-positive rate (as in the BAF paper).
Random guessing gives a PR-AUC of about 0.014 (the fraud rate).

| Model | PR-AUC | ROC-AUC | Recall @ 5% FPR |
|---|---|---|---|
| Logistic regression (baseline) | 0.164 | 0.883 | 0.507 |
| LightGBM (default params) | 0.186 | 0.889 | **0.536** |
| XGBoost (default params) | 0.187 | 0.892 | 0.528 |

## Experiment tracking and model versioning

Runs are stored in a local SQLite database (`mlflow.db`). To browse them:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000.

Registered models live in the MLflow Model Registry. Every training, tuning and evaluation run is kept.

| Version | Model | Recall @ 5% FPR | Alias |
|---|---|---|---|
| v1 | Logistic regression (baseline) | 0.507 | |
| v2 | XGBoost | 0.528 | `champion` |

The alias `champion` always points to the version used in production, so serving code loads the model
without a version number:

```python
import mlflow

model = mlflow.pyfunc.load_model("models:/fraud-detector@champion")
proba = model.predict(applications_df)[:, 1]  # fraud probability per application
```

## Tests

```powershell
pytest
```
