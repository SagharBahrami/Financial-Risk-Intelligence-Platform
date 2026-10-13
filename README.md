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
| Features + time split | `python -m risk_platform.fraud.features` | `data/processed/fraud_{train,test}.parquet` |
| Train + evaluate | `python -m risk_platform.fraud.train` | MLflow run in experiment `fraud` |

- **Ingest** reads the raw CSV, drops constant columns, sets categorical types, validates the data with a
  [pandera](https://pandera.readthedocs.io/) schema (`src/risk_platform/fraud/schema.py`) and saves it as Parquet.
- **Features** turns negative "missing" codes into NaN plus a `<column>_missing` flag, then splits by time:
  months 0-5 for training, months 6-7 for testing.
- **Train** fits a scikit-learn pipeline and logs parameters, metrics and the model to MLflow.

Settings live in `configs/fraud.yaml`.

### Results (test set: months 6-7)

Metrics for imbalanced data: PR-AUC, and recall at 5% false-positive rate (as in the BAF paper).

| Model | PR-AUC | ROC-AUC | Recall @ 5% FPR |
|---|---|---|---|
| Logistic regression (baseline) | 0.164 | 0.883 | 0.507 |

## Experiment tracking

Runs are stored in a local SQLite database (`mlflow.db`). To browse them:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000.

## Tests

```powershell
pytest
```
