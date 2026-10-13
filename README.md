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
pip install -e ".[eda]"
pip install kaggle
```

Put your Kaggle API token in `%USERPROFILE%\.kaggle\access_token` (plain text, no file extension), then download the data:

```powershell
kaggle datasets download sgpjesus/bank-account-fraud-dataset-neurips-2022 -p data\raw\fraud_baf --unzip
kaggle competitions download home-credit-default-risk -p data\raw\credit_home_credit
kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv -p data\raw\aml_ibm
```

(Unzip the Home Credit archive, and repeat the last command for the other `*-Small_*` AML files.)
