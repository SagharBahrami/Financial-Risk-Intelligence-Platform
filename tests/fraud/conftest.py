import copy

import numpy as np
import pandas as pd
import pytest

from risk_platform.common.config import load_config


@pytest.fixture
def fraud_config() -> dict:
    return load_config("fraud")


@pytest.fixture
def raw_baf() -> pd.DataFrame:
    """Three rows with the same columns and types as data/raw/fraud_baf/Base.csv."""
    return pd.DataFrame(
        {
            "fraud_bool": [0, 1, 0],
            "income": [0.3, 0.8, 0.1],
            "name_email_similarity": [0.98, 0.12, 0.5],
            "prev_address_months_count": [-1, 12, 30],
            "current_address_months_count": [25, 89, -1],
            "customer_age": [40, 60, 20],
            "days_since_request": [0.006, 0.01, 5.7],
            "intended_balcon_amount": [102.4, -0.8, 47.1],
            "payment_type": ["AA", "AC", "AB"],
            "zip_count_4w": [1059, 1658, 2339],
            "velocity_6h": [13096.0, -50.2, 7601.5],
            "velocity_24h": [7850.9, 5745.2, 5124.0],
            "velocity_4w": [6742.0, 5941.6, 5940.7],
            "bank_branch_count_8w": [5, 3, 1],
            "date_of_birth_distinct_emails_4w": [5, 18, 6],
            "employment_status": ["CB", "CC", "CA"],
            "credit_risk_score": [163, -20, 91],
            "email_is_free": [1, 1, 0],
            "housing_status": ["BC", "BA", "BC"],
            "phone_home_valid": [0, 1, 1],
            "phone_mobile_valid": [1, 1, 0],
            "bank_months_count": [9, -1, 2],
            "has_other_cards": [0, 0, 1],
            "proposed_credit_limit": [1500.0, 200.0, 500.0],
            "foreign_request": [0, 1, 0],
            "source": ["INTERNET", "TELEAPP", "INTERNET"],
            "session_length_in_minutes": [16.2, 3.4, -1.0],
            "device_os": ["linux", "windows", "other"],
            "keep_alive_session": [1, 0, 1],
            "device_distinct_emails_8w": [1, 2, -1],
            "device_fraud_count": [0, 0, 0],
            "month": [0, 7, 3],
        }
    )


@pytest.fixture
def interim_baf(raw_baf, fraud_config) -> pd.DataFrame:
    from risk_platform.fraud.ingest import clean

    return clean(raw_baf, fraud_config["drop_columns"], fraud_config["categorical_columns"])


@pytest.fixture
def baf_files(raw_baf, fraud_config, tmp_path) -> dict:
    """Synthetic BAF data for months 0-7, saved as train and test files the way features.py does.

    Fraud is more likely for high income, so a model can learn something. Returns a copy of the
    fraud config that points to these files and uses small, fast models.
    """
    from risk_platform.fraud.features import time_split
    from risk_platform.fraud.ingest import clean

    rng = np.random.default_rng(0)
    rows_per_month = 200
    df = raw_baf.sample(8 * rows_per_month, replace=True, random_state=0).reset_index(drop=True)
    df["month"] = np.repeat(np.arange(8), rows_per_month)
    df["income"] = rng.uniform(0, 1, len(df))
    df["fraud_bool"] = (rng.uniform(size=len(df)) < 0.02 + 0.4 * df["income"] ** 4).astype(int)

    config = copy.deepcopy(fraud_config)
    df = clean(df, config["drop_columns"], config["categorical_columns"])
    train, test = time_split(df, config["time_column"], config["split"]["train_months"], config["split"]["test_months"])
    for name, part in [("train", train), ("test", test)]:
        path = tmp_path / f"fraud_{name}.parquet"
        part.to_parquet(path, index=False)
        config["data"][f"{name}_path"] = str(path)

    config["models"]["lightgbm"]["n_estimators"] = 20
    config["models"]["xgboost"]["n_estimators"] = 20
    config["tuning"]["report_every"] = 50
    return config
