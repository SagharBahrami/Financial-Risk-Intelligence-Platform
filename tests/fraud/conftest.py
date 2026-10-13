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
