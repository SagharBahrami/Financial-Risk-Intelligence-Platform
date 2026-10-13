"""Data validation rules for the ingested (interim) BAF dataset.

In BAF, missing values are encoded as negative numbers (-1 for counts), so those
columns allow -1 instead of NaN. Ranges follow the BAF datasheet and EDA.
"""
from pandera.pandas import Check, Column, DataFrameSchema

PAYMENT_TYPES = ["AA", "AB", "AC", "AD", "AE"]
EMPLOYMENT_STATUSES = ["CA", "CB", "CC", "CD", "CE", "CF", "CG"]
HOUSING_STATUSES = ["BA", "BB", "BC", "BD", "BE", "BF", "BG"]
SOURCES = ["INTERNET", "TELEAPP"]
DEVICE_OS = ["linux", "macintosh", "other", "windows", "x11"]

binary = Check.isin([0, 1])
count_or_missing = Check.ge(-1)  # -1 = missing
non_negative = Check.ge(0)

interim_schema = DataFrameSchema(
    {
        "fraud_bool": Column(int, binary),
        "income": Column(float, Check.in_range(0, 1)),
        "name_email_similarity": Column(float, Check.in_range(0, 1)),
        "prev_address_months_count": Column(int, count_or_missing),
        "current_address_months_count": Column(int, count_or_missing),
        "customer_age": Column(int, Check.isin(range(10, 100, 10))),
        "days_since_request": Column(float, non_negative),
        "intended_balcon_amount": Column(float),  # negative = missing
        "payment_type": Column("category", Check.isin(PAYMENT_TYPES)),
        "zip_count_4w": Column(int, non_negative),
        "velocity_6h": Column(float),  # slightly negative values exist in the source data
        "velocity_24h": Column(float, non_negative),
        "velocity_4w": Column(float, non_negative),
        "bank_branch_count_8w": Column(int, non_negative),
        "date_of_birth_distinct_emails_4w": Column(int, non_negative),
        "employment_status": Column("category", Check.isin(EMPLOYMENT_STATUSES)),
        "credit_risk_score": Column(int),  # can be negative
        "email_is_free": Column(int, binary),
        "housing_status": Column("category", Check.isin(HOUSING_STATUSES)),
        "phone_home_valid": Column(int, binary),
        "phone_mobile_valid": Column(int, binary),
        "bank_months_count": Column(int, count_or_missing),
        "has_other_cards": Column(int, binary),
        "proposed_credit_limit": Column(float, Check.gt(0)),
        "foreign_request": Column(int, binary),
        "source": Column("category", Check.isin(SOURCES)),
        "session_length_in_minutes": Column(float, count_or_missing),
        "device_os": Column("category", Check.isin(DEVICE_OS)),
        "keep_alive_session": Column(int, binary),
        "device_distinct_emails_8w": Column(int, count_or_missing),
        "month": Column(int, Check.in_range(0, 7)),
    },
    strict=True,  # fail on unexpected columns
    name="baf_interim",
)
