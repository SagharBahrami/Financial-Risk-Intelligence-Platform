import numpy as np
import pandera.pandas as pa
import pytest

from risk_platform.fraud.schema import interim_schema


def test_valid_data_passes(interim_baf):
    interim_schema.validate(interim_baf, lazy=True)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("fraud_bool", 2),
        ("month", 8),
        ("customer_age", 35),
        ("income", 1.5),
        ("bank_months_count", -5),
        ("income", np.nan),
    ],
)
def test_invalid_value_fails(interim_baf, column, bad_value):
    interim_baf.loc[0, column] = bad_value
    with pytest.raises(pa.errors.SchemaErrors):
        interim_schema.validate(interim_baf, lazy=True)


def test_unknown_category_fails(interim_baf):
    interim_baf["payment_type"] = interim_baf["payment_type"].cat.add_categories("ZZ")
    interim_baf.loc[0, "payment_type"] = "ZZ"
    with pytest.raises(pa.errors.SchemaErrors):
        interim_schema.validate(interim_baf, lazy=True)


def test_unexpected_column_fails(interim_baf):
    interim_baf["extra"] = 1
    with pytest.raises(pa.errors.SchemaErrors):
        interim_schema.validate(interim_baf, lazy=True)


def test_missing_column_fails(interim_baf):
    with pytest.raises(pa.errors.SchemaErrors):
        interim_schema.validate(interim_baf.drop(columns="month"), lazy=True)
