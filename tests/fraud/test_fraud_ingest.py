import pandas as pd

from risk_platform.fraud.ingest import clean, run


def test_clean_drops_constant_column(interim_baf):
    assert "device_fraud_count" not in interim_baf.columns


def test_clean_sets_categorical_dtypes(interim_baf, fraud_config):
    for col in fraud_config["categorical_columns"]:
        assert isinstance(interim_baf[col].dtype, pd.CategoricalDtype), col


def test_clean_does_not_modify_input(raw_baf, fraud_config):
    before = raw_baf.copy()
    clean(raw_baf, fraud_config["drop_columns"], fraud_config["categorical_columns"])
    pd.testing.assert_frame_equal(raw_baf, before)


def test_run_writes_parquet(raw_baf, fraud_config, tmp_path):
    raw_path = tmp_path / "Base.csv"
    raw_baf.to_csv(raw_path, index=False)
    config = {**fraud_config, "data": {"raw_path": raw_path, "interim_path": tmp_path / "out.parquet"}}

    out_path = run(config)

    result = pd.read_parquet(out_path)
    assert len(result) == len(raw_baf)
    assert isinstance(result["payment_type"].dtype, pd.CategoricalDtype)
