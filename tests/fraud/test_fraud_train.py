import json

import mlflow
import numpy as np
import pandas as pd
import pytest

from risk_platform.fraud import train, tune
from risk_platform.fraud.train import split_xy

REPORTED = [
    "pr_auc", "roc_auc", "brier", "brier_skill", "uncalibrated_brier", "uncalibrated_brier_skill",
    "recall_at_threshold", "precision_at_threshold", "f2_at_threshold", "fpr_at_threshold",
]


@pytest.fixture
def mlflow_runs(local_mlflow, monkeypatch):
    """Keep train.py and tune.py on the throwaway MLflow database of local_mlflow."""
    monkeypatch.setattr(train, "setup_mlflow", lambda experiment: None)
    monkeypatch.setattr(tune, "setup_mlflow", lambda experiment: None)


def last_run(name: str) -> pd.Series:
    runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{name}'", order_by=["start_time DESC"])
    return runs.iloc[0]


@pytest.mark.parametrize("model_name", ["logistic_regression", "lightgbm"])
def test_train_logs_metrics_before_and_after_calibration(baf_files, mlflow_runs, model_name):
    metrics = train.run(baf_files, model_name)

    assert set(REPORTED) <= set(metrics)
    logged = last_run(model_name)
    for name in ["decision_threshold", "cal_brier", "cal_uncalibrated_brier", "cv_sigmoid_brier", "cv_isotonic_brier"]:
        assert not np.isnan(logged[f"metrics.{name}"])
    assert logged["params.fit_months"] == "[0, 1, 2, 3, 4]"
    assert logged["params.calibration_months"] == "[5]"


def test_threshold_respects_fpr_limit_on_calibration_month(baf_files, mlflow_runs):
    train.run(baf_files, "lightgbm")
    assert last_run("lightgbm")["metrics.cal_fpr_at_threshold"] <= baf_files["evaluation"]["max_fpr"]


def test_served_model_is_calibrated_and_carries_its_threshold(baf_files, mlflow_runs):
    train.run(baf_files, "lightgbm")
    logged = last_run("lightgbm")
    uri = logged["tags.model_uri"]

    served = mlflow.pyfunc.load_model(uri)
    threshold = served.metadata.metadata["decision_threshold"]
    assert threshold == pytest.approx(logged["metrics.decision_threshold"])

    X, _ = split_xy(pd.read_parquet(baf_files["data"]["test_path"]), baf_files)
    request = pd.DataFrame(json.loads(X.head(20).to_json(orient="records")))  # same format as the API
    proba = served.predict(request)[:, 1]

    model = mlflow.sklearn.load_model(uri)
    assert model.threshold == pytest.approx(threshold)
    assert np.array_equal(model.predict(request), (proba >= threshold).astype(int))


def test_tuned_params_are_used_for_refit(baf_files, mlflow_runs):
    best = tune.run(baf_files, "lightgbm", n_trials=2)
    tune_run_id = last_run("tune_lightgbm")["run_id"]

    model_name, params = train.load_tuned_params(tune_run_id)
    assert model_name == "lightgbm"
    assert params == best
    assert params["random_state"] == 42  # fixed params are saved too, not only the searched ones

    train.run(baf_files, model_name, params, extra_params={"tune_run_id": tune_run_id})
    logged = last_run("lightgbm")
    assert logged["params.tune_run_id"] == tune_run_id
    assert logged["params.n_estimators"] == str(params["n_estimators"])


@pytest.mark.parametrize("months", [{"validation_months": [5]}, {"fit_months": [0, 1, 2, 3, 6]}])
def test_tuning_cannot_use_held_out_months(fraud_config, months):
    fraud_config["tuning"].update(months)
    with pytest.raises(ValueError, match="held-out months"):
        tune.check_months(fraud_config)


def test_default_config_months_do_not_overlap(fraud_config):
    tune.check_months(fraud_config)
    training = fraud_config["training"]
    assert not set(training["fit_months"]) & set(training["calibration_months"])
    assert not set(training["calibration_months"]) & set(fraud_config["split"]["test_months"])
