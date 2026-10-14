import json

import mlflow
import optuna
import pandas as pd
import pytest

from risk_platform.common.tracking import log_sklearn_model
from risk_platform.fraud.models import BUILDERS, SEARCH_SPACES, build_pipeline
from risk_platform.fraud.train import split_xy


@pytest.fixture
def xy(interim_baf, fraud_config):
    return split_xy(interim_baf, fraud_config)


@pytest.fixture
def missing_columns(fraud_config):
    return fraud_config["features"]["missing_as_negative"]


def test_split_xy_removes_target_and_time(xy):
    X, _ = xy
    assert "fraud_bool" not in X.columns
    assert "month" not in X.columns


@pytest.mark.parametrize("model_name", list(BUILDERS))
def test_every_model_fits_and_predicts(xy, fraud_config, missing_columns, model_name):
    X, y = xy
    pipeline = build_pipeline(model_name, fraud_config["models"][model_name], missing_columns).fit(X, y)

    scores = pipeline.predict_proba(X)[:, 1]
    assert scores.shape == (len(X),)
    assert ((scores >= 0) & (scores <= 1)).all()


@pytest.mark.parametrize("model_name", list(BUILDERS))
def test_model_accepts_json_input(xy, fraud_config, missing_columns, model_name):
    """At serving time, data arrives as JSON: plain strings and ints, sometimes with new category values."""
    X, y = xy
    pipeline = build_pipeline(model_name, fraud_config["models"][model_name], missing_columns).fit(X, y)

    X_json = pd.DataFrame(json.loads(X.to_json(orient="records")))
    X_json.loc[0, "device_os"] = "android"  # not seen in training

    assert pipeline.predict_proba(X_json).shape == (len(X), 2)


def test_logged_model_scores_json_input(xy, fraud_config, missing_columns, local_mlflow):
    """Same path as the API: log to MLflow, load as pyfunc (with schema checks), score a JSON request."""
    X, y = xy
    pipeline = build_pipeline("lightgbm", fraud_config["models"]["lightgbm"], missing_columns).fit(X, y)
    with mlflow.start_run():
        info = log_sklearn_model(pipeline, X)

    model = mlflow.pyfunc.load_model(info.model_uri)
    request = pd.DataFrame(json.loads(X.head(1).to_json(orient="records")))
    proba = model.predict(request)

    # Served output = probabilities, identical to the in-memory pipeline
    assert proba.shape == (1, 2)
    assert proba[0, 1] == pytest.approx(pipeline.predict_proba(X.head(1))[0, 1])


def test_every_model_has_config(fraud_config):
    assert set(BUILDERS) == set(fraud_config["models"])


def test_unknown_model_fails():
    with pytest.raises(ValueError):
        build_pipeline("random_forest", {}, [])


@pytest.mark.parametrize("model_name", list(SEARCH_SPACES))
def test_search_space_builds_a_model(xy, missing_columns, model_name):
    X, y = xy
    params = SEARCH_SPACES[model_name](optuna.create_study().ask())
    build_pipeline(model_name, params, missing_columns).fit(X, y)
