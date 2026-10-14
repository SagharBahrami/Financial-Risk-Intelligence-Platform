import optuna
import pytest

from risk_platform.common.metrics import evaluate_binary
from risk_platform.fraud.models import build_pipeline
from risk_platform.fraud.train import split_xy
from risk_platform.fraud.tune import fit_with_pruning

MODELS = ["lightgbm", "xgboost"]


@pytest.fixture
def data(interim_baf, fraud_config):
    X, y = split_xy(interim_baf, fraud_config)

    def score_fn(scores):
        return evaluate_binary(y, scores, max_fpr=0.05)["recall_at_5pct_fpr"]

    return X, y, score_fn


def build(model_name, fraud_config):
    params = {"n_estimators": 10, "random_state": 0}
    if model_name == "lightgbm":
        params["verbose"] = -1
    return build_pipeline(model_name, params, fraud_config["features"]["missing_as_negative"])


@pytest.mark.parametrize("model_name", MODELS)
def test_trial_is_reported_every_n_rounds(data, fraud_config, model_name):
    X, y, score_fn = data
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())  # never prunes
    trial = study.ask()

    fit_with_pruning(build(model_name, fraud_config), model_name, trial, X, y, X, score_fn, every=5)
    study.tell(trial, 0.0)

    assert sorted(study.trials[0].intermediate_values) == [5, 10]


@pytest.mark.parametrize("model_name", MODELS)
def test_bad_trial_is_pruned(data, fraud_config, model_name):
    X, y, score_fn = data
    # recall is at most 1, so every reported value is below the threshold and the trial is pruned
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.ThresholdPruner(lower=2.0))
    trial = study.ask()

    with pytest.raises(optuna.TrialPruned):
        fit_with_pruning(build(model_name, fraud_config), model_name, trial, X, y, X, score_fn, every=5)
