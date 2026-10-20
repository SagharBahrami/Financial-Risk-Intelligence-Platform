import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression

from risk_platform.common.registry import (
    CHALLENGER,
    CHAMPION,
    get_champion,
    get_version_by_alias,
    register_and_promote,
)
from risk_platform.common.tracking import log_sklearn_model

NAME = "test-model"
METRIC = "recall_at_5pct_fpr"


def train_and_log() -> tuple[str, str]:
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    model = LogisticRegression().fit(X, [0, 0, 1, 1])
    with mlflow.start_run() as run:
        info = log_sklearn_model(model, X)
    return run.info.run_id, info.model_uri


def test_first_version_becomes_champion(local_mlflow):
    run_id, uri = train_and_log()
    version, promoted = register_and_promote(NAME, run_id, uri, METRIC, 0.50)

    assert promoted
    assert get_champion(NAME).version == version.version


def test_worse_version_does_not_replace_champion(local_mlflow):
    first, _ = register_and_promote(NAME, *train_and_log(), METRIC, 0.50)
    second, promoted = register_and_promote(NAME, *train_and_log(), METRIC, 0.40)

    assert not promoted
    assert get_champion(NAME).version == first.version
    assert get_version_by_alias(NAME, CHALLENGER).version == second.version


def test_better_version_replaces_champion(local_mlflow):
    register_and_promote(NAME, *train_and_log(), METRIC, 0.50)
    better, promoted = register_and_promote(NAME, *train_and_log(), METRIC, 0.60)

    assert promoted
    assert get_champion(NAME).version == better.version


def test_same_run_is_not_registered_twice(local_mlflow):
    run_id, uri = train_and_log()
    first, _ = register_and_promote(NAME, run_id, uri, METRIC, 0.50)
    again, _ = register_and_promote(NAME, run_id, uri, METRIC, 0.50)

    assert again.version == first.version


def test_champion_can_be_loaded_by_alias(local_mlflow):
    register_and_promote(NAME, *train_and_log(), METRIC, 0.50)
    model = mlflow.pyfunc.load_model(f"models:/{NAME}@{CHAMPION}")
    proba = model.predict(pd.DataFrame({"x": [3.0]}))
    assert proba[0, 1] > 0.5


def test_champion_without_the_metric_is_not_replaced(local_mlflow):
    """A champion registered with another selection metric cannot be compared with the new one."""
    old, _ = register_and_promote(NAME, *train_and_log(), "old_metric", 0.50)
    new, promoted = register_and_promote(NAME, *train_and_log(), METRIC, 0.90)

    assert not promoted
    assert get_champion(NAME).version == old.version
    assert get_version_by_alias(NAME, CHALLENGER).version == new.version


def test_forced_promotion_replaces_champion(local_mlflow):
    register_and_promote(NAME, *train_and_log(), "old_metric", 0.50)
    new, promoted = register_and_promote(NAME, *train_and_log(), METRIC, 0.10, force=True)

    assert promoted
    assert get_champion(NAME).version == new.version
    assert get_version_by_alias(NAME, CHALLENGER) is None
