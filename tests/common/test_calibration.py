import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from risk_platform.common.calibration import calibrate, compare_methods, with_threshold
from risk_platform.common.tracking import log_sklearn_model


def make_data(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """About 5% positives; the positive rate grows with x."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(2 * x - 4)))).astype(int)
    return pd.DataFrame({"x": x}), y


@pytest.fixture
def overconfident():
    """class_weight='balanced' ranks well but predicts probabilities far too high."""
    X, y = make_data(2000, seed=0)
    return LogisticRegression(class_weight="balanced").fit(X, y)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibration_improves_brier_on_new_data(overconfident, method):
    X_cal, y_cal = make_data(2000, seed=1)
    X_new, y_new = make_data(2000, seed=2)

    calibrated = calibrate(overconfident, X_cal, y_cal, method)

    before = brier_score_loss(y_new, overconfident.predict_proba(X_new)[:, 1])
    after = brier_score_loss(y_new, calibrated.predict_proba(X_new)[:, 1])
    assert after < before


def test_calibration_does_not_refit_the_model(overconfident):
    coef = overconfident.coef_.copy()
    calibrate(overconfident, *make_data(2000, seed=1))
    assert np.array_equal(overconfident.coef_, coef)


def test_unknown_method_fails(overconfident):
    with pytest.raises(ValueError):
        calibrate(overconfident, *make_data(100, seed=1), method="beta")


def test_compare_methods_reports_each_method(overconfident):
    results = compare_methods(overconfident, *make_data(2000, seed=1))
    assert set(results) == {"cv_sigmoid_brier", "cv_sigmoid_brier_skill", "cv_isotonic_brier", "cv_isotonic_brier_skill"}


def test_threshold_decides_and_keeps_calibrated_probabilities(overconfident):
    X_cal, y_cal = make_data(2000, seed=1)
    calibrated = calibrate(overconfident, X_cal, y_cal)
    model = with_threshold(calibrated, 0.2, X_cal, y_cal)

    proba = model.predict_proba(X_cal)[:, 1]
    assert np.allclose(proba, calibrated.predict_proba(X_cal)[:, 1])
    assert np.array_equal(model.predict(X_cal), (proba >= 0.2).astype(int))


def test_infinite_threshold_fails(overconfident):
    X, y = make_data(100, seed=1)
    with pytest.raises(ValueError):
        with_threshold(overconfident, np.inf, X, y)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_logged_model_keeps_calibrator_and_threshold(overconfident, local_mlflow, method):
    X_cal, y_cal = make_data(2000, seed=1)
    model = with_threshold(calibrate(overconfident, X_cal, y_cal, method), 0.2, X_cal, y_cal)
    with mlflow.start_run():
        info = log_sklearn_model(model, X_cal, metadata={"decision_threshold": 0.2})

    served = mlflow.pyfunc.load_model(info.model_uri)
    assert served.metadata.metadata["decision_threshold"] == 0.2
    assert np.allclose(served.predict(X_cal)[:, 1], model.predict_proba(X_cal)[:, 1])
