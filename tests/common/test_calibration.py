import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from risk_platform.common.calibration import best_method, calibrate, compare_methods, with_threshold
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


def test_none_keeps_the_model(overconfident):
    assert calibrate(overconfident, *make_data(100, seed=1), method="none") is overconfident


def test_compare_methods_reports_each_method(overconfident):
    results = compare_methods(overconfident, *make_data(2000, seed=1))
    assert set(results) == {"none", "sigmoid", "isotonic"}
    assert all({"brier", "brier_skill", "recall_at_5pct_fpr"} <= set(values) for values in results.values())


def test_overconfident_model_needs_calibration(overconfident):
    # a larger sample: with ~100 positives, one case moves recall by a full point and the choice is noise
    results = compare_methods(overconfident, *make_data(20000, seed=1))
    assert results["none"]["brier_skill"] < 0
    assert best_method(results) != "none"


def result(brier: float, recall: float, brier_skill: float = 0.09) -> dict[str, float]:
    return {"brier": brier, "brier_skill": brier_skill, "recall_at_5pct_fpr": recall}


def test_best_method_has_lowest_brier_when_ranking_is_kept():
    results = {"none": result(0.0106, 0.530), "sigmoid": result(0.0111, 0.531), "isotonic": result(0.0105, 0.528)}
    assert best_method(results) == "isotonic"


def test_best_method_rejects_methods_that_lose_ranking():
    # the lightgbm and xgboost month-5 numbers: isotonic has the best Brier but loses about 1 point of recall
    lightgbm = {"none": result(0.0107, 0.5308), "sigmoid": result(0.0106, 0.5294), "isotonic": result(0.0106, 0.5195)}
    xgboost = {"none": result(0.0106, 0.5344), "sigmoid": result(0.0111, 0.5337), "isotonic": result(0.0106, 0.5259)}
    assert best_method(lightgbm) == "sigmoid"
    assert best_method(xgboost) == "none"


def test_best_method_tolerance_is_adjustable():
    results = {"none": result(0.0106, 0.5344), "isotonic": result(0.0105, 0.5259)}
    assert best_method(results, max_recall_loss=0.005) == "none"
    assert best_method(results, max_recall_loss=0.01) == "isotonic"


def test_best_method_never_leaves_a_worse_than_base_rate_model_uncalibrated():
    # few frauds: recall is noisy and sigmoid "loses" 0.8 points, but "none" is far worse than the base rate
    results = {
        "none": result(0.1327, 0.5372, brier_skill=-8.0),
        "sigmoid": result(0.0426, 0.5289),
        "isotonic": result(0.0431, 0.4545),
    }
    assert best_method(results) == "sigmoid"


def test_best_method_falls_back_to_lowest_brier_when_no_method_beats_the_base_rate():
    results = {"none": result(0.30, 0.5, brier_skill=-3.0), "sigmoid": result(0.20, 0.5, brier_skill=-1.0)}
    assert best_method(results) == "sigmoid"


def test_best_method_without_none_uses_the_best_recall_as_reference():
    results = {"sigmoid": result(0.0111, 0.5337), "isotonic": result(0.0105, 0.5259)}
    assert best_method(results) == "sigmoid"


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
