"""Probability calibration and a fixed decision threshold for a trained classifier, shared by all use cases.

The trained model is wrapped in FrozenEstimator, so nothing here refits it:
- calibrate()       learns a mapping from model score to real probability on separate calibration data
- with_threshold()  adds the decision threshold, so predict() returns 0/1 and predict_proba() the
                    calibrated probability. Model, calibrator and threshold are saved as one object.
"""
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import FixedThresholdClassifier, StratifiedKFold, cross_val_predict

from risk_platform.common.metrics import brier_skill_score

METHODS = ("sigmoid", "isotonic")  # sigmoid = Platt scaling (2 parameters); isotonic = step function


def calibrate(model: ClassifierMixin, X, y, method: str = "sigmoid") -> CalibratedClassifierCV:
    """Fit one calibrator on (X, y) on top of an already trained model."""
    if method not in METHODS:
        raise ValueError(f"Unknown calibration method '{method}'. Choose one of: {', '.join(METHODS)}")
    return CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X, y)


def compare_methods(model: ClassifierMixin, X, y, methods=METHODS, n_splits: int = 5) -> dict[str, float]:
    """Out-of-fold Brier score of each calibration method on the calibration data.

    Each fold's calibrator is scored on rows it did not see, so the methods can be compared
    without touching the test data.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}
    for method in methods:
        calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=method)
        proba = cross_val_predict(calibrator, X, y, cv=cv, method="predict_proba")[:, 1]
        results[f"cv_{method}_brier"] = float(brier_score_loss(y, proba))
        results[f"cv_{method}_brier_skill"] = brier_skill_score(y, proba)
    return results


def with_threshold(model: ClassifierMixin, threshold: float, X, y) -> FixedThresholdClassifier:
    """Attach a fixed decision threshold: predict() flags a case when its probability is >= threshold.

    (X, y) are only needed because scikit-learn requires a fit() call; the model is not refit.
    """
    if not np.isfinite(threshold):
        raise ValueError(f"Threshold must be a finite number, got {threshold}")
    return FixedThresholdClassifier(
        FrozenEstimator(model), threshold=threshold, response_method="predict_proba"
    ).fit(X, y)
