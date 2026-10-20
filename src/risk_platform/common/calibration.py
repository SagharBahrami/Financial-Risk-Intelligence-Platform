"""Probability calibration and a fixed decision threshold for a trained classifier, shared by all use cases.

The trained model is wrapped in FrozenEstimator, so nothing here refits it:
- compare_methods() scores each calibration method by cross-validation on the calibration data
- calibrate()       learns a mapping from model score to real probability on the calibration data
- with_threshold()  adds the decision threshold, so predict() returns 0/1 and predict_proba() the
                    calibrated probability. Model, calibrator and threshold are saved as one object.
"""
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import FixedThresholdClassifier, StratifiedKFold, cross_val_predict

from risk_platform.common.metrics import evaluate_binary

# none = keep the model's own probabilities; sigmoid = Platt scaling (2 parameters); isotonic = step function
METHODS = ("none", "sigmoid", "isotonic")
RECALL = "recall_at_5pct_fpr"


def _check_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"Unknown calibration method '{method}'. Choose one of: {', '.join(METHODS)}")


def calibrate(model: ClassifierMixin, X, y, method: str = "sigmoid") -> ClassifierMixin:
    """Fit one calibrator on (X, y) on top of an already trained model ("none" returns the model)."""
    _check_method(method)
    if method == "none":
        return model
    return CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X, y)


def compare_methods(
    model: ClassifierMixin, X, y, methods=METHODS, n_splits: int = 5, max_fpr: float = 0.05
) -> dict[str, dict[str, float]]:
    """Out-of-fold metrics (evaluate_binary) of each calibration method on the calibration data.

    Brier scores show probability quality. The ranking metrics show whether a method loses ranking
    quality: isotonic calibration is a step function, so it turns different scores into ties.

    Each fold's calibrator is scored on rows it did not see, so the methods can be compared without
    touching the test data. "none" fits nothing, so its probabilities are scored directly.
    Picking the best one (best_method) makes the chosen method's score slightly optimistic.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {}
    for method in methods:
        _check_method(method)
        if method == "none":
            proba = model.predict_proba(X)[:, 1]
        else:
            calibrator = CalibratedClassifierCV(FrozenEstimator(model), method=method)
            proba = cross_val_predict(calibrator, X, y, cv=cv, method="predict_proba")[:, 1]
        results[method] = evaluate_binary(y, proba, max_fpr)
    return results


def best_method(results: dict[str, dict[str, float]], max_recall_loss: float = 0.005) -> str:
    """The method with the lowest out-of-fold Brier score that keeps the ranking.

    A method is allowed only if its recall at the false-positive limit is at most max_recall_loss
    below the recall of "none" (the model's own ranking). Without "none" in the results, the best
    recall of any method is the reference. Isotonic calibration can gain a little Brier score and
    lose ranking quality through ties; this rule does not trade one for the other.
    The reference method is always allowed, so there is always a choice.

    Guard: a method with a negative Brier skill score is worse than always predicting the base rate.
    It is never chosen while another method does better than that, even if it keeps the ranking.
    This matters on small calibration samples, where recall is noisy and could otherwise leave a
    badly miscalibrated model uncalibrated.
    """
    reference = results["none"][RECALL] if "none" in results else max(r[RECALL] for r in results.values())
    allowed = [m for m, r in results.items() if r[RECALL] >= reference - max_recall_loss]
    for candidates in (allowed, list(results)):
        useful = [m for m in candidates if results[m]["brier_skill"] >= 0]
        if useful:
            return min(useful, key=lambda method: results[method]["brier"])
    return min(results, key=lambda method: results[method]["brier"])


def with_threshold(model: ClassifierMixin, threshold: float, X, y) -> FixedThresholdClassifier:
    """Attach a fixed decision threshold: predict() flags a case when its probability is >= threshold.

    (X, y) are only needed because scikit-learn requires a fit() call; the model is not refit.
    """
    if not np.isfinite(threshold):
        raise ValueError(f"Threshold must be a finite number, got {threshold}")
    return FixedThresholdClassifier(
        FrozenEstimator(model), threshold=threshold, response_method="predict_proba"
    ).fit(X, y)
