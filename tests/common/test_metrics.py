import numpy as np
import pytest

from risk_platform.common.metrics import brier_skill_score, evaluate_at_threshold, evaluate_binary, recall_at_fpr


def test_perfect_model_has_full_recall():
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    recall, _ = recall_at_fpr(y, scores, max_fpr=0.0)
    assert recall == 1.0


def test_recall_at_fpr_respects_fpr_limit():
    # 10 negatives, 2 positives. One positive scores below every negative.
    y = np.array([0] * 10 + [1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.05])

    recall, threshold = recall_at_fpr(y, scores, max_fpr=0.1)

    flagged = scores >= threshold
    fpr = (flagged & (y == 0)).sum() / (y == 0).sum()
    assert fpr <= 0.1
    assert recall == pytest.approx(0.5)


def test_evaluate_binary_keys():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.2, 0.7, 0.4, 0.9])
    metrics = evaluate_binary(y, scores, max_fpr=0.05)
    # No threshold: the best threshold for the evaluation data is not one a deployed model would use
    assert set(metrics) == {"pr_auc", "roc_auc", "brier", "brier_skill", "recall_at_5pct_fpr"}


def test_brier_skill_is_zero_for_base_rate_model():
    y = np.array([0, 0, 0, 1])
    scores = np.full(4, 0.25)  # always predicts the fraud rate
    assert brier_skill_score(y, scores) == pytest.approx(0.0)


def test_brier_skill_is_one_for_perfect_model():
    y = np.array([0, 0, 0, 1])
    assert brier_skill_score(y, y.astype(float)) == pytest.approx(1.0)


def test_brier_skill_is_negative_for_overconfident_model():
    # Same ranking as a perfect model, but probabilities far too high (like class_weight="balanced")
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    scores = np.array([0.6] * 9 + [0.9])
    assert brier_skill_score(y, scores) < 0


def test_evaluate_at_threshold():
    # Flagged (score >= 0.5): one fraud and one normal case. The other fraud is missed.
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.9, 0.4])

    metrics = evaluate_at_threshold(y, scores, threshold=0.5)

    assert metrics["recall_at_threshold"] == pytest.approx(0.5)
    assert metrics["precision_at_threshold"] == pytest.approx(0.5)
    assert metrics["fpr_at_threshold"] == pytest.approx(0.25)
    assert metrics["f2_at_threshold"] == pytest.approx(0.5)  # precision == recall, so F2 == both
