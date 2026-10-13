import numpy as np
import pytest

from risk_platform.common.metrics import evaluate_binary, recall_at_fpr


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
    assert set(metrics) == {"pr_auc", "roc_auc", "recall_at_5pct_fpr", "threshold_at_5pct_fpr"}
