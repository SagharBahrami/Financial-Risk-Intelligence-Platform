"""Evaluation metrics for imbalanced binary classification (fraud, default, money laundering)."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def recall_at_fpr(y_true, y_score, max_fpr: float) -> tuple[float, float]:
    """Highest recall (TPR) reachable while keeping the false-positive rate <= max_fpr.

    Returns (recall, threshold): flag a case when its score is >= threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, max_fpr, side="right") - 1  # last point with fpr <= max_fpr
    return float(tpr[idx]), float(thresholds[idx])


def evaluate_binary(y_true, y_score, max_fpr: float = 0.05) -> dict[str, float]:
    recall, threshold = recall_at_fpr(y_true, y_score, max_fpr)
    pct = round(max_fpr * 100)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        f"recall_at_{pct}pct_fpr": recall,
        f"threshold_at_{pct}pct_fpr": threshold,
    }
