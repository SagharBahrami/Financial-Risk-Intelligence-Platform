"""Evaluation metrics for imbalanced binary classification (fraud, default, money laundering)."""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def recall_at_fpr(y_true, y_score, max_fpr: float) -> tuple[float, float]:
    """Highest recall (TPR) reachable while keeping the false-positive rate <= max_fpr.

    Returns (recall, threshold): flag a case when its score is >= threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, max_fpr, side="right") - 1  # last point with fpr <= max_fpr
    return float(tpr[idx]), float(thresholds[idx])


def brier_skill_score(y_true, y_score) -> float:
    """Brier score compared with a dummy model that always predicts the base rate.

    1 = perfect, 0 = no better than the dummy, < 0 = worse than the dummy.
    """
    base_rate = np.mean(y_true)
    dummy_brier = base_rate * (1 - base_rate)  # Brier score of always predicting base_rate
    return float(1 - brier_score_loss(y_true, y_score) / dummy_brier)


def evaluate_binary(y_true, y_score, max_fpr: float = 0.05) -> dict[str, float]:
    """Threshold-free metrics: how well the scores rank cases, and how good they are as probabilities.

    recall_at_<pct>pct_fpr uses the best threshold for this data, so it measures ranking quality,
    not the decisions of a deployed model; use evaluate_at_threshold() with a frozen threshold for that.
    """
    recall, _ = recall_at_fpr(y_true, y_score, max_fpr)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "brier_skill": brier_skill_score(y_true, y_score),
        f"recall_at_{round(max_fpr * 100)}pct_fpr": recall,
    }


def evaluate_at_threshold(y_true, y_score, threshold: float) -> dict[str, float]:
    """Metrics of the 0/1 decisions made with a fixed threshold (flag when score >= threshold).

    F2 weighs recall twice as much as precision: missing a fraud costs more than a false alarm.
    """
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "f2_at_threshold": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "fpr_at_threshold": float(fp / (fp + tn)),
    }
