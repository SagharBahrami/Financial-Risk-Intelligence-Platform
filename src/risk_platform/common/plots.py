"""Evaluation charts shared by all use cases."""
from matplotlib.figure import Figure
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import PrecisionRecallDisplay


def pr_curve_figure(y_true, scores_by_label: dict) -> Figure:
    """Precision-recall curves on the same axes, one per model ({label: scores})."""
    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    for i, (label, scores) in enumerate(scores_by_label.items()):
        PrecisionRecallDisplay.from_predictions(
            y_true, scores, name=label, ax=ax, plot_chance_level=(i == 0)
        )
    ax.set_title("Precision-recall curve (test set)")
    fig.tight_layout()
    return fig


def calibration_figure(y_true, scores_by_label: dict, n_bins: int = 10) -> Figure:
    """Reliability diagram: mean predicted probability vs. real positive rate, one line per model.

    Bins hold equal numbers of cases (quantiles); with rare positives, equal-width bins would put
    almost every case in the first bin. Both axes are logarithmic for the same reason.
    """
    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    for label, scores in scores_by_label.items():
        CalibrationDisplay.from_predictions(y_true, scores, n_bins=n_bins, strategy="quantile", name=label, ax=ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Calibration (test set)")
    fig.tight_layout()
    return fig
