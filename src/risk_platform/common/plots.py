"""Evaluation charts shared by all use cases."""
from matplotlib.figure import Figure
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
