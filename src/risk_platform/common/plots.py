"""Evaluation charts shared by all use cases."""
from matplotlib.figure import Figure
from sklearn.metrics import PrecisionRecallDisplay


def pr_curve_figure(y_true, y_score, label: str) -> Figure:
    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_score, name=label, ax=ax, plot_chance_level=True)
    ax.set_title("Precision-recall curve (test set)")
    fig.tight_layout()
    return fig
