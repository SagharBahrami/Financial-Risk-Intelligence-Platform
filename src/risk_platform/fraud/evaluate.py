"""Compare a candidate model with the current champion on the same test set, as a report.

Both models are loaded the way serving loads them (mlflow.pyfunc) and scored on the test months.
Models that carry a decision threshold (metadata "decision_threshold") are also scored on their
0/1 decisions. The comparison is logged as its own MLflow run named "evaluate_<candidate run name>".

The champion is selected on month 5 (register.py), not with these test results.

Run with:  python -m risk_platform.fraud.evaluate --run-id <training run id>
"""
import argparse
import logging
from typing import Any

import mlflow
import pandas as pd

from risk_platform.common.config import load_config, resolve_path
from risk_platform.common.metrics import evaluate_at_threshold, evaluate_binary
from risk_platform.common.plots import pr_curve_figure
from risk_platform.common.registry import CHAMPION, get_champion
from risk_platform.common.tracking import setup_mlflow
from risk_platform.fraud.train import split_xy

logger = logging.getLogger(__name__)


def score(model_uri: str, X: pd.DataFrame) -> tuple[pd.Series, float | None]:
    """Fraud probabilities, and the model's decision threshold (None for models trained without one)."""
    model = mlflow.pyfunc.load_model(model_uri)
    threshold = (model.metadata.metadata or {}).get("decision_threshold")
    return model.predict(X)[:, 1], threshold


def evaluate(y: pd.Series, scores: pd.Series, threshold: float | None, max_fpr: float) -> dict[str, float]:
    metrics = evaluate_binary(y, scores, max_fpr)
    if threshold is not None:
        metrics.update(decision_threshold=threshold, **evaluate_at_threshold(y, scores, threshold))
    return metrics


def run(config: dict[str, Any], run_id: str) -> dict[str, dict[str, float]]:
    setup_mlflow(config["mlflow"]["experiment"])
    name = config["registry"]["model_name"]
    max_fpr = config["evaluation"]["max_fpr"]

    candidate_run = mlflow.get_run(run_id)
    candidate_name = candidate_run.data.tags["mlflow.runName"]
    X, y = split_xy(pd.read_parquet(resolve_path(config["data"]["test_path"])), config)
    # Same input format as serving: categorical columns as plain strings
    X = X.astype({col: str for col in config["categorical_columns"]})

    scores = {f"candidate ({candidate_name})": score(candidate_run.data.tags["model_uri"], X)}
    champion = get_champion(name)
    if champion is not None:
        scores[f"champion (v{champion.version})"] = score(f"models:/{name}@{CHAMPION}", X)

    results = {label: evaluate(y, s, threshold, max_fpr) for label, (s, threshold) in scores.items()}
    table = pd.DataFrame(results).T
    logger.info("Test set comparison:\n%s", table.round(4).to_string())

    with mlflow.start_run(run_name=f"evaluate_{candidate_name}"):
        mlflow.log_params({
            "candidate_run_id": run_id,
            "champion_version": champion.version if champion else "none",
            "test_months": config["split"]["test_months"],
        })
        for label, values in results.items():
            prefix = label.split(" ")[0]  # "candidate" or "champion"
            mlflow.log_metrics({f"{prefix}_{k}": v for k, v in values.items()})
        if champion is not None:
            candidate, champion_metrics = results.values()
            shared = candidate.keys() & champion_metrics.keys()
            mlflow.log_metrics({f"delta_{k}": candidate[k] - champion_metrics[k] for k in shared})
        mlflow.log_figure(pr_curve_figure(y, {label: s for label, (s, _) in scores.items()}), "pr_curve_comparison.png")
        mlflow.log_text(table.to_csv(), "comparison.csv")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a candidate fraud model with the champion")
    parser.add_argument("--run-id", required=True, help="training run of the candidate model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(load_config("fraud"), args.run_id)


if __name__ == "__main__":
    main()
