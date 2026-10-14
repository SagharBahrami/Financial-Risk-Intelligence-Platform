"""Train a fraud model on months 0-5, evaluate on months 6-7, and log everything to MLflow.

Run with:  python -m risk_platform.fraud.train                   (model from configs/fraud.yaml)
           python -m risk_platform.fraud.train --model xgboost
"""
import argparse
import logging
from typing import Any

import mlflow
import pandas as pd

from risk_platform.common.config import load_config, resolve_path
from risk_platform.common.metrics import evaluate_binary
from risk_platform.common.plots import pr_curve_figure
from risk_platform.common.tracking import log_sklearn_model, setup_mlflow
from risk_platform.fraud.models import build_pipeline

logger = logging.getLogger(__name__)


def split_xy(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the features (X) from the target (y). The time column is not a feature."""
    X = df.drop(columns=[config["target"], config["time_column"]])
    y = df[config["target"]]
    return X, y


def run(config: dict[str, Any], model_name: str) -> dict[str, float]:
    params = config["models"][model_name]
    train = pd.read_parquet(resolve_path(config["data"]["train_path"]))
    test = pd.read_parquet(resolve_path(config["data"]["test_path"]))
    X_train, y_train = split_xy(train, config)
    X_test, y_test = split_xy(test, config)

    setup_mlflow(config["mlflow"]["experiment"])
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({"model": model_name, **params})
        mlflow.log_params({
            "train_months": config["split"]["train_months"],
            "test_months": config["split"]["test_months"],
            "n_features": X_train.shape[1],
        })

        pipeline = build_pipeline(model_name, params, config["features"]["missing_as_negative"])
        pipeline.fit(X_train, y_train)
        logger.info("%s trained on %d rows", model_name, len(X_train))

        scores = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_binary(y_test, scores, max_fpr=config["evaluation"]["max_fpr"])
        mlflow.log_metrics(metrics)
        for name, value in metrics.items():
            logger.info("%s = %.4f", name, value)

        mlflow.log_figure(pr_curve_figure(y_test, {model_name: scores}), "pr_curve.png")
        log_sklearn_model(pipeline, X_test)
    return metrics


def main() -> None:
    config = load_config("fraud")
    parser = argparse.ArgumentParser(description="Train a fraud detection model")
    parser.add_argument("--model", default=config["model"], choices=list(config["models"]))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(config, args.model)


if __name__ == "__main__":
    main()
