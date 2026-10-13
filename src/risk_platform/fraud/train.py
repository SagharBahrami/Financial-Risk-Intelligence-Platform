"""Train the fraud model on months 0-5, evaluate on months 6-7, and log everything to MLflow.

Run with:  python -m risk_platform.fraud.train
"""
import logging
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from risk_platform.common.config import load_config, resolve_path
from risk_platform.common.metrics import evaluate_binary
from risk_platform.common.tracking import setup_mlflow

logger = logging.getLogger(__name__)


def split_xy(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the features (X) from the target (y). The time column is not a feature."""
    X = df.drop(columns=[config["target"], config["time_column"]])
    y = df[config["target"]]
    return X, y


def build_pipeline(X: pd.DataFrame, config: dict[str, Any]) -> Pipeline:
    categorical = config["categorical_columns"]
    numeric = [c for c in X.columns if c not in categorical]

    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(**config["model"]["params"]))])


def run(config: dict[str, Any]) -> dict[str, float]:
    train = pd.read_parquet(resolve_path(config["data"]["train_path"]))
    test = pd.read_parquet(resolve_path(config["data"]["test_path"]))
    X_train, y_train = split_xy(train, config)
    X_test, y_test = split_xy(test, config)

    setup_mlflow(config["mlflow"]["experiment"])
    with mlflow.start_run(run_name=config["model"]["name"]):
        mlflow.log_params({"model": config["model"]["name"], **config["model"]["params"]})
        mlflow.log_params({
            "train_months": config["split"]["train_months"],
            "test_months": config["split"]["test_months"],
            "n_features": X_train.shape[1],
        })

        pipeline = build_pipeline(X_train, config)
        pipeline.fit(X_train, y_train)
        logger.info("Model trained on %d rows", len(X_train))

        scores = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_binary(y_test, scores, max_fpr=config["evaluation"]["max_fpr"])
        mlflow.log_metrics(metrics)
        for name, value in metrics.items():
            logger.info("%s = %.4f", name, value)

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(X_test, scores),
            input_example=X_test.head(5),
            skops_trusted_types=["numpy.dtype"],  # OneHotEncoder stores its output dtype
        )
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(load_config("fraud"))


if __name__ == "__main__":
    main()
