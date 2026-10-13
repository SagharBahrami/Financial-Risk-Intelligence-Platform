"""MLflow setup shared by all use cases.

By default, runs are stored in a local SQLite file (mlflow.db) in the project root.
Set the MLFLOW_TRACKING_URI environment variable to use an MLflow server instead.
"""
import os

import mlflow

from risk_platform.common.config import PROJECT_ROOT

DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"


def setup_mlflow(experiment: str) -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(experiment)
