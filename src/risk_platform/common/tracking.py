"""MLflow setup shared by all use cases.

By default, runs are stored in a local SQLite file (mlflow.db) in the project root.
Set the MLFLOW_TRACKING_URI environment variable to use an MLflow server instead.
"""
import os

import mlflow
import mlflow.sklearn
import pandas as pd
import skops.io as sio
from mlflow.models import infer_signature
from mlflow.models.model import ModelInfo

from risk_platform.common.config import PROJECT_ROOT

DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"


def setup_mlflow(experiment: str) -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(experiment)


def log_sklearn_model(model, X_example: pd.DataFrame) -> ModelInfo:
    """Log a fitted scikit-learn classifier pipeline to the active MLflow run.

    - When loaded for serving (mlflow.pyfunc), the model returns class probabilities
      (predict_proba), not 0/1 labels, so the decision threshold can be chosen separately.
    - MLflow saves sklearn models with skops, which only loads object types marked as trusted.
      The pipeline was built by this project's code, so the types it contains are trusted.
    - The model URI is saved as the run tag "model_uri", so the run's model can be registered later.
    """
    example = X_example.head(5)
    trusted_types = sio.get_untrusted_types(data=sio.dumps(model))
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        signature=infer_signature(example, model.predict_proba(example)),
        input_example=example,
        pyfunc_predict_fn="predict_proba",
        skops_trusted_types=trusted_types,
    )
    mlflow.set_tag("model_uri", info.model_uri)
    return info
