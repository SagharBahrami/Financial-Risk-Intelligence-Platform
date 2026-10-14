import mlflow
import pytest


@pytest.fixture
def local_mlflow(tmp_path):
    """A throwaway MLflow database and artifact folder for each test."""
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(experiment_id=mlflow.create_experiment("test", artifact_location=tmp_path.as_uri()))
    yield
    mlflow.set_tracking_uri(None)
