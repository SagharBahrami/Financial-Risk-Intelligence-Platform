import mlflow
import pytest

from risk_platform.fraud import evaluate, register, train


@pytest.fixture
def mlflow_runs(local_mlflow, monkeypatch):
    """Keep train.py, register.py and evaluate.py on the throwaway MLflow database of local_mlflow."""
    for module in (train, register, evaluate):
        monkeypatch.setattr(module, "setup_mlflow", lambda experiment: None)
    monkeypatch.setitem(register.__dict__, "mlflow", mlflow)


def last_run_id(name: str) -> str:
    runs = mlflow.search_runs(filter_string=f"tags.mlflow.runName = '{name}'", order_by=["start_time DESC"])
    return runs.iloc[0]["run_id"]


def experiment_name() -> str:
    return mlflow.get_experiment(mlflow.search_runs().iloc[0]["experiment_id"]).name


def test_candidate_and_champion_are_scored_at_their_own_thresholds(baf_files, mlflow_runs):
    train.run(baf_files, "logistic_regression")
    baf_files["mlflow"]["experiment"] = experiment_name()
    register.run(baf_files, last_run_id("logistic_regression"))
    train.run(baf_files, "lightgbm")

    results = evaluate.run(baf_files, last_run_id("lightgbm"))

    assert len(results) == 2  # candidate and champion
    for metrics in results.values():
        assert {"decision_threshold", "recall_at_threshold", "fpr_at_threshold"} <= set(metrics)
    logged = mlflow.search_runs(filter_string="tags.mlflow.runName = 'evaluate_lightgbm'").iloc[0]
    assert "metrics.delta_recall_at_threshold" in logged
