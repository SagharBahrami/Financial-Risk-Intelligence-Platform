"""Tune model hyperparameters with Optuna on a validation month taken from the training period.

The test months (6-7) are never used here. Each trial is logged to MLflow as a nested run.

Run with:  python -m risk_platform.fraud.tune --model lightgbm --trials 30
"""
import argparse
import logging
from typing import Any

import mlflow
import optuna
import pandas as pd
import yaml

from risk_platform.common.config import load_config, resolve_path
from risk_platform.common.metrics import evaluate_binary
from risk_platform.common.tracking import setup_mlflow
from risk_platform.fraud.features import time_split
from risk_platform.fraud.models import SEARCH_SPACES, build_pipeline
from risk_platform.fraud.train import split_xy

logger = logging.getLogger(__name__)


def run(config: dict[str, Any], model_name: str, n_trials: int) -> dict[str, Any]:
    tuning = config["tuning"]
    train = pd.read_parquet(resolve_path(config["data"]["train_path"]))
    fit_months = [m for m in config["split"]["train_months"] if m not in tuning["validation_months"]]
    fit_df, val_df = time_split(train, config["time_column"], fit_months, tuning["validation_months"])
    X_fit, y_fit = split_xy(fit_df, config)
    X_val, y_val = split_xy(val_df, config)
    missing_columns = config["features"]["missing_as_negative"]
    metric = tuning["metric"]

    def objective(trial: optuna.Trial) -> float:
        params = SEARCH_SPACES[model_name](trial)
        pipeline = build_pipeline(model_name, params, missing_columns).fit(X_fit, y_fit)
        metrics = evaluate_binary(y_val, pipeline.predict_proba(X_val)[:, 1], config["evaluation"]["max_fpr"])
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        logger.info("trial %d: %s = %.4f", trial.number, metric, metrics[metric])
        return metrics[metric]

    setup_mlflow(config["mlflow"]["experiment"])
    with mlflow.start_run(run_name=f"tune_{model_name}"):
        mlflow.log_params({
            "model": model_name,
            "n_trials": n_trials,
            "fit_months": fit_months,
            "validation_months": tuning["validation_months"],
            "metric": metric,
        })
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        # Trial 0 = current params from the config, so tuning can only improve on them
        study.enqueue_trial(config["models"][model_name])
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric(f"best_val_{metric}", study.best_value)

    logger.info("Best validation %s = %.4f", metric, study.best_value)
    logger.info("Best params (copy into configs/fraud.yaml):\n%s", yaml.safe_dump(study.best_params, sort_keys=False))
    return study.best_params


def main() -> None:
    config = load_config("fraud")
    parser = argparse.ArgumentParser(description="Tune a fraud model with Optuna")
    parser.add_argument("--model", default="lightgbm", choices=list(SEARCH_SPACES))
    parser.add_argument("--trials", type=int, default=config["tuning"]["n_trials"])
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(config, args.model, args.trials)


if __name__ == "__main__":
    main()
