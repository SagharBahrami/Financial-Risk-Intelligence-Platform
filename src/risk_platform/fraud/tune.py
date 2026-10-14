"""Tune model hyperparameters with Optuna on a validation month taken from the training period.

- Search: TPE sampler (Bayesian optimization; the first trials are random).
- Pruning: Hyperband. Every few boosting rounds the trial reports its validation score;
  trials that look clearly worse than earlier ones are stopped early to save time.
- The test months (6-7) are never used here.
- Every trial, including pruned ones, is logged to MLflow as a nested run.

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
from risk_platform.common.pruning import XGBoostPruningCallback, lightgbm_pruning_callback
from risk_platform.common.tracking import setup_mlflow
from risk_platform.fraud.features import time_split
from risk_platform.fraud.models import SEARCH_SPACES, build_pipeline
from risk_platform.fraud.train import split_xy

logger = logging.getLogger(__name__)


def fit_with_pruning(pipeline, model_name: str, trial: optuna.Trial, X_fit, y_fit, X_val, score_fn, every: int):
    """Fit the pipeline while reporting validation scores to Optuna every `every` boosting rounds."""
    # The steps before the model are stateless (feature flags, category casting)
    X_val_ready = pipeline[:-1].fit(X_fit).transform(X_val)
    if model_name == "lightgbm":
        callback = lightgbm_pruning_callback(trial, X_val_ready, score_fn, every)
        return pipeline.fit(X_fit, y_fit, model__callbacks=[callback])
    if model_name == "xgboost":
        pipeline.set_params(model__callbacks=[XGBoostPruningCallback(trial, X_val_ready, score_fn, every)])
        return pipeline.fit(X_fit, y_fit)
    raise ValueError(f"No pruning support for model '{model_name}'")


def run(config: dict[str, Any], model_name: str, n_trials: int) -> dict[str, Any]:
    tuning = config["tuning"]
    max_fpr = config["evaluation"]["max_fpr"]
    metric = tuning["metric"]

    train = pd.read_parquet(resolve_path(config["data"]["train_path"]))
    fit_months = [m for m in config["split"]["train_months"] if m not in tuning["validation_months"]]
    fit_df, val_df = time_split(train, config["time_column"], fit_months, tuning["validation_months"])
    X_fit, y_fit = split_xy(fit_df, config)
    X_val, y_val = split_xy(val_df, config)
    missing_columns = config["features"]["missing_as_negative"]

    def score_fn(scores) -> float:
        return evaluate_binary(y_val, scores, max_fpr)[metric]

    def objective(trial: optuna.Trial) -> float:
        params = SEARCH_SPACES[model_name](trial)
        pipeline = build_pipeline(model_name, params, missing_columns)
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            try:
                fit_with_pruning(pipeline, model_name, trial, X_fit, y_fit, X_val, score_fn, tuning["report_every"])
            except optuna.TrialPruned as pruned:
                mlflow.set_tag("pruned", "true")
                logger.info("trial %d: %s", trial.number, pruned)
                raise
            metrics = evaluate_binary(y_val, pipeline.predict_proba(X_val)[:, 1], max_fpr)
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
        logger.info("trial %d: %s = %.4f", trial.number, metric, metrics[metric])
        return metrics[metric]

    setup_mlflow(config["mlflow"]["experiment"])
    with mlflow.start_run(run_name=f"tune_{model_name}"):
        mlflow.log_params({
            "model": model_name,
            "n_trials": n_trials,
            "sampler": "TPE",
            "pruner": "Hyperband",
            "fit_months": fit_months,
            "validation_months": tuning["validation_months"],
            "metric": metric,
        })
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner(min_resource=tuning["report_every"]),
        )
        # Trial 0 = current params from the config, so tuning can only improve on them
        study.enqueue_trial(config["models"][model_name])
        study.optimize(objective, n_trials=n_trials)

        n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metrics({f"best_val_{metric}": study.best_value, "n_pruned_trials": n_pruned})

    logger.info("Pruned %d of %d trials", n_pruned, n_trials)
    logger.info("Best trial: %d, validation %s = %.4f", study.best_trial.number, metric, study.best_value)
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
