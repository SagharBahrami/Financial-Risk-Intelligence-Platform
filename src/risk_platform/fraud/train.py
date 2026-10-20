"""Train the final fraud model, calibrate it, choose its decision threshold, and test it once.

1. Fit the pipeline on months 0-4 (hyperparameters from the config, or from a tuning run).
2. Freeze the pipeline. On month 5, compare the calibration methods (none, sigmoid, isotonic) by
   cross-validation and fit the chosen calibrator.
3. Choose the decision threshold on month 5 with the calibrated probabilities:
   the highest recall with a false-positive rate of at most 5%.
4. Evaluate the frozen model (pipeline + calibrator + threshold) on months 6-7.
   The test results are only reported; they are never used to change or select a model.

The logged model returns calibrated probabilities (predict_proba) and applies the threshold in
predict(). The threshold is also saved in the model metadata, for serving with mlflow.pyfunc.

Logged metrics:
- no prefix          test months, after calibration; *_at_threshold = decisions with the frozen threshold
- "uncalibrated_"    test months, before calibration
- "cal_"             month 5, before calibration. No model, calibrator or threshold is fitted on these
                     scores; cal_recall_at_5pct_fpr is used to select the champion (register.py). Once it
                     is used to pick between models, the winner's value is no longer an unbiased estimate
                     of future performance; the test months give that estimate.
- "cv_<method>_"     month 5, out-of-fold metrics of each calibration method. With "auto", the lowest
                     Brier score among methods that lose at most max_recall_loss recall at the FPR limit
                     (compared with "none") is chosen, so the chosen method's score is slightly optimistic.

recall_at_5pct_fpr uses the best threshold for the data it is measured on, so it measures ranking
quality. The decisions of the deployed model are measured by the *_at_threshold metrics.

Run with:  python -m risk_platform.fraud.train                      (model and params from configs/fraud.yaml)
           python -m risk_platform.fraud.train --model xgboost
           python -m risk_platform.fraud.train --tune-run-id <id>   (best params of a tuning run)
"""
import argparse
import logging
from typing import Any

import mlflow
import pandas as pd
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.pipeline import Pipeline

from risk_platform.common.calibration import best_method, calibrate, compare_methods, with_threshold
from risk_platform.common.config import load_config, resolve_path
from risk_platform.common.metrics import evaluate_at_threshold, evaluate_binary, recall_at_fpr
from risk_platform.common.plots import calibration_figure, pr_curve_figure
from risk_platform.common.tracking import log_sklearn_model, setup_mlflow
from risk_platform.fraud.features import time_split
from risk_platform.fraud.models import build_pipeline

logger = logging.getLogger(__name__)


def split_xy(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the features (X) from the target (y). The time column is not a feature."""
    X = df.drop(columns=[config["target"], config["time_column"]])
    y = df[config["target"]]
    return X, y


def load_tuned_params(tune_run_id: str) -> tuple[str, dict[str, Any]]:
    """Model name and best hyperparameters saved by tune.py."""
    best = mlflow.artifacts.load_dict(f"runs:/{tune_run_id}/best_params.json")
    return best["model"], best["params"]


def evaluate_model(
    pipeline: Pipeline, model: FixedThresholdClassifier, X: pd.DataFrame, y: pd.Series, max_fpr: float
) -> dict[str, float]:
    """Scores before calibration, after calibration, and of the 0/1 decisions at the model's threshold."""
    uncalibrated = pipeline.predict_proba(X)[:, 1]
    calibrated = model.predict_proba(X)[:, 1]
    return {
        **{f"uncalibrated_{k}": v for k, v in evaluate_binary(y, uncalibrated, max_fpr).items()},
        **evaluate_binary(y, calibrated, max_fpr),
        **evaluate_at_threshold(y, calibrated, model.threshold),
    }


def run(
    config: dict[str, Any],
    model_name: str,
    params: dict[str, Any] | None = None,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, float]:
    params = config["models"][model_name] if params is None else params
    training = config["training"]
    max_fpr = config["evaluation"]["max_fpr"]

    train = pd.read_parquet(resolve_path(config["data"]["train_path"]))
    test = pd.read_parquet(resolve_path(config["data"]["test_path"]))
    fit_df, cal_df = time_split(train, config["time_column"], training["fit_months"], training["calibration_months"])
    X_fit, y_fit = split_xy(fit_df, config)
    X_cal, y_cal = split_xy(cal_df, config)
    X_test, y_test = split_xy(test, config)

    setup_mlflow(config["mlflow"]["experiment"])
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({"model": model_name, **params})
        mlflow.log_params({
            "fit_months": training["fit_months"],
            "calibration_months": training["calibration_months"],
            "test_months": config["split"]["test_months"],
            "calibration_setting": training["calibration_method"],
            "n_features": X_fit.shape[1],
            **(extra_params or {}),
        })

        pipeline = build_pipeline(model_name, params, config["features"]["missing_as_negative"])
        pipeline.fit(X_fit, y_fit)
        logger.info("%s trained on %d rows", model_name, len(X_fit))

        # From here on the pipeline is frozen: only the calibrator and the threshold are fitted
        cal_metrics = evaluate_binary(y_cal, pipeline.predict_proba(X_cal)[:, 1], max_fpr)
        mlflow.log_metrics({f"cal_{k}": v for k, v in cal_metrics.items()})

        comparison = compare_methods(pipeline, X_cal, y_cal, training["compare_methods"], max_fpr=max_fpr)
        for method, values in comparison.items():
            mlflow.log_metrics({f"cv_{method}_{k}": v for k, v in values.items()})
            logger.info("month %s, out-of-fold Brier with %s calibration = %.5f",
                        training["calibration_months"], method, values["brier"])
        method = training["calibration_method"]
        if method == "auto":
            method = best_method(comparison, training["max_recall_loss"])
        mlflow.log_param("calibration_method", method)

        calibrated = calibrate(pipeline, X_cal, y_cal, method)
        _, threshold = recall_at_fpr(y_cal, calibrated.predict_proba(X_cal)[:, 1], max_fpr)
        model = with_threshold(calibrated, threshold, X_cal, y_cal)
        mlflow.log_metric("decision_threshold", threshold)
        logger.info("Calibration: %s; decision threshold = %.4f", method, threshold)

        # Final test of the frozen model: reported only, never used to change or select it
        metrics = evaluate_model(pipeline, model, X_test, y_test, max_fpr)
        mlflow.log_metrics(metrics)
        for name, value in metrics.items():
            logger.info("test %s = %.4f", name, value)

        test_scores = {"uncalibrated": pipeline.predict_proba(X_test)[:, 1]}
        if method != "none":
            test_scores[f"calibrated ({method})"] = model.predict_proba(X_test)[:, 1]
        mlflow.log_figure(pr_curve_figure(y_test, {model_name: model.predict_proba(X_test)[:, 1]}), "pr_curve.png")
        mlflow.log_figure(calibration_figure(y_test, test_scores), "calibration_curve.png")
        log_sklearn_model(model, X_test, metadata={
            "decision_threshold": threshold,
            "calibration_method": method,
            "max_fpr": max_fpr,
        })
    return metrics


def main() -> None:
    config = load_config("fraud")
    parser = argparse.ArgumentParser(description="Train, calibrate and test a fraud detection model")
    parser.add_argument("--model", choices=list(config["models"]), help=f"default: {config['model']}")
    parser.add_argument("--tune-run-id", help="use the best hyperparameters of this tuning run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.tune_run_id is None:
        run(config, args.model or config["model"])
        return

    setup_mlflow(config["mlflow"]["experiment"])
    model_name, params = load_tuned_params(args.tune_run_id)
    if args.model is not None and args.model != model_name:
        parser.error(f"tuning run {args.tune_run_id} tuned '{model_name}', not '{args.model}'")
    run(config, model_name, params, extra_params={"tune_run_id": args.tune_run_id})


if __name__ == "__main__":
    main()
