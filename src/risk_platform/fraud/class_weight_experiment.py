"""Controlled experiment: does weighting the fraud class help LightGBM and XGBoost?

The baseline is the current pipeline with no class weighting (scale_pos_weight = 1). Everything else is
identical in every run: data, months, hyperparameters, random seed, calibration comparison and threshold rule.
Only scale_pos_weight changes:

    1                    baseline (no weighting)
    sqrt(negatives/positives)   partial weighting
    negatives/positives         full balancing

The ratio is computed on the fit months only. Runs go to their own MLflow experiment, so the main
"fraud" experiment and its champion selection are not touched. The test months are only reported.

Weighting changes the scores' scale, so compare:
- ranking quality (cv_none_*, cal_*, uncalibrated_*): not affected by calibration
- probability quality after calibration (cv_<method>_brier, test brier)
- the deployed decisions (*_at_threshold)

Run with:  python -m risk_platform.fraud.class_weight_experiment
"""
import argparse
import logging
import math
from typing import Any

import pandas as pd

from risk_platform.common.config import load_config, resolve_path
from risk_platform.fraud.features import time_split
from risk_platform.fraud.train import run, split_xy

logger = logging.getLogger(__name__)

EXPERIMENT = "fraud_class_weight"
MODELS = ["lightgbm", "xgboost"]


def class_ratio(config: dict[str, Any]) -> float:
    """negatives / positives in the fit months."""
    train = pd.read_parquet(resolve_path(config["data"]["train_path"]))
    fit_df, _ = time_split(train, config["time_column"], config["training"]["fit_months"],
                           config["training"]["calibration_months"])
    _, y = split_xy(fit_df, config)
    return float((y == 0).sum() / (y == 1).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Class-weighting experiment for LightGBM and XGBoost")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config("fraud")
    config["mlflow"] = {**config["mlflow"], "experiment": EXPERIMENT}

    ratio = class_ratio(config)
    weights = {"baseline": 1.0, "sqrt": math.sqrt(ratio), "full": ratio}
    logger.info("negatives/positives in fit months = %.1f; weights = %s", ratio, weights)

    for model_name in args.models:
        for label, weight in weights.items():
            params = {**config["models"][model_name], "scale_pos_weight": weight}
            logger.info("=== %s, %s (scale_pos_weight = %.2f)", model_name, label, weight)
            run(config, model_name, params, extra_params={"weighting": label, "class_ratio": ratio})


if __name__ == "__main__":
    main()
