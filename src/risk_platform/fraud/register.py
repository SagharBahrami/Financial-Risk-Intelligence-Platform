"""Register a fraud model in the MLflow Model Registry and promote it to champion if it is better.

The selection metric (registry.metric in the config) is measured on month 5, not on the test months,
so choosing a champion never uses the test data. Runs without this metric (older training runs)
are skipped.

Run with:  python -m risk_platform.fraud.register                           (best run in the experiment)
           python -m risk_platform.fraud.register --run-id <id>             (a specific run)
           python -m risk_platform.fraud.register --run-id <id> --promote   (make it champion without comparing)
"""
import argparse
import logging
from typing import Any

import mlflow

from risk_platform.common.config import load_config
from risk_platform.common.registry import register_and_promote
from risk_platform.common.tracking import setup_mlflow

logger = logging.getLogger(__name__)


def run(config: dict[str, Any], run_id: str | None = None, force: bool = False) -> None:
    setup_mlflow(config["mlflow"]["experiment"])
    metric = config["registry"]["metric"]

    # Only training runs that logged a model have the "model_uri" tag (tuning runs do not)
    filter_string = f"tags.model_uri LIKE 'models:/%' and metrics.{metric} >= 0"
    if run_id is not None:
        filter_string += f" and attributes.run_id = '{run_id}'"
    runs = mlflow.search_runs(
        experiment_names=[config["mlflow"]["experiment"]],
        filter_string=filter_string,
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"No training run with a logged model and a {metric} value found")

    best = runs.iloc[0]
    logger.info("Selected run %s (%s)", best["run_id"], best["tags.mlflow.runName"])
    register_and_promote(
        name=config["registry"]["model_name"],
        run_id=best["run_id"],
        model_uri=best["tags.model_uri"],
        metric=metric,
        value=float(best[f"metrics.{metric}"]),
        force=force,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a fraud model in the MLflow Model Registry")
    parser.add_argument("--run-id", help="register this run instead of the best one")
    parser.add_argument("--promote", action="store_true", help="make it champion without comparing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(load_config("fraud"), args.run_id, args.promote)


if __name__ == "__main__":
    main()
