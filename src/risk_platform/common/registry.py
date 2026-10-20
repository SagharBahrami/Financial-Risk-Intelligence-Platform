"""Model versioning with the MLflow Model Registry, shared by all use cases.

Each registered version points to the model of one training run. Two aliases mark its role:
- "champion"   the version used in production; serving code loads "models:/<name>@champion"
- "challenger" the newest candidate that did not beat the champion (yet)

A new version is registered as challenger and only becomes champion if its metric is better.
If the champion has no value for the metric (it was registered when another metric was used),
the two cannot be compared: the new version stays challenger unless promotion is forced.
"""
import logging

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

CHAMPION = "champion"
CHALLENGER = "challenger"


def get_version_by_alias(name: str, alias: str) -> ModelVersion | None:
    try:
        return MlflowClient().get_model_version_by_alias(name, alias)
    except MlflowException:
        return None  # model or alias does not exist yet


def get_champion(name: str) -> ModelVersion | None:
    return get_version_by_alias(name, CHAMPION)


def register_and_promote(
    name: str,
    run_id: str,
    model_uri: str,
    metric: str,
    value: float,
    higher_is_better: bool = True,
    force: bool = False,
) -> tuple[ModelVersion, bool]:
    """Register the model of a run as a new version and make it champion if its metric is better.

    force=True makes it champion without comparing.

    Returns (version, promoted).
    """
    client = MlflowClient()

    existing = client.search_model_versions(f"name = '{name}' and run_id = '{run_id}'")
    if existing:
        version = existing[0]
        logger.info("Run %s is already registered as %s v%s", run_id, name, version.version)
    else:
        version = mlflow.register_model(model_uri, name, tags={metric: str(value)})
        logger.info("Registered %s v%s (%s = %.4f)", name, version.version, metric, value)

    champion = get_champion(name)
    if champion is not None and champion.version == version.version:
        return version, False

    if champion is None or force:
        promoted = True
    elif metric not in champion.tags:
        promoted = False
        logger.warning(
            "Champion v%s has no %s value, so it cannot be compared; promote with force=True if intended",
            champion.version, metric,
        )
    else:
        champion_value = float(champion.tags[metric])
        promoted = value > champion_value if higher_is_better else value < champion_value
        logger.info("Current champion: v%s (%s = %.4f)", champion.version, metric, champion_value)

    if promoted:
        client.set_registered_model_alias(name, CHAMPION, version.version)
        challenger = get_version_by_alias(name, CHALLENGER)
        if challenger is not None and challenger.version == version.version:
            client.delete_registered_model_alias(name, CHALLENGER)
        logger.info("v%s is the new champion", version.version)
    else:
        client.set_registered_model_alias(name, CHALLENGER, version.version)
        logger.info("v%s is not better than the champion; registered as challenger", version.version)
    return version, promoted
