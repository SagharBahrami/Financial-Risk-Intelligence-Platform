"""Model versioning with the MLflow Model Registry, shared by all use cases.

Each registered version points to the model of one training run. The alias "champion" marks
the version to use in production, so serving code loads "models:/<name>@champion" and never
needs a version number. A new version only becomes champion if it beats the current one.
"""
import logging

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)

CHAMPION = "champion"


def get_champion(name: str) -> ModelVersion | None:
    try:
        return MlflowClient().get_model_version_by_alias(name, CHAMPION)
    except MlflowException:
        return None  # model or alias does not exist yet


def register_and_promote(
    name: str, run_id: str, model_uri: str, metric: str, value: float, higher_is_better: bool = True
) -> tuple[ModelVersion, bool]:
    """Register the model of a run as a new version and make it champion if its metric is better.

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

    if champion is None:
        promoted = True
    else:
        champion_value = float(champion.tags[metric])
        promoted = value > champion_value if higher_is_better else value < champion_value
        logger.info("Current champion: v%s (%s = %.4f)", champion.version, metric, champion_value)

    if promoted:
        client.set_registered_model_alias(name, CHAMPION, version.version)
        logger.info("v%s is the new champion", version.version)
    else:
        logger.info("v%s is not better than the champion; champion unchanged", version.version)
    return version, promoted
