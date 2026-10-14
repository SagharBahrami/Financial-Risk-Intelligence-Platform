"""Feature functions for BAF, and the time-based train/test split.

The feature transformations are applied inside the model pipeline (see models.py), not here,
so the model takes raw application data and training and serving run exactly the same code.

Run with:  python -m risk_platform.fraud.features   (writes the train and test files)
"""
import logging
from typing import Any

import numpy as np
import pandas as pd

from risk_platform.common.config import load_config, resolve_path

logger = logging.getLogger(__name__)


def add_missing_flags(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """For each column, add <column>_missing (1 if the value is negative) and replace negatives with NaN."""
    df = df.copy()
    for col in columns:
        is_missing = df[col] < 0
        df[f"{col}_missing"] = is_missing.astype("int8")
        df[col] = df[col].where(~is_missing, np.nan)
    return df


def time_split(
    df: pd.DataFrame, time_column: str, train_periods: list[int], test_periods: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(train_periods) & set(test_periods):
        raise ValueError("Train and test periods overlap")
    train = df[df[time_column].isin(train_periods)].reset_index(drop=True)
    test = df[df[time_column].isin(test_periods)].reset_index(drop=True)
    return train, test


def run(config: dict[str, Any]) -> None:
    df = pd.read_parquet(resolve_path(config["data"]["interim_path"]))

    split = config["split"]
    train, test = time_split(df, config["time_column"], split["train_months"], split["test_months"])
    target = config["target"]
    for name, part in [("train", train), ("test", test)]:
        logger.info("%s: %d rows, fraud rate %.2f%%", name, len(part), 100 * part[target].mean())
        path = resolve_path(config["data"][f"{name}_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(path, index=False)
        logger.info("Saved %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(load_config("fraud"))


if __name__ == "__main__":
    main()
