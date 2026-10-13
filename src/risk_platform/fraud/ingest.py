"""Ingest the raw BAF CSV: fix column types, drop unused columns, validate and save as Parquet.

Run with:  python -m risk_platform.fraud.ingest
"""
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from risk_platform.common.config import load_config, resolve_path
from risk_platform.fraud.schema import interim_schema

logger = logging.getLogger(__name__)


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame, drop_columns: list[str], categorical_columns: list[str]) -> pd.DataFrame:
    """Drop unused columns and set categorical dtypes. Returns a new DataFrame."""
    df = df.drop(columns=drop_columns)
    df[categorical_columns] = df[categorical_columns].astype("category")
    return df


def run(config: dict[str, Any]) -> Path:
    raw_path = resolve_path(config["data"]["raw_path"])
    out_path = resolve_path(config["data"]["interim_path"])

    df = load_raw(raw_path)
    logger.info("Loaded %s: %d rows, %d columns", raw_path.name, *df.shape)

    df = clean(df, config["drop_columns"], config["categorical_columns"])
    df = interim_schema.validate(df, lazy=True)  # lazy=True reports all errors at once
    logger.info("Validation passed")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved %s (%.0f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(load_config("fraud"))


if __name__ == "__main__":
    main()
