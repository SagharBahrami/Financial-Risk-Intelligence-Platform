"""Load YAML configs and resolve project paths."""
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml as a dictionary."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Turn a path from a config file (relative to the project root) into an absolute path."""
    return PROJECT_ROOT / relative_path
