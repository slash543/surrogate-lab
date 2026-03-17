"""Config loader — single source of truth is configs/config.yaml."""
from pathlib import Path
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    """Load YAML config. Raises FileNotFoundError if missing."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path.resolve()}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_feature_names(cfg: dict) -> list[str]:
    return cfg["features"]["inputs"]


def get_target_name(cfg: dict) -> str:
    return cfg["features"]["target"]
