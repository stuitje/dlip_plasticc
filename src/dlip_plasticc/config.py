from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass
class PathsConfig:
    data_dir: str
    features_dir: str
    predictions_dir: str
    classifiers_dir: str
    runs_dir: str


@dataclass
class DatasetConfig:
    train_name: str
    test_name: str
    total_chunks: int = 500


@dataclass
class AppConfig:
    paths: PathsConfig
    dataset: DatasetConfig
    raw: dict


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    default_path: str | Path = "configs/default.toml",
    local_path: str | Path = "configs/local.toml",
) -> AppConfig:
    default_path = Path(default_path)
    local_path = Path(local_path)

    with default_path.open("rb") as f:
        cfg = tomllib.load(f)

    if local_path.exists():
        with local_path.open("rb") as f:
            local_cfg = tomllib.load(f)
        cfg = _deep_update(cfg, local_cfg)

    paths = PathsConfig(**cfg["paths"])
    dataset = DatasetConfig(**cfg["dataset"])

    return AppConfig(paths=paths, dataset=dataset, raw=cfg)


def apply_avocado_settings(cfg: AppConfig) -> None:
    """Push configured paths into avocado.settings.settings."""
    from avocado.settings import settings as avocado_settings

    avocado_settings["data_directory"] = cfg.paths.data_dir
    avocado_settings["features_directory"] = cfg.paths.features_dir
    avocado_settings["predictions_directory"] = cfg.paths.predictions_dir
    avocado_settings["classifier_directory"] = cfg.paths.classifiers_dir