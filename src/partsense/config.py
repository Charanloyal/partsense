"""Central typed configuration for PartSense platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    name: str = "PartSense"
    version: str = "0.1.0"
    random_seed: int = 42


@dataclass
class DataConfig:
    dataset_name: str = "AI4I 2020 Predictive Maintenance Dataset"
    uci_id: int = 601
    raw_path: str = "data/raw/ai4i2020.csv"
    fallback_url: str = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    )
    processed_train_path: str = "data/processed/train.parquet"
    processed_test_path: str = "data/processed/test.parquet"
    synthetic_dir: str = "data/synthetic"
    duckdb_path: str = "data/synthetic/aftersales.duckdb"
    test_size: float = 0.20
    target_column: str = "Machine failure"
    failure_mode_columns: list[str] = field(
        default_factory=lambda: ["TWF", "HDF", "PWF", "OSF", "RNF"]
    )
    leakage_forbidden_columns: list[str] = field(
        default_factory=lambda: ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    )


@dataclass
class FeaturesConfig:
    numerical_columns: list[str] = field(
        default_factory=lambda: [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
    )
    categorical_columns: list[str] = field(default_factory=lambda: ["Type"])
    engineered_physics_features: list[str] = field(
        default_factory=lambda: [
            "power_W",
            "temp_diff_K",
            "strain_minNm",
            "Type_L",
            "Type_M",
            "Type_H",
        ]
    )


@dataclass
class AppConfig:
    project: ProjectConfig
    data: DataConfig
    features: FeaturesConfig
    synthetic_aftersales: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    reports: dict[str, Any] = field(default_factory=dict)


def load_config(config_path: str | Path = "configs/config.yaml") -> AppConfig:
    """Load and parse the YAML configuration file."""
    path = Path(config_path)
    if not path.is_absolute():
        current_dir = Path(__file__).resolve().parent
        repo_root = current_dir.parent.parent
        resolved_path = repo_root / path
        if resolved_path.exists():
            path = resolved_path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    project_cfg = ProjectConfig(**raw.get("project", {}))
    data_cfg = DataConfig(**raw.get("data", {}))
    features_cfg = FeaturesConfig(**raw.get("features", {}))
    synthetic_cfg = raw.get("synthetic_aftersales", {})
    models_cfg = raw.get("models", {})
    reports_cfg = raw.get("reports", {})

    return AppConfig(
        project=project_cfg,
        data=data_cfg,
        features=features_cfg,
        synthetic_aftersales=synthetic_cfg,
        models=models_cfg,
        reports=reports_cfg,
    )
