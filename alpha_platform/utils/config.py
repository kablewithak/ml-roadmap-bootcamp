"""Configuration management for the Alpha Platform."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field


class PlatformConfig(BaseModel):
    """Main platform configuration."""

    data_ingestion: Dict[str, Any] = Field(default_factory=dict)
    feature_engineering: Dict[str, Any] = Field(default_factory=dict)
    alpha_generation: Dict[str, Any] = Field(default_factory=dict)
    explainable_ai: Dict[str, Any] = Field(default_factory=dict)
    backtesting: Dict[str, Any] = Field(default_factory=dict)
    trading: Dict[str, Any] = Field(default_factory=dict)
    model_governance: Dict[str, Any] = Field(default_factory=dict)
    infrastructure: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)
    compliance: Dict[str, Any] = Field(default_factory=dict)
    reporting: Dict[str, Any] = Field(default_factory=dict)
    logging: Dict[str, Any] = Field(default_factory=dict)


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    # Go up from alpha_platform/utils/config.py to project root
    return current.parent.parent.parent


def load_config(config_path: Optional[str] = None) -> PlatformConfig:
    """
    Load platform configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        PlatformConfig object with all settings.
    """
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "configs" / "platform_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return PlatformConfig(**config_dict)


def get_data_path(subdir: str = "") -> Path:
    """Get path to data directory."""
    project_root = get_project_root()
    data_path = project_root / "data" / subdir
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path


def get_model_path(subdir: str = "") -> Path:
    """Get path to models directory."""
    project_root = get_project_root()
    model_path = project_root / "models" / subdir
    model_path.mkdir(parents=True, exist_ok=True)
    return model_path


def get_log_path() -> Path:
    """Get path to logs directory."""
    project_root = get_project_root()
    log_path = project_root / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    return log_path


# Global config instance
_config: Optional[PlatformConfig] = None


def get_config() -> PlatformConfig:
    """Get global config instance (singleton pattern)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
