from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Load YAML config with project-root-relative path resolution.
# - Default configuration source is configs/custom.yaml.
# - Keep config loading lightweight and deterministic.

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def validate_config(config: dict[str, Any]) -> None:
    required_top_keys = ("model", "analysis", "output")
    missing = [key for key in required_top_keys if key not in config]
    if missing:
        raise ValueError(f"Config missing required top-level keys: {missing}")

    model_cfg = config.get("model") or {}
    if not model_cfg.get("model_name_or_path"):
        raise ValueError("Config requires model.model_name_or_path")

    analysis_cfg = config.get("analysis") or {}
    if "target_layer" not in analysis_cfg:
        raise ValueError("Config requires analysis.target_layer")


def load_config(config_path: str | Path = "configs/custom.yaml") -> dict[str, Any]:
    resolved = _resolve_project_path(config_path)
    with resolved.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    validate_config(config)
    return config
