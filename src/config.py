from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "configs/custom.yaml") -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
