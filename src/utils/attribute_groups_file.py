from __future__ import annotations

from pathlib import Path
import json


_REL_PATH = Path("data/cache/attribute_groups.json")


def default_attribute_groups_payload() -> dict:
    return {
        "groups": [
            {
                "group_name": "color_basic",
                "tokens": ["red", "blue", "green", "yellow"],
                "filter": {"min_abs_mean": 0.2, "top_k_per_layer": 12},
            }
        ]
    }


def attribute_groups_path(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    return (root / _REL_PATH).resolve()


def ensure_attribute_groups_file(project_root: str | Path) -> Path:
    path = attribute_groups_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            json.dumps(default_attribute_groups_payload(), ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    return path


def load_attribute_groups_text(project_root: str | Path) -> tuple[Path, str]:
    path = ensure_attribute_groups_file(project_root)
    text = path.read_text(encoding="utf-8")
    return path, text

