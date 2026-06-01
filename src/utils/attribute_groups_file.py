from __future__ import annotations

from pathlib import Path
import json


_REL_PATH = Path("data/cache/attribute_groups.json")


def default_attribute_groups_payload() -> dict:
    return {
        "groups": [
            {
                "group_name": "color_basic",
                "tokens": "red, blue, green, yellow",
            }
        ]
    }

def _tokens_to_csv(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return ", ".join(parts)
    return str(value or "").strip()


def format_attribute_groups_json(payload: object) -> str:
    groups: list[object] = []
    if isinstance(payload, dict) and isinstance(payload.get("groups"), list):
        groups = list(payload.get("groups") or [])
    elif isinstance(payload, list):
        groups = list(payload)
    else:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    normalized_groups: list[dict[str, object]] = []
    for idx, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        one = dict(group)
        name = str(one.get("group_name") or one.get("name") or f"group_{idx + 1}").strip()
        if not name:
            continue
        normalized_groups.append(
            {
                "group_name": name,
                "tokens": _tokens_to_csv(one.get("tokens")),
            }
        )
    lines = [json.dumps(g, ensure_ascii=False, separators=(",", ":")) for g in normalized_groups]
    body = ",\n    ".join(lines)
    if body:
        return "{\n  \"groups\": [\n    " + body + "\n  ]\n}"
    return "{\n  \"groups\": []\n}"


def attribute_groups_path(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    return (root / _REL_PATH).resolve()


def ensure_attribute_groups_file(project_root: str | Path) -> Path:
    path = attribute_groups_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            format_attribute_groups_json(default_attribute_groups_payload()),
            encoding="utf-8",
        )
    return path


def load_attribute_groups_text(project_root: str | Path) -> tuple[Path, str]:
    path = ensure_attribute_groups_file(project_root)
    text = path.read_text(encoding="utf-8")
    return path, text
