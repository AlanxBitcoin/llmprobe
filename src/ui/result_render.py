from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


SUPPORTED_ARTIFACT_SUFFIXES = {".csv", ".json", ".png", ".jpg", ".jpeg", ".html", ".mp4"}


def preview_csv(path: str | Path, max_rows: int = 200) -> dict[str, Any]:
    csv_path = Path(path)
    if not csv_path.exists():
        return {"path": str(csv_path), "headers": [], "rows": [], "error": "CSV file not found"}
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        rows: list[dict[str, str]] = []
        for index, row in enumerate(reader):
            if index >= max_rows:
                break
            rows.append({key: value for key, value in row.items()})
    return {"path": str(csv_path), "headers": headers, "rows": rows}


def collect_recent_artifacts(output_root: str | Path, since_timestamp: float, limit: int = 40) -> list[dict[str, Any]]:
    root = Path(output_root)
    if not root.exists():
        return []
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_ARTIFACT_SUFFIXES:
            continue
        try:
            if path.stat().st_mtime >= since_timestamp:
                candidates.append(path)
        except OSError:
            continue
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return [_artifact_payload(path, root) for path in candidates[:limit]]


def newest_csv_preview(artifacts: list[dict[str, Any]], max_rows: int = 200) -> dict[str, Any] | None:
    for artifact in artifacts:
        if artifact["type"] == "csv":
            return preview_csv(artifact["path"], max_rows=max_rows)
    return None


def _artifact_payload(path: Path, output_root: Path) -> dict[str, Any]:
    suffix = path.suffix.lower().lstrip(".")
    try:
        relative = path.relative_to(output_root)
    except ValueError:
        relative = path
    return {
        "name": path.name,
        "path": str(path),
        "relative_path": str(relative).replace("\\", "/"),
        "type": suffix,
        "size": path.stat().st_size,
        "modified": path.stat().st_mtime,
        "url": f"/outputs/{str(relative).replace('\\', '/')}",
    }
