from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import yaml


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def safe_stem(text: str) -> str:
    normalized = re.sub(r"[\\/:*?\"<>|]+", "_", text.strip())
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = normalized.replace("+", "plus")
    normalized = re.sub(r"_+", "_", normalized).strip("._")
    return normalized or "item"


def read_lines(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def chunked(items: Iterable[str], size: int) -> list[list[str]]:
    batch: list[str] = []
    chunks: list[list[str]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            chunks.append(batch)
            batch = []
    if batch:
        chunks.append(batch)
    return chunks


def write_json(path: str | Path, payload: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def write_text(path: str | Path, content: str) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        fh.write(content)


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for header in headers:
            value = str(row.get(header, ""))
            escaped = value.replace('"', '""')
            values.append(f'"{escaped}"')
        lines.append(",".join(values))
    write_text(path, "\n".join(lines) + "\n")


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
