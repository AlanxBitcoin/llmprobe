from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def _parse_tokens(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value or "")
    return [x.strip() for x in text.split(",") if x.strip()]


def _is_single_token(tokenizer, word: str) -> bool:
    ids = tokenizer(word, add_special_tokens=False).get("input_ids") or []
    return len(ids) == 1


def _filter_group(tokenizer, group: dict[str, Any], min_keep: int) -> dict[str, Any] | None:
    name = str(group.get("group_name", "")).strip()
    if not name:
        return None
    words = _parse_tokens(group.get("tokens", ""))
    kept: list[str] = []
    seen: set[str] = set()
    for word in words:
        if word in seen:
            continue
        seen.add(word)
        if _is_single_token(tokenizer, word):
            kept.append(word)
    if len(kept) < min_keep:
        return None
    return {"group_name": name, "tokens": ", ".join(kept)}


def run(input_path: Path, output_path: Path, model_path: str, min_keep: int) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    groups = payload.get("groups", [])
    if not isinstance(groups, list):
        raise ValueError("Input JSON must contain a list field: groups")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    filtered: list[dict[str, Any]] = []
    for item in groups:
        if not isinstance(item, dict):
            continue
        result = _filter_group(tokenizer, item, min_keep=min_keep)
        if result is not None:
            filtered.append(result)

    output = {"groups": filtered}
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"input_groups={len(groups)}")
    print(f"output_groups={len(filtered)}")
    print(f"output_file={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter words by tokenizer single-token constraint; no generation."
    )
    parser.add_argument("--input", required=True, help="Input JSON file with groups.")
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    parser.add_argument(
        "--model-path",
        default="C:/AI_Model/Llama3_8B_Instruct",
        help="Tokenizer model path.",
    )
    parser.add_argument(
        "--min-keep",
        type=int,
        default=1,
        help="Drop groups with fewer than this many kept words.",
    )
    args = parser.parse_args()

    run(
        input_path=Path(args.input),
        output_path=Path(args.output),
        model_path=args.model_path,
        min_keep=max(0, int(args.min_keep)),
    )


if __name__ == "__main__":
    main()
