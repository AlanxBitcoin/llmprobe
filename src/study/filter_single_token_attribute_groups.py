from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_tokenizer_from_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    model_cfg = dict(cfg.get("model") or {})
    tokenizer_path = model_cfg.get("tokenizer_name_or_path") or model_cfg.get("model_name_or_path")
    if not tokenizer_path:
        raise ValueError("Config missing model.tokenizer_name_or_path and model.model_name_or_path")
    return AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )


def _split_words(raw_tokens: Any) -> list[str]:
    if isinstance(raw_tokens, list):
        return [str(x).strip() for x in raw_tokens if str(x).strip()]
    if isinstance(raw_tokens, str):
        return [x.strip() for x in raw_tokens.split(",") if x.strip()]
    return []


def filter_groups(input_path: Path, output_path: Path, config_path: Path) -> None:
    tokenizer = _load_tokenizer_from_config(config_path)
    with input_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise ValueError("Input JSON must contain a list at key 'groups'")

    total_kept = 0
    total_removed = 0
    for idx, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        words = _split_words(group.get("tokens", ""))
        kept: list[str] = []
        removed_count = 0
        for word in words:
            token_ids = tokenizer(word, add_special_tokens=False).get("input_ids") or []
            if len(token_ids) == 1:
                kept.append(word)
            else:
                removed_count += 1
        group["tokens"] = ", ".join(kept)
        total_kept += len(kept)
        total_removed += removed_count
        name = str(group.get("group_name", f"group_{idx}"))
        print(f"[{idx:03d}] {name}: kept={len(kept)} removed={removed_count}")

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"Done. output={output_path}")
    print(f"Total kept={total_kept}, total removed={total_removed}")


def main() -> None:
    root = _project_root()
    parser = argparse.ArgumentParser(
        description="Filter attribute-group words to keep only single-token words for the configured tokenizer."
    )
    parser.add_argument(
        "--input",
        default=str(root / "data" / "cache" / "attribute_groups.en_tmp.json"),
        help="Input JSON path",
    )
    parser.add_argument(
        "--output",
        default=str(root / "data" / "cache" / "attribute_groups.en_tmp.json"),
        help="Output JSON path (default: overwrite input)",
    )
    parser.add_argument(
        "--config",
        default=str(root / "configs" / "custom.yaml"),
        help="Config YAML path (for tokenizer path)",
    )
    args = parser.parse_args()

    filter_groups(
        input_path=Path(args.input).resolve(),
        output_path=Path(args.output).resolve(),
        config_path=Path(args.config).resolve(),
    )


if __name__ == "__main__":
    main()
