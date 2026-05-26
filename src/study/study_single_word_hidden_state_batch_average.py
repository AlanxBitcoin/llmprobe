from __future__ import annotations

# Study: Single Word Hidden State Batch Average
# - Input is a comma-separated word list.
# - For each word, reuse the single-word hidden-state probe path (store-first).
# - Average all successful hidden-state matrices element-wise.
# - Return averaged heatmap + top-15 logits from the averaged last-layer vector.

from pathlib import Path
from typing import Any
import re

import numpy as np

from ..config import load_config
from ..probes.probe_hidden_state import rank_last_layer_logits_from_heatmap
from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api
from ..utils.extract_hidden import extract_single_word_hidden_matrix
from ..utils.token_hidden_store import protocol_from_flags


def _parse_words_csv(words_csv: str) -> list[str]:
    raw = str(words_csv or "").strip()
    # Support common separators: English comma, Chinese comma, semicolon, new lines.
    parts = [str(x).strip() for x in re.split(r"[,\n;，；]+", raw)]
    words = [x for x in parts if x]
    if not words:
        raise ValueError("Please provide at least one word (comma-separated).")
    return words


def run_study(
    *,
    words_csv: str,
    include_bos: bool = True,
    include_assistant: bool = False,
    config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/custom.yaml",
) -> dict[str, Any]:
    cfg = config or load_config(config_path)
    words = _parse_words_csv(words_csv)
    merged_cfg = dict(cfg)
    hs_cfg = dict((cfg or {}).get("hidden_store") or {})
    hs_cfg["protocol"] = protocol_from_flags(
        bos=bool(include_bos),
        assistant=bool(include_assistant) and bool(include_bos),
    )
    merged_cfg["hidden_store"] = hs_cfg

    try:
        api = get_runtime_api()
    except RuntimeError:
        api = start_llama_api(cfg)
    bundle = api.execute_model_call(RuntimeRequest(config=cfg, force_reload=False)).bundle

    ok_items: list[dict[str, Any]] = []
    failed_items: list[dict[str, Any]] = []
    sources: set[str] = set()
    protocol = ""
    rows = 0
    cols = 0

    for word in words:
        item = extract_single_word_hidden_matrix(
            bundle,
            word=word,
            config=merged_cfg,
        )
        if not isinstance(item, dict) or not item.get("ok"):
            failed_items.append(
                {
                    "word": word,
                    "reason": (item or {}).get("reason", "unknown") if isinstance(item, dict) else "unknown",
                    "token_count": (item or {}).get("token_count") if isinstance(item, dict) else None,
                }
            )
            continue
        matrix = np.asarray(item.get("matrix") or [], dtype=np.float32)
        if matrix.ndim != 2 or matrix.size == 0:
            failed_items.append({"word": word, "reason": "invalid_matrix"})
            continue
        if not ok_items:
            rows, cols = int(matrix.shape[0]), int(matrix.shape[1])
        elif matrix.shape != (rows, cols):
            failed_items.append(
                {
                    "word": word,
                    "reason": "shape_mismatch",
                    "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                    "expected_shape": [rows, cols],
                }
            )
            continue
        ok_items.append({"word": word, "matrix": matrix, "cache_source": str(item.get("cache_source") or "")})
        if item.get("cache_source"):
            sources.add(str(item.get("cache_source")))
        if not protocol and item.get("protocol"):
            protocol = str(item.get("protocol"))

    if not ok_items:
        return {
            "ok": False,
            "reason": "no_valid_words",
            "words_input": words_csv,
            "words": words,
            "successful_words": [],
            "failed_words": failed_items,
            "include_bos": bool(include_bos),
            "include_assistant": bool(include_assistant),
            "matrix": [],
            "rows": 0,
            "cols": 0,
            "row_labels": [],
            "top_logits": [],
            "logits_source": "none",
            "logits_error": None,
            "ui_tasks": [
                {"name": "render_heatmap", "value_key": "heatmaps"},
                {"name": "render_logits", "value_key": "top_logits"},
            ],
        }

    stacked = np.stack([x["matrix"] for x in ok_items], axis=0)  # [N, rows, cols]
    mean_matrix = np.mean(stacked, axis=0, dtype=np.float32)
    mad_matrix = np.mean(np.abs(stacked - mean_matrix[None, :, :]), axis=0, dtype=np.float32)
    # Third heatmap rule:
    # start from mean matrix; set point to zero when:
    # 1) any word has abs(value) < 0.1 at this point, OR
    # 2) any word has opposite sign vs mean at this point.
    any_abs_lt_0_1_mask = np.any(np.abs(stacked) < 0.1, axis=0)
    opposite_sign_mask = np.any(stacked * mean_matrix[None, :, :] < 0.0, axis=0)
    zero_mask = np.logical_or(any_abs_lt_0_1_mask, opposite_sign_mask)
    mean_any_lt_0_1_zero = np.array(mean_matrix, copy=True)
    mean_any_lt_0_1_zero[zero_mask] = 0.0

    # Batch score rule requested by user:
    # score = mean * abs(mean) / mad, mad==0 -> MAX, and clamp oversized values to MAX.
    max_value = float(np.finfo(np.float16).max)  # 65504.0
    numerator = np.multiply(mean_matrix, np.abs(mean_matrix, dtype=np.float32), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        score_matrix = np.divide(
            numerator,
            mad_matrix,
            out=np.full_like(numerator, fill_value=max_value, dtype=np.float32),
            where=mad_matrix > 0,
        )
    score_matrix = np.nan_to_num(score_matrix, nan=max_value, posinf=max_value, neginf=-max_value)
    score_matrix = np.clip(score_matrix, -max_value, max_value)
    cache_source = next(iter(sources)) if len(sources) == 1 else ("mixed" if sources else "unknown")
    row_labels = ["embedding"] + [f"layer_{idx}" for idx in range(1, int(score_matrix.shape[0]))]

    heatmap: dict[str, Any] = {
        "ok": True,
        "word": ", ".join([x["word"] for x in ok_items]),
        "words_input": words_csv,
        "words": words,
        "successful_words": [x["word"] for x in ok_items],
        "failed_words": failed_items,
        "word_count": int(len(ok_items)),
        "rows": int(mean_matrix.shape[0]),
        "cols": int(mean_matrix.shape[1]),
        "row_labels": row_labels,
        # Keep default matrix as mean matrix for compatibility and for logits.
        "matrix": mean_matrix.tolist(),
        # Secondary matrix: emphasized score matrix.
        "matrix_score": score_matrix.tolist(),
        "heatmaps": [
            {"key": "mean", "title": "Mean Hidden State Heatmap", "matrix": mean_matrix.tolist()},
            {"key": "score", "title": "Mean*Abs(Mean)/MAD Heatmap", "matrix": score_matrix.tolist()},
            {
                "key": "mean_any_lt_0_1_zero",
                "title": "Mean Heatmap (Any |v|<0.1 OR Opposite Sign -> 0)",
                "matrix": mean_any_lt_0_1_zero.tolist(),
            },
        ],
        "tokens": [],
        "cache_source": cache_source,
        "protocol": protocol,
        "include_bos": bool(include_bos),
        "include_assistant": bool(include_assistant),
        "batch_reduce": "mean_mul_abs_mean_div_mad",
        "batch_reduce_max_value": max_value,
    }

    # Logits must be ranked from the mean matrix (not from score matrix).
    logits_heatmap = dict(heatmap)
    logits_heatmap["matrix"] = mean_matrix.tolist()
    logits_rows, logits_source, logits_error = rank_last_layer_logits_from_heatmap(
        heatmap=logits_heatmap,
        config=cfg,
        top_k=15,
    )
    heatmap["top_logits"] = logits_rows
    heatmap["logits_source"] = logits_source
    heatmap["logits_error"] = logits_error
    heatmap["ui_tasks"] = [
        {"name": "render_heatmap", "value_key": "heatmaps"},
        {"name": "render_logits", "value_key": "top_logits"},
    ]
    return heatmap
