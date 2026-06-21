from __future__ import annotations

"""双词元隐藏状态差分可视化入口。

功能:
- 接收两个词元输入并分别提取隐藏状态热力图。
- 复用单词隐藏状态流程（缓存优先）降低重复计算。
- 计算 A-B 的逐元素差分矩阵。
- 返回 token A、token B 及差分热力图用于联动展示。
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config
from ..probes.probe_hidden_state import fetch_single_word_hidden_state


def run_study(
    *,
    token_a: str,
    token_b: str,
    config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/custom.yaml",
) -> dict[str, Any]:
    cfg = config or load_config(config_path)
    a = str(token_a or "").strip()
    b = str(token_b or "").strip()
    if not a or not b:
        return {
            "ok": False,
            "reason": "token_inputs_required",
            "matrix": [],
            "heatmaps": [],
            "ui_tasks": [{"name": "render_heatmap", "value_key": "heatmaps"}],
        }

    # Match current default single-word protocol: BOS on, assistant off.
    hm_a = fetch_single_word_hidden_state(
        word=a,
        include_bos=True,
        include_assistant=False,
        config=cfg,
    )
    if not isinstance(hm_a, dict) or not hm_a.get("ok"):
        reason = (hm_a or {}).get("reason", "token_a_failed") if isinstance(hm_a, dict) else "token_a_failed"
        return {
            "ok": False,
            "reason": f"token_a_failed:{reason}",
            "token_a": a,
            "token_b": b,
            "matrix": [],
            "heatmaps": [],
            "ui_tasks": [{"name": "render_heatmap", "value_key": "heatmaps"}],
        }

    hm_b = fetch_single_word_hidden_state(
        word=b,
        include_bos=True,
        include_assistant=False,
        config=cfg,
    )
    if not isinstance(hm_b, dict) or not hm_b.get("ok"):
        reason = (hm_b or {}).get("reason", "token_b_failed") if isinstance(hm_b, dict) else "token_b_failed"
        return {
            "ok": False,
            "reason": f"token_b_failed:{reason}",
            "token_a": a,
            "token_b": b,
            "matrix": [],
            "heatmaps": [],
            "ui_tasks": [{"name": "render_heatmap", "value_key": "heatmaps"}],
        }

    matrix_a = np.asarray(hm_a.get("matrix") or [], dtype=np.float32)
    matrix_b = np.asarray(hm_b.get("matrix") or [], dtype=np.float32)
    if matrix_a.ndim != 2 or matrix_b.ndim != 2:
        return {
            "ok": False,
            "reason": "invalid_matrix",
            "token_a": a,
            "token_b": b,
            "matrix": [],
            "heatmaps": [],
            "ui_tasks": [{"name": "render_heatmap", "value_key": "heatmaps"}],
        }
    if matrix_a.shape != matrix_b.shape:
        return {
            "ok": False,
            "reason": "shape_mismatch",
            "token_a": a,
            "token_b": b,
            "shape_a": [int(matrix_a.shape[0]), int(matrix_a.shape[1])],
            "shape_b": [int(matrix_b.shape[0]), int(matrix_b.shape[1])],
            "matrix": [],
            "heatmaps": [],
            "ui_tasks": [{"name": "render_heatmap", "value_key": "heatmaps"}],
        }

    diff = matrix_a - matrix_b
    row_labels = ["embedding"] + [f"layer_{idx}" for idx in range(1, int(diff.shape[0]))]
    heatmaps = [
        {"key": "token_a", "title": f"Token A: {a}", "matrix": matrix_a.tolist()},
        {"key": "token_b", "title": f"Token B: {b}", "matrix": matrix_b.tolist()},
        {"key": "token_diff", "title": f"Diff (A-B): {a} - {b}", "matrix": diff.tolist()},
    ]
    return {
        "ok": True,
        "study": "token_diff",
        "token_a": a,
        "token_b": b,
        "rows": int(diff.shape[0]),
        "cols": int(diff.shape[1]),
        "row_labels": row_labels,
        "matrix": diff.tolist(),
        "heatmaps": heatmaps,
        "cache_source_a": str(hm_a.get("cache_source") or "unknown"),
        "cache_source_b": str(hm_b.get("cache_source") or "unknown"),
        "protocol_a": str(hm_a.get("protocol") or ""),
        "protocol_b": str(hm_b.get("protocol") or ""),
        "ui_tasks": [{"name": "render_heatmap", "value_key": "heatmaps"}],
    }


def register_cli(subparsers: argparse._SubParsersAction, bool_parser) -> None:
    del bool_parser
    parser = subparsers.add_parser(
        "run-token-diff",
        help="Given two single-token words, render token A/B hidden states and A-B diff heatmap.",
    )
    parser.add_argument("token_a", help="Token A text")
    parser.add_argument("token_b", help="Token B text")


def try_execute_cli(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any] | None:
    if args.command != "run-token-diff":
        return None
    heatmap = run_study(
        token_a=str(args.token_a or ""),
        token_b=str(args.token_b or ""),
        config=config,
        config_path=args.config,
    )
    return {"hidden_state_heatmap": heatmap}
