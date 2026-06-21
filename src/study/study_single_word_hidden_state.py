from __future__ import annotations

"""单词隐藏状态可视化与 logits 联动分析入口。

功能:
- 接收单词输入并走缓存优先路径提取完整隐藏状态矩阵。
- 组织热力图所需的结构化数据（层 x 神经元）。
- 基于末层向量计算 top-k logits 排名。
- 统一返回热力图、logits 与 ui_tasks，供前端一次渲染。
"""

import argparse
from pathlib import Path
from typing import Any

from ..config import load_config
from ..probes.probe_hidden_state import fetch_single_word_hidden_state, rank_last_layer_logits_from_heatmap


def run_study(
    *,
    word: str,
    include_bos: bool = True,
    include_assistant: bool = False,
    config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/custom.yaml",
) -> dict[str, Any]:
    cfg = config or load_config(config_path)
    heatmap = fetch_single_word_hidden_state(
        word=word,
        include_bos=bool(include_bos),
        include_assistant=bool(include_assistant),
        config=cfg,
    )
    if isinstance(heatmap, dict):
        heatmap["include_bos"] = bool(include_bos)
        heatmap["include_assistant"] = bool(include_assistant)

    if not isinstance(heatmap, dict) or not heatmap.get("ok"):
        if isinstance(heatmap, dict):
            heatmap.setdefault("top_logits", [])
            heatmap.setdefault("logits_source", "none")
            heatmap.setdefault("logits_error", None)
            heatmap["ui_tasks"] = [
                {"name": "render_heatmap", "value_key": "matrix"},
                {"name": "render_logits", "value_key": "top_logits"},
            ]
        return heatmap

    rows, source, error = rank_last_layer_logits_from_heatmap(heatmap=heatmap, config=cfg, top_k=15)
    heatmap["top_logits"] = rows
    heatmap["logits_source"] = source
    heatmap["logits_error"] = error
    heatmap["ui_tasks"] = [
        {"name": "render_heatmap", "value_key": "matrix"},
        {"name": "render_logits", "value_key": "top_logits"},
    ]

    return heatmap


def register_cli(subparsers: argparse._SubParsersAction, bool_parser) -> None:
    parser = subparsers.add_parser(
        "run-single-word-hidden-state",
        help="Compute and return full hidden-state matrix (embedding + layers) for one word",
    )
    parser.add_argument("word", help="Bare English word")
    parser.add_argument(
        "--include-bos",
        type=bool_parser,
        default=True,
        help="Whether to include BOS context in single-word hidden-state extraction (true/false)",
    )
    parser.add_argument(
        "--include-assistant",
        type=bool_parser,
        default=False,
        help="Whether to include assistant chat-prefix context (true/false; requires include-bos=true)",
    )


def try_execute_cli(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any] | None:
    if args.command != "run-single-word-hidden-state":
        return None
    heatmap = run_study(
        word=args.word,
        include_bos=bool(args.include_bos),
        include_assistant=bool(args.include_assistant),
        config=config,
        config_path=args.config,
    )
    return {"hidden_state_heatmap": heatmap}
