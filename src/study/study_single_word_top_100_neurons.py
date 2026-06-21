from __future__ import annotations

"""单词 top 神经元稀疏干预实验入口。

功能:
- 获取单词隐藏状态并计算基线 top-k logits。
- 在倒数第二层保留绝对值最大的 top-N 神经元，其余置零。
- 通过 hook 注入稀疏向量并继续推理，得到干预后输出。
- 对比干预前后 logits，支持神经元重要性分析。
"""

import argparse
from pathlib import Path
from typing import Any

from ..config import load_config
from ..probes.probe_layer_neuron import rank_logits_after_penultimate_topk_intervention
from ..probes.probe_hidden_state import (
    fetch_single_word_hidden_state,
    rank_last_layer_logits_from_heatmap,
)


def run_study(
    *,
    word: str,
    config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/custom.yaml",
    top_k_neurons: int = 100,
    intervention_layer: int = 30,
) -> dict[str, Any]:
    cfg = config or load_config(config_path)
    heatmap = fetch_single_word_hidden_state(word=word, include_bos=True, include_assistant=False, config=cfg)
    heatmap["top100_request"] = {
        "top_k_neurons": int(top_k_neurons),
        "intervention_layer": int(intervention_layer),
    }

    if not isinstance(heatmap, dict) or not heatmap.get("ok"):
        if isinstance(heatmap, dict):
            heatmap.setdefault("top_logits", [])
            heatmap.setdefault("logits_source", "none")
            heatmap.setdefault("logits_error", None)
            heatmap.setdefault("top_logits_top100", [])
            heatmap.setdefault("top_logits_top100_source", "none")
            heatmap.setdefault("top_logits_top100_error", None)
            heatmap.setdefault("top100_intervention", {})
            heatmap["ui_tasks"] = [
                {"name": "render_heatmap", "value_key": "matrix"},
                {"name": "render_logits", "value_key": "top_logits"},
                {"name": "render_logits_top100", "value_key": "top_logits_top100"},
            ]
        return heatmap

    base_rows, base_source, base_error = rank_last_layer_logits_from_heatmap(
        heatmap=heatmap,
        config=cfg,
        top_k=15,
    )
    heatmap["top_logits"] = base_rows
    heatmap["logits_source"] = base_source
    heatmap["logits_error"] = base_error

    top100_rows, top100_meta, top100_source, top100_error = rank_logits_after_penultimate_topk_intervention(
        heatmap=heatmap,
        word=word,
        config=cfg,
        keep_k=int(top_k_neurons),
        intervention_layer=int(intervention_layer),
        top_k=15,
    )
    heatmap["top_logits_top100"] = top100_rows
    heatmap["top_logits_top100_source"] = top100_source
    heatmap["top_logits_top100_error"] = top100_error
    heatmap["top100_intervention"] = top100_meta
    heatmap["ui_tasks"] = [
        {"name": "render_heatmap", "value_key": "matrix"},
        {"name": "render_logits", "value_key": "top_logits"},
        {"name": "render_logits_top100", "value_key": "top_logits_top100"},
    ]
    return heatmap


def register_cli(subparsers: argparse._SubParsersAction, bool_parser) -> None:
    del bool_parser
    parser = subparsers.add_parser(
        "run-single-word-top-100-neurons",
        help="Single-word study: heatmap + baseline top15 logits + penultimate-top100 intervention top15 logits",
    )
    parser.add_argument("word", help="Bare English word")
    parser.add_argument("--top-k-neurons", type=int, default=100, help="Keep abs top-K neurons at intervention layer")
    parser.add_argument("--intervention-layer", type=int, default=30, help="Decoder layer index to intervene (default: 30)")


def try_execute_cli(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any] | None:
    if args.command != "run-single-word-top-100-neurons":
        return None
    heatmap = run_study(
        word=args.word,
        config=config,
        config_path=args.config,
        top_k_neurons=int(args.top_k_neurons),
        intervention_layer=int(args.intervention_layer),
    )
    return {"hidden_state_heatmap": heatmap}
