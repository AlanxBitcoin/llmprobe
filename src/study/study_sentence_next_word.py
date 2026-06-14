from __future__ import annotations

"""句子末 token 的下一词预测 study 入口。

功能:
- 对输入句子提取末 token 的各层隐藏状态热力图。
- 在上下文可用时基于末层向量计算 top-k logits。
- 输出热力图与 logits 渲染任务，供前端直接展示。
"""

from pathlib import Path
from typing import Any

from ..config import load_config
from ..probes.probe_hidden_state import (
    fetch_sentence_last_token_hidden_state,
    rank_last_layer_logits_from_heatmap,
)


def run_study(
    *,
    sentence: str,
    config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/custom.yaml",
) -> dict[str, Any]:
    cfg = config or load_config(config_path)
    ctx = fetch_sentence_last_token_hidden_state(sentence=sentence, config=cfg)
    if not isinstance(ctx, dict) or not ctx.get("ok"):
        if isinstance(ctx, dict):
            ctx.setdefault("top_logits", [])
            ctx.setdefault("logits_source", "none")
            ctx.setdefault("logits_error", None)
            ctx["ui_tasks"] = [{"name": "render_logits", "value_key": "top_logits"}]
        return ctx

    rows, source, error = rank_last_layer_logits_from_heatmap(heatmap=ctx, config=cfg, top_k=15)
    return {
        "ok": True,
        "study": "sentence_next_word",
        "sentence": str(sentence or ""),
        "top_logits": rows,
        "logits_source": source,
        "logits_error": error,
        "ui_tasks": [{"name": "render_logits", "value_key": "top_logits"}],
    }
