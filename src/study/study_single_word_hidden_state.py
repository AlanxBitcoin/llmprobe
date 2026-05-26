from __future__ import annotations

# Study: Single Word Hidden State - Working Principle
# 1) UI button ("Single Word Hidden State") triggers backend CLI action:
#    run-single-word-hidden-state.
# 2) main.py routes this action to study.run_study(...) in this file.
# 3) Study calls probe.fetch_single_word_hidden_state(...):
#    - Probe checks hidden_store first (disk cache).
#    - On cache miss, probe/runtime fallback runs model forward and writes back to store.
#    - Probe returns full hidden-state matrix for one token:
#      [embedding + hidden layers] x [hidden_dim].
# 4) Study treats that matrix as heatmap payload data and then calls
#    probe.rank_last_layer_logits_from_heatmap(...) for top-k logits ranking.
# 5) Study returns one combined payload to UI:
#    - heatmap data (matrix/rows/cols/metadata)
#    - logits table (top_logits/logits_source/logits_error)
#    - ui_tasks (render instructions)
# 6) UI renders by payload value:
#    - heatmap renderer uses matrix data
#    - logits renderer uses top_logits data
#    - if logits rows are empty, UI still shows source/error status.
#
# Design requirements (moved from PROJECT_DESIGN.md):
# - Study-layer orchestration for single-word hidden-state workflow.
# - Compose: hidden-state extraction -> heatmap payload -> top logits ranking.
# - Keep runtime/model lifecycle concerns in runtime_api, not UI coupling here.

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
