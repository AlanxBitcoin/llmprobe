from __future__ import annotations

# Study: Single Word Top 100 Neurons (Penultimate Intervention)
# 1) Fetch full hidden-state heatmap for one single-token word (store-first).
# 2) Compute baseline top-15 logits from the original last-layer vector.
# 3) Take penultimate-layer vector, keep abs top-100 neurons, zero others.
# 4) Inject that sparse vector into penultimate layer (last token) via hook.
# 5) Re-run model path so final two layers are naturally recomputed.
# 6) Compute another top-15 logits table from intervened output.

from pathlib import Path
from typing import Any

from ..config import load_config
from ..probes.single_word_hidden_state_probe import (
    fetch_single_word_hidden_state,
    rank_last_layer_logits_from_heatmap,
    rank_logits_after_penultimate_topk_intervention,
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
