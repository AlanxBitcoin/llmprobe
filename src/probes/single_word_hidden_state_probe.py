from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Probe-layer API for single-word hidden-state retrieval.
# - Probe should access runtime API + hidden_store/model internals.
# - Study layer composes probe outputs for UI payloads.

from typing import Any

from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api
from ..utils.extract_hidden import extract_single_word_hidden_matrix_store_first
from ..utils.logits import rank_vector_logits_and_cosine


def _get_or_start_runtime_api(config: dict[str, Any]):
    try:
        return get_runtime_api()
    except RuntimeError:
        return start_llama_api(config)


def fetch_single_word_hidden_state(word: str, config: dict[str, Any]) -> dict[str, Any]:
    """Store-first hidden-state retrieval for a single-token word."""
    api = _get_or_start_runtime_api(config)

    def _bundle_loader():
        result = api.execute_model_call(RuntimeRequest(config=config, force_reload=False))
        return result.bundle

    return extract_single_word_hidden_matrix_store_first(
        word=word,
        config=config,
        bundle_loader=_bundle_loader,
    )


def rank_last_layer_logits_from_heatmap(
    *,
    heatmap: dict[str, Any],
    config: dict[str, Any],
    top_k: int = 15,
) -> tuple[list[dict[str, Any]], str, str | None]:
    """Rank logits for the last layer vector from a hidden-state heatmap."""
    if not isinstance(heatmap, dict) or not heatmap.get("ok"):
        return [], "none", None

    matrix = heatmap.get("matrix") or []
    if not matrix:
        return [], "none", None

    try:
        api = _get_or_start_runtime_api(config)
        bundle = api.get_bundle()
        rows = rank_vector_logits_and_cosine(
            model=bundle.model,
            vector=matrix[-1],
            tokenizer=bundle.tokenizer,
            top_k=int(top_k),
        )
        return rows, "probe", None
    except Exception as exc:  # noqa: BLE001 - logits failure should not block heatmap output.
        return [], "error", str(exc)

