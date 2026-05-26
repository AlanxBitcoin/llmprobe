from __future__ import annotations

# Probe: single-word hidden-state only.
# Keep this file focused on store-first single-word read + baseline logits ranking.

from typing import Any

import numpy as np
import torch

from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api
from ..utils.extract_hidden import extract_single_word_hidden_matrix_store_first
from ..utils.logits import rank_vector_logits_and_cosine
from ..utils.token_hidden_store import build_protocol_input_ids, parse_token_ids_with_bos_alias


def _get_or_start_runtime_api(config: dict[str, Any]):
    try:
        return get_runtime_api()
    except RuntimeError:
        return start_llama_api(config)


def fetch_single_word_hidden_state(
    word: str,
    include_bos: bool,
    include_assistant: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Store-first hidden-state retrieval for a single-token word."""
    api = _get_or_start_runtime_api(config)

    def _bundle_loader():
        result = api.execute_model_call(RuntimeRequest(config=config, force_reload=False))
        return result.bundle

    return extract_single_word_hidden_matrix_store_first(
        word=word,
        include_bos=bool(include_bos),
        include_assistant=bool(include_assistant),
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
            apply_final_norm=False,
        )
        return rows, "probe", None
    except Exception as exc:  # noqa: BLE001
        return [], "error", str(exc)


def fetch_sentence_last_token_hidden_state(
    sentence: str,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run a full sentence once and return:
    - last-token hidden matrix [embedding + layers, hidden_dim]
    - all-token hidden states [embedding + layers, seq_len, hidden_dim]
    - last-token attention-to-all-previous-tokens for each layer (head-averaged)
    """
    text = str(sentence or "").strip()
    if not text:
        return {
            "ok": False,
            "reason": "prefix_text_required",
            "matrix": [],
            "all_token_hidden_by_layer": [],
            "last_token_attention_by_layer": [],
        }

    try:
        api = _get_or_start_runtime_api(config)
        bundle = api.execute_model_call(RuntimeRequest(config=config, force_reload=False)).bundle
        model = bundle.model
        tokenizer = bundle.tokenizer
        protocol = str(((config or {}).get("hidden_store") or {}).get("protocol", "bos1_assistant0"))

        token_ids = [int(x) for x in parse_token_ids_with_bos_alias(tokenizer, text)]
        if not token_ids:
            return {
                "ok": False,
                "reason": "prefix_tokenize_failed",
                "matrix": [],
                "all_token_hidden_by_layer": [],
                "last_token_attention_by_layer": [],
            }

        input_ids = build_protocol_input_ids(tokenizer, protocol, token_ids)
        device = next(model.parameters()).device
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        model_cfg = getattr(model, "config", None)
        prev_attn_impl = getattr(model_cfg, "_attn_implementation", None) if model_cfg is not None else None
        if model_cfg is not None:
            setattr(model_cfg, "_attn_implementation", "eager")
        try:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_tensor,
                    output_hidden_states=True,
                    output_attentions=True,
                    return_dict=True,
                    use_cache=False,
                )
        finally:
            if model_cfg is not None:
                setattr(model_cfg, "_attn_implementation", prev_attn_impl)

        hidden_states = getattr(outputs, "hidden_states", None) or []
        if not hidden_states:
            return {
                "ok": False,
                "reason": "missing_hidden_states",
                "matrix": [],
                "last_token_attention_by_layer": [],
            }

        matrix = []
        all_token_hidden_by_layer = []
        for h in hidden_states:
            v = h[0, -1, :].detach().float().cpu().numpy()
            matrix.append(v.tolist())
            seq_hidden = h[0, :, :].detach().float().cpu().numpy()
            all_token_hidden_by_layer.append(seq_hidden.tolist())
        matrix_arr = np.asarray(matrix, dtype=np.float32)
        row_labels = ["embedding"] + [f"layer_{idx}" for idx in range(1, int(matrix_arr.shape[0]))]

        attentions = getattr(outputs, "attentions", None) or []
        attention_rows: list[dict[str, Any]] = []
        for layer_idx, attn in enumerate(attentions):
            # attn: [B, heads, S, S], take last query token -> [heads, S], head-average -> [S]
            last_query = attn[0, :, -1, :].detach().float().cpu()
            head_avg = last_query.mean(dim=0).numpy().astype(np.float32)
            attention_rows.append(
                {
                    "layer": int(layer_idx),
                    "last_token_to_all_tokens": head_avg.tolist(),
                }
            )
        if not attention_rows:
            return {
                "ok": False,
                "reason": "attention_unavailable",
                "matrix": matrix_arr.tolist(),
                "all_token_hidden_by_layer": all_token_hidden_by_layer,
                "last_token_attention_by_layer": [],
            }

        return {
            "ok": True,
            "sentence": text,
            "protocol": protocol,
            "prefix_token_ids": token_ids,
            "input_ids": [int(x) for x in input_ids],
            "input_tokens": tokenizer.convert_ids_to_tokens(input_ids),
            "rows": int(matrix_arr.shape[0]),
            "cols": int(matrix_arr.shape[1]),
            "sequence_length": int(len(input_ids)),
            "row_labels": row_labels,
            "matrix": matrix_arr.tolist(),
            "all_token_hidden_by_layer": all_token_hidden_by_layer,
            "last_token_attention_by_layer": attention_rows,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "reason": str(exc),
            "matrix": [],
            "all_token_hidden_by_layer": [],
            "last_token_attention_by_layer": [],
        }
