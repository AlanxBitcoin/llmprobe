from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Probe-layer API for single-word hidden-state retrieval.
# - Probe should access runtime API + hidden_store/model internals.
# - Study layer composes probe outputs for UI payloads.

from typing import Any
import torch

from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api
from ..utils.extract_hidden import extract_single_word_hidden_matrix_store_first
from ..utils.hooks import starting_from_middle_layer
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
            apply_final_norm=False,
        )
        return rows, "probe", None
    except Exception as exc:  # noqa: BLE001 - logits failure should not block heatmap output.
        return [], "error", str(exc)


def _build_protocol_input_ids_from_word(tokenizer, *, word: str, protocol: str) -> list[int]:
    token_ids = [int(x) for x in (tokenizer(word, add_special_tokens=False).get("input_ids") or [])]
    if len(token_ids) != 1:
        raise ValueError(f"single_token_required: token_count={len(token_ids)}")
    token_id = int(token_ids[0])

    if protocol == "bos0_assistant0":
        return [token_id]
    if protocol == "bos1_assistant0":
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            raise ValueError("Tokenizer has no bos_token_id but protocol requires BOS")
        return [int(bos_id), token_id]
    if protocol == "bos1_assistant1":
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support apply_chat_template for assistant protocol")
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if not prefix:
            raise ValueError("Chat template returned empty prefix")
        return [int(x) for x in prefix] + [token_id]
    raise ValueError(f"Unsupported hidden_store protocol: {protocol}")


def _sparsify_abs_topk(vector_1d: torch.Tensor, keep_k: int) -> tuple[torch.Tensor, list[int]]:
    vec = vector_1d.flatten()
    width = int(vec.shape[0])
    if width <= 0:
        return vec, []
    k = max(1, min(int(keep_k), width))
    abs_vec = torch.abs(vec)
    _, idx = torch.topk(abs_vec, k=k)
    sparse = torch.zeros_like(vec)
    sparse[idx] = vec[idx]
    return sparse, [int(x) for x in idx.detach().cpu().tolist()]


def run_starting_from_middle_layer_probe(
    *,
    word: str,
    config: dict[str, Any],
    start_layer_idx: int,
    hidden_state: torch.Tensor,
) -> tuple[dict[str, Any] | None, str | None]:
    """Probe-layer transit function: study -> probe -> hooks.starting_from_middle_layer."""
    try:
        api = _get_or_start_runtime_api(config)
        bundle = api.get_bundle()
        model = bundle.model
        tokenizer = bundle.tokenizer
        protocol = str(((config or {}).get("hidden_store") or {}).get("protocol", "bos1_assistant0"))
        input_ids = _build_protocol_input_ids_from_word(tokenizer, word=str(word).strip(), protocol=protocol)
        device = next(model.parameters()).device
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Build full-sequence hidden state at the target layer first, then replace
        # only the last-token vector. This preserves sequence context for tail layers.
        with torch.no_grad():
            ref_outputs = model(
                input_ids=input_tensor,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        ref_hidden_states = getattr(ref_outputs, "hidden_states", None)
        if ref_hidden_states is None:
            return None, "Model did not return hidden_states for middle-layer continuation"
        row_idx = int(start_layer_idx) + 1  # row0=embedding
        if not (0 <= row_idx < len(ref_hidden_states)):
            return None, f"start_layer_idx out of range for hidden_states: {start_layer_idx}"
        start_hidden_full = ref_hidden_states[row_idx].detach().clone()
        replacement = hidden_state.to(device=start_hidden_full.device, dtype=start_hidden_full.dtype).flatten()
        if replacement.shape[0] != start_hidden_full.shape[-1]:
            return None, (
                f"Hidden size mismatch: expected {start_hidden_full.shape[-1]}, got {replacement.shape[0]}"
            )
        start_hidden_full[:, -1, :] = replacement.unsqueeze(0)

        result = starting_from_middle_layer(
            model,
            start_layer_idx=int(start_layer_idx),
            hidden_state=start_hidden_full,
            input_ids=input_tensor,
            output_hidden_states=True,
            return_dict=True,
        )
        return {"bundle": bundle, "result": result, "protocol": protocol}, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def rank_logits_after_penultimate_topk_intervention(
    *,
    heatmap: dict[str, Any],
    word: str,
    config: dict[str, Any],
    keep_k: int = 100,
    intervention_layer: int = 30,
    top_k: int = 15,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str | None]:
    """Intervention path:
    use penultimate-layer vector -> keep abs top-k neurons -> inject at layer (N-2)
    last-token output via forward hook -> continue normal forward to logits.
    """
    if not isinstance(heatmap, dict) or not heatmap.get("ok"):
        return [], {}, "none", None

    matrix = heatmap.get("matrix") or []
    if not matrix or len(matrix) < 3:
        return [], {}, "none", "Insufficient hidden-state rows for penultimate intervention"

    try:
        api = _get_or_start_runtime_api(config)
        bundle = api.get_bundle()
        model = bundle.model
        tokenizer = bundle.tokenizer

        base_model = getattr(model, "model", None)
        layers = getattr(base_model, "layers", None)
        if layers is None or len(layers) < 2:
            return [], {}, "error", "Model does not expose enough decoder layers"

        injection_layer_index = int(intervention_layer)
        if not (0 <= injection_layer_index < len(layers)):
            return [], {}, "error", f"intervention_layer out of range: {injection_layer_index}"

        device = next(model.parameters()).device
        matrix_row_idx = injection_layer_index + 1  # row0=embedding, row(i+1)=layer_i output
        if not (0 <= matrix_row_idx < len(matrix)):
            return [], {}, "error", f"hidden-state row out of range for layer {injection_layer_index}"
        desired = torch.as_tensor(matrix[matrix_row_idx], dtype=torch.float32, device=device).flatten()
        sparse_vec, keep_indices = _sparsify_abs_topk(desired, keep_k=int(keep_k))
        sparse_vec = sparse_vec.to(dtype=next(model.parameters()).dtype)
        transit, transit_error = run_starting_from_middle_layer_probe(
            word=word,
            config=config,
            start_layer_idx=int(injection_layer_index),
            hidden_state=sparse_vec,
        )
        if transit is None:
            return [], {}, "error", str(transit_error or "starting_from_middle_layer probe failed")
        outputs = transit["result"]["outputs"]
        final_vec = outputs.hidden_states[-1][0, -1, :].detach()

        rows = rank_vector_logits_and_cosine(
            model=model,
            vector=final_vec,
            tokenizer=tokenizer,
            top_k=int(top_k),
            apply_final_norm=False,
        )
        meta = {
            "keep_k": int(min(max(1, int(keep_k)), int(desired.shape[0]))),
            "intervened_vector_from_layer_row": int(matrix_row_idx),
            "injection_layer_index": int(injection_layer_index),
            "recompute_layers": int(max(0, len(layers) - injection_layer_index - 1)),
            "protocol": str(transit.get("protocol", "")),
            "kept_neuron_indices": keep_indices,
        }
        return rows, meta, "probe_intervention", None
    except Exception as exc:  # noqa: BLE001
        return [], {}, "error", str(exc)
