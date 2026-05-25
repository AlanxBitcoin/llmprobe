from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Probe-layer API for single-word hidden-state retrieval.
# - Probe should access runtime API + hidden_store/model internals.
# - Study layer composes probe outputs for UI payloads.

from typing import Any
import torch

from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api
from ..utils.extract_hidden import extract_single_word_hidden_matrix_store_first
from ..utils.hooks import (
    build_ffn_post_silu_neuron_output_matrix,
    build_ffn_post_silu_neuron_output_vector,
    starting_from_middle_layer,
)
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
    except Exception as exc:  # noqa: BLE001 - logits failure should not block heatmap output.
        return [], "error", str(exc)


def _build_protocol_input_ids_from_word(tokenizer, *, word: str, protocol: str) -> list[int]:
    token_ids = parse_token_ids_with_bos_alias(tokenizer, word)
    if len(token_ids) != 1:
        raise ValueError(f"single_token_required: token_count={len(token_ids)}")
    return build_protocol_input_ids(tokenizer, protocol, [int(token_ids[0])])


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
    word: str | None,
    config: dict[str, Any],
    start_layer_idx: int,
    hidden_state: torch.Tensor,
    input_ids_override: torch.Tensor | None = None,
    bundle_override=None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Probe-layer transit function: study -> probe -> hooks.starting_from_middle_layer."""
    try:
        if bundle_override is not None:
            bundle = bundle_override
        else:
            api = _get_or_start_runtime_api(config)
            bundle = api.get_bundle()
        model = bundle.model
        tokenizer = bundle.tokenizer
        protocol = str(((config or {}).get("hidden_store") or {}).get("protocol", "bos1_assistant0"))
        device = next(model.parameters()).device
        if input_ids_override is not None:
            input_tensor = input_ids_override.to(device=device, dtype=torch.long)
            if input_tensor.ndim == 1:
                input_tensor = input_tensor.unsqueeze(0)
        else:
            if not word:
                return None, "word is required when input_ids_override is not provided"
            input_ids = _build_protocol_input_ids_from_word(tokenizer, word=str(word).strip(), protocol=protocol)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Fast path for single-token override:
        # if sequence length is 1, replacement fully defines that step,
        # so we can skip the reference full forward.
        if (
            int(input_tensor.shape[0]) == 1
            and int(input_tensor.shape[1]) == 1
            and int(hidden_state.ndim) == 1
        ):
            replacement = hidden_state.to(device=device, dtype=next(model.parameters()).dtype).flatten()
            start_hidden_full = replacement.view(1, 1, -1)
        else:
            # General path: build full-sequence hidden state at target layer first,
            # then replace only the last-token vector.
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


def run_single_ffn_neuron_from_layer_probe(
    *,
    config: dict[str, Any],
    layer_idx: int,
    ffn_neuron_idx: int,
    activation_value: float,
    input_ids_override: torch.Tensor | None = None,
    bundle_override=None,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Probe-layer transit:
    1) Build layer-output hidden vector from one post-SiLU FFN neuron activation.
    2) Continue from the same decoder layer to the end via starting_from_middle_layer.
    """
    try:
        if bundle_override is not None:
            bundle = bundle_override
        else:
            api = _get_or_start_runtime_api(config)
            bundle = api.get_bundle()
        model = bundle.model
        hidden_vec = build_ffn_post_silu_neuron_output_vector(
            model,
            layer_idx=int(layer_idx),
            neuron_idx=int(ffn_neuron_idx),
            activation_value=float(activation_value),
        )
        return run_starting_from_middle_layer_probe(
            word=None,
            config=config,
            start_layer_idx=int(layer_idx),
            hidden_state=hidden_vec,
            input_ids_override=input_ids_override,
            bundle_override=bundle,
        )
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def run_multi_ffn_neurons_from_layer_probe(
    *,
    config: dict[str, Any],
    layer_idx: int,
    ffn_neuron_indices: list[int],
    activation_value: float,
    input_ids_override: torch.Tensor | None = None,
    bundle_override=None,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Batched FFN-neuron transit:
    1) Build B layer-output vectors from B post-SiLU neuron activations.
    2) Continue from same decoder layer with batch size B in one forward path.
    """
    try:
        if not ffn_neuron_indices:
            return None, "ffn_neuron_indices is empty"
        if bundle_override is not None:
            bundle = bundle_override
        else:
            api = _get_or_start_runtime_api(config)
            bundle = api.get_bundle()
        model = bundle.model

        hidden_matrix = build_ffn_post_silu_neuron_output_matrix(
            model,
            layer_idx=int(layer_idx),
            neuron_indices=[int(x) for x in ffn_neuron_indices],
            activation_value=float(activation_value),
        )  # [B,H]

        bsz = int(hidden_matrix.shape[0])
        hidden_full = hidden_matrix.unsqueeze(1)  # [B,1,H]
        device = next(model.parameters()).device
        if input_ids_override is not None:
            input_tensor = input_ids_override.to(device=device, dtype=torch.long)
            if input_tensor.ndim == 1:
                input_tensor = input_tensor.unsqueeze(0)
            if int(input_tensor.shape[0]) == 1 and bsz > 1:
                input_tensor = input_tensor.expand(bsz, -1).contiguous()
            elif int(input_tensor.shape[0]) != bsz:
                return None, (
                    f"input_ids_override batch mismatch: expected 1 or {bsz}, got {int(input_tensor.shape[0])}"
                )
        else:
            # Bootstrap token only; sequence length 1 is enough for this probe.
            bos_id = getattr(bundle.tokenizer, "bos_token_id", None)
            if bos_id is None:
                encoded = bundle.tokenizer.encode("", add_special_tokens=True)
                if not encoded:
                    return None, "Tokenizer does not provide bootstrap token id"
                bos_id = int(encoded[0])
            input_tensor = torch.full((bsz, 1), int(bos_id), dtype=torch.long, device=device)

        result = starting_from_middle_layer(
            model,
            start_layer_idx=int(layer_idx),
            hidden_state=hidden_full,
            input_ids=input_tensor,
            output_hidden_states=True,
            return_dict=True,
        )
        return {"bundle": bundle, "result": result}, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
