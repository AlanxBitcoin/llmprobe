from __future__ import annotations

from typing import Any

import torch

from ..runtime_api import get_runtime_api, start_llama_api
from ..utils.hooks import (
    build_ffn_post_silu_neuron_output_matrix,
    build_ffn_post_silu_neuron_output_vector,
    starting_from_middle_layer,
)
from .probe_layer_neuron import run_starting_from_middle_layer_probe


def _get_or_start_runtime_api(config: dict[str, Any]):
    try:
        return get_runtime_api()
    except RuntimeError:
        return start_llama_api(config)


def run_single_ffn_neuron_from_layer_probe(
    *,
    config: dict[str, Any],
    layer_idx: int,
    ffn_neuron_idx: int,
    activation_value: float,
    input_ids_override: torch.Tensor | None = None,
    bundle_override=None,
) -> tuple[dict[str, Any] | None, str | None]:
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
        )

        bsz = int(hidden_matrix.shape[0])
        hidden_full = hidden_matrix.unsqueeze(1)
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
