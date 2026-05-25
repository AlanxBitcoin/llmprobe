from __future__ import annotations

# Study: Layer Neuron Single-Activation Logits Table
# - Choose one decoder layer (default 30).
# - For each neuron id in hidden_dim:
#   set only that neuron to value=10 (others=0) as layer output for last token.
# - Continue forward from that middle layer to the end.
# - Rank LM-head top-15 logits from final-layer output.
# - Return a table payload:
#   rows = neuron id, columns = rank-wise text/logit pairs.

from pathlib import Path
from typing import Any

import torch

from ..config import load_config
from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api
from ..utils.logits import rank_vector_by_logits
from ..probes.single_word_hidden_state_probe import run_starting_from_middle_layer_probe


def _get_or_start_runtime_api(config: dict[str, Any]):
    try:
        return get_runtime_api()
    except RuntimeError:
        return start_llama_api(config)


def _resolve_bootstrap_token_id(tokenizer) -> int:
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        return int(bos_id)
    encoded = tokenizer.encode("", add_special_tokens=True)
    if encoded:
        return int(encoded[0])
    raise ValueError("Tokenizer does not provide a BOS/special token for bootstrap input.")


def run_study(
    *,
    intervention_layer: int = 30,
    activation_value: float = 10.0,
    threshold: float = 15.0,
    return_batch_size: int = 128,
    config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/custom.yaml",
) -> dict[str, Any]:
    cfg = config or load_config(config_path)
    api = _get_or_start_runtime_api(cfg)
    bundle = api.execute_model_call(RuntimeRequest(config=cfg, force_reload=False)).bundle
    model = bundle.model
    tokenizer = bundle.tokenizer

    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        return {
            "ok": False,
            "reason": "model_missing_layers",
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }
    num_layers = int(len(layers))
    if num_layers <= 0:
        return {
            "ok": False,
            "reason": "model_has_zero_layers",
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }

    raw_layer = intervention_layer
    try:
        numeric_layer = float(raw_layer)
    except (TypeError, ValueError):
        return {
            "ok": False,
            "reason": "invalid_intervention_layer_type",
            "requested_layer": raw_layer,
            "valid_layer_min": 0,
            "valid_layer_max": int(num_layers - 1),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }
    if not numeric_layer.is_integer():
        return {
            "ok": False,
            "reason": "intervention_layer_must_be_integer",
            "requested_layer": raw_layer,
            "valid_layer_min": 0,
            "valid_layer_max": int(num_layers - 1),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }
    layer_idx = int(numeric_layer)
    if not (0 <= layer_idx < num_layers):
        return {
            "ok": False,
            "reason": "invalid_intervention_layer",
            "requested_layer": int(layer_idx),
            "valid_layer_min": 0,
            "valid_layer_max": int(num_layers - 1),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }

    hidden_dim = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
    if hidden_dim <= 0:
        lm_head = getattr(model, "lm_head", None)
        hidden_dim = int(getattr(lm_head, "in_features", 0) or 0)
    if hidden_dim <= 0:
        return {
            "ok": False,
            "reason": "hidden_size_unavailable",
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }

    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    token_id = _resolve_bootstrap_token_id(tokenizer)
    input_ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)

    top_k = 15
    batch_size = max(1, int(return_batch_size))
    top1_threshold = float(threshold)
    rows: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    current_batch: list[dict[str, Any]] = []
    filtered_out_count = 0

    for neuron_id in range(hidden_dim):
        one_hot = torch.zeros(hidden_dim, dtype=model_dtype, device=device)
        one_hot[neuron_id] = float(activation_value)
        transit, transit_error = run_starting_from_middle_layer_probe(
            word=None,
            config=cfg,
            start_layer_idx=layer_idx,
            hidden_state=one_hot,
            input_ids_override=input_ids,
            bundle_override=bundle,
        )
        if transit is None:
            return {
                "ok": False,
                "reason": "probe_starting_from_middle_layer_failed",
                "error": str(transit_error or "unknown"),
                "neuron_logits_rows": rows,
                "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
            }
        outputs = transit["result"]["outputs"]
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            return {
                "ok": False,
                "reason": "missing_hidden_states",
                "neuron_logits_rows": rows,
                "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
            }
        final_vec = hidden_states[-1][0, -1, :].detach()
        logits_rows = rank_vector_by_logits(
            model=model,
            vector=final_vec,
            tokenizer=tokenizer,
            top_k=top_k,
            apply_final_norm=False,
        )
        top1_logit = float(logits_rows[0]["logit"]) if logits_rows else float("-inf")
        if top1_logit < top1_threshold:
            filtered_out_count += 1
            if (neuron_id + 1) % 128 == 0 or neuron_id + 1 == hidden_dim:
                print(
                    f"[layer_neuron_logits_table] progress processed={neuron_id+1}/{hidden_dim} kept={len(rows)} filtered={filtered_out_count}"
                )
            continue
        row = {
            "neuron_id": int(neuron_id),
            "top_logits": logits_rows,
        }
        rows.append(row)
        current_batch.append(row)
        if len(current_batch) >= batch_size:
            start_id = int(current_batch[0]["neuron_id"])
            end_id = int(current_batch[-1]["neuron_id"])
            batches.append(
                {
                    "batch_index": int(len(batches)),
                    "start_neuron_id": start_id,
                    "end_neuron_id": end_id,
                    "rows": current_batch,
                }
            )
            print(
                f"[layer_neuron_logits_table] batch_done index={len(batches)-1} range={start_id}-{end_id} processed={neuron_id+1}/{hidden_dim}"
            )
            current_batch = []
        if (neuron_id + 1) % 128 == 0 or neuron_id + 1 == hidden_dim:
            print(
                f"[layer_neuron_logits_table] progress processed={neuron_id+1}/{hidden_dim} kept={len(rows)} filtered={filtered_out_count}"
            )
    if current_batch:
        start_id = int(current_batch[0]["neuron_id"])
        end_id = int(current_batch[-1]["neuron_id"])
        batches.append(
            {
                "batch_index": int(len(batches)),
                "start_neuron_id": start_id,
                "end_neuron_id": end_id,
                "rows": current_batch,
            }
        )
        print(
            f"[layer_neuron_logits_table] batch_done index={len(batches)-1} range={start_id}-{end_id} processed={hidden_dim}/{hidden_dim}"
        )

    return {
        "ok": True,
        "study": "layer_neuron_single_activation_logits",
        "intervention_layer": int(layer_idx),
        "activation_value": float(activation_value),
        "threshold": float(top1_threshold),
        "top_k": int(top_k),
        "hidden_dim": int(hidden_dim),
        "returned_rows": int(len(rows)),
        "filtered_out_rows": int(filtered_out_count),
        "return_batch_size": int(batch_size),
        "neuron_logits_rows": rows,
        "neuron_logits_batches": batches,
        "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_batches"}],
    }
