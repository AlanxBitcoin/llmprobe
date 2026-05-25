from __future__ import annotations

# Study: Layer FFN Neuron (post-SiLU) Single-Activation Logits Table
# - Choose one decoder layer (default 30).
# - For each FFN neuron id in intermediate_size:
#   activate only that post-SiLU FFN neuron with value=activation_value.
# - Convert to layer output vector via down_proj, then continue forward from same layer.
# - Rank LM-head top-15 logits from final-layer output.
# - Return a table payload:
#   rows = neuron id, columns = rank-wise text/logit pairs.

from pathlib import Path
from typing import Any
from datetime import datetime

import torch

from ..config import load_config
from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api
from ..utils.logits import rank_vector_by_logits
from ..probes.single_word_hidden_state_probe import run_multi_ffn_neurons_from_layer_probe
from ..utils.utils import ensure_dir, write_csv


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


def _history_dir() -> Path:
    return ensure_dir(Path("data") / "outputs" / "layer_ffn_neuron_logits_table" / "history")


def _build_history_csv_rows(rows: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {"neuron_id": int(row.get("neuron_id", -1))}
        logits = row.get("top_logits") or []
        for rank in range(1, int(top_k) + 1):
            src = logits[rank - 1] if rank - 1 < len(logits) else {}
            item[f"rank_{rank}_text"] = str(src.get("text", ""))
            item[f"rank_{rank}_logit"] = src.get("logit", "")
        out.append(item)
    return out


def run_study(
    *,
    intervention_layer: int = 30,
    activation_value: float = 10.0,
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
            "valid_layer_min": 1,
            "valid_layer_max": int(num_layers),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }
    if not numeric_layer.is_integer():
        return {
            "ok": False,
            "reason": "intervention_layer_must_be_integer",
            "requested_layer": raw_layer,
            "valid_layer_min": 1,
            "valid_layer_max": int(num_layers),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }
    layer_number = int(numeric_layer)
    if not (1 <= layer_number <= num_layers):
        return {
            "ok": False,
            "reason": "invalid_intervention_layer",
            "requested_layer": int(layer_number),
            "valid_layer_min": 1,
            "valid_layer_max": int(num_layers),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }
    layer_idx = int(layer_number - 1)

    layer = layers[layer_idx]
    mlp = getattr(layer, "mlp", None)
    down_proj = getattr(mlp, "down_proj", None) if mlp is not None else None
    ffn_dim = int(getattr(down_proj, "in_features", 0) or 0)
    if ffn_dim <= 0:
        weight = getattr(down_proj, "weight", None) if down_proj is not None else None
        if weight is not None and weight.ndim == 2:
            ffn_dim = int(weight.shape[1])
    if ffn_dim <= 0:
        return {
            "ok": False,
            "reason": "ffn_dim_unavailable",
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }

    device = next(model.parameters()).device
    token_id = _resolve_bootstrap_token_id(tokenizer)
    input_ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)

    top_k = 15
    batch_size = max(1, int(return_batch_size))
    compute_batch_size = max(
        1,
        int((((cfg or {}).get("runtime") or {}).get("ffn_parallel_batch_size", 64))),
    )
    rows: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    current_batch: list[dict[str, Any]] = []

    for start in range(0, int(ffn_dim), int(compute_batch_size)):
        neuron_ids = list(range(start, min(start + int(compute_batch_size), int(ffn_dim))))
        transit, transit_error = run_multi_ffn_neurons_from_layer_probe(
            config=cfg,
            layer_idx=layer_idx,
            ffn_neuron_indices=neuron_ids,
            activation_value=float(activation_value),
            input_ids_override=input_ids,
            bundle_override=bundle,
        )
        if transit is None:
            return {
                "ok": False,
                "reason": "probe_single_ffn_neuron_failed",
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
        final_batch = hidden_states[-1].detach()  # [B,S,H]
        bsz = int(final_batch.shape[0])
        for local_idx in range(bsz):
            neuron_id = int(neuron_ids[local_idx])
            final_vec = final_batch[local_idx, -1, :]
            logits_rows = rank_vector_by_logits(
                model=model,
                vector=final_vec,
                tokenizer=tokenizer,
                top_k=top_k,
                apply_final_norm=False,
            )
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
                    f"[layer_ffn_neuron_logits_table] batch_done index={len(batches)-1} range={start_id}-{end_id} processed={len(rows)}/{ffn_dim}"
                )
                current_batch = []
        if len(rows) % 128 == 0 or len(rows) == ffn_dim:
            print(
                f"[layer_ffn_neuron_logits_table] progress processed={len(rows)}/{ffn_dim} kept={len(rows)} compute_batch_size={compute_batch_size}"
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
            f"[layer_ffn_neuron_logits_table] batch_done index={len(batches)-1} range={start_id}-{end_id} processed={ffn_dim}/{ffn_dim}"
        )

    # Persist study result as CSV history for quick reload in UI.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = (
        f"ffn_layer{int(layer_number)}"
        f"_act{str(float(activation_value)).replace('.', 'p')}"
        f"_{timestamp}.csv"
    )
    csv_path = _history_dir() / csv_name
    history_rows = _build_history_csv_rows(rows, top_k=top_k)
    history_rel = ""
    if history_rows:
        write_csv(csv_path, history_rows)
        history_rel = str(csv_path.as_posix())

    return {
        "ok": True,
        "study": "layer_ffn_neuron_single_activation_logits",
        "intervention_layer": int(layer_number),
        "activation_value": float(activation_value),
        "threshold": 15.0,
        "top_k": int(top_k),
        "hidden_dim": int(ffn_dim),
        "neuron_kind": "ffn_post_silu",
        "returned_rows": int(len(rows)),
        "filtered_out_rows": 0,
        "return_batch_size": int(batch_size),
        "compute_batch_size": int(compute_batch_size),
        "neuron_logits_rows": rows,
        "neuron_logits_batches": batches,
        "history_csv_path": history_rel,
        "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_batches"}],
    }
