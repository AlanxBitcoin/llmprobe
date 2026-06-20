from __future__ import annotations

"""逐个激活隐藏层神经元并生成 logits 排名表。

功能:
- 在指定 decoder 层（默认 30）对隐藏维神经元逐个做单点激活。
- 从该中间层继续前向推理至模型末层。
- 对每次干预结果计算 LM Head top-k logits。
- 以“神经元 x 排名位”的表格形式返回结果，便于对比分析。
"""

from pathlib import Path
from typing import Any

import torch

from ..config import load_config
from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api
from ..probes.probe_layer_neuron import run_layer_neuron_batch_to_logits_probe
from ..probes.probe_hidden_state import fetch_sentence_last_token_hidden_state


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
    use_prefix_context: bool = False,
    prefix_text: str = "",
    return_batch_size: int = 1000,
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
    base_last_hidden = None
    base_sequence_hidden = None
    prefix_token_ids: list[int] = []
    prefix_attention_by_layer: list[dict[str, Any]] = []
    attention_reused_for_intervention = False
    prefix_enabled = bool(use_prefix_context)
    if prefix_enabled:
        text = str(prefix_text or "").strip()
        if not text:
            return {
                "ok": False,
                "reason": "prefix_text_required",
                "message": "prefix_text_required",
                "neuron_logits_rows": [],
                "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
            }
        prefix_ctx = fetch_sentence_last_token_hidden_state(
            sentence=text,
            config=cfg,
        )
        if not isinstance(prefix_ctx, dict) or not prefix_ctx.get("ok"):
            reason = (
                str(prefix_ctx.get("reason"))
                if isinstance(prefix_ctx, dict) and prefix_ctx.get("reason")
                else "prefix_hidden_fetch_failed"
            )
            return {
                "ok": False,
                "reason": reason,
                "message": reason,
                "prefix_word": text,
                "prefix_detail": prefix_ctx if isinstance(prefix_ctx, dict) else {},
                "neuron_logits_rows": [],
                "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
            }
        try:
            prefix_token_ids = [int(x) for x in (prefix_ctx.get("prefix_token_ids") or [])]
            matrix = prefix_ctx.get("matrix") or []
            row_idx = int(layer_idx) + 1  # row0=embedding, row(i+1)=layer_i output
            if not (0 <= row_idx < len(matrix)):
                return {
                    "ok": False,
                    "reason": f"prefix_hidden_unavailable_for_layer:{layer_idx}",
                    "message": f"prefix_hidden_unavailable_for_layer:{layer_idx}",
                    "neuron_logits_rows": [],
                    "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
                }
            base_last_hidden = torch.as_tensor(matrix[row_idx], device=device, dtype=model_dtype).flatten()
            input_ids_list = [int(x) for x in (prefix_ctx.get("input_ids") or [])]
            if not input_ids_list:
                return {
                    "ok": False,
                    "reason": "prefix_input_ids_missing",
                    "message": "prefix_input_ids_missing",
                    "neuron_logits_rows": [],
                    "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
                }
            input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
            prefix_attention_by_layer = list(prefix_ctx.get("last_token_attention_by_layer") or [])
            # Reuse probe-returned full sequence hidden states (no extra sentence forward).
            prefix_hidden_states = prefix_ctx.get("all_token_hidden_by_layer") or []
            if not prefix_hidden_states:
                return {
                    "ok": False,
                    "reason": "prefix_hidden_states_missing",
                    "message": "prefix_hidden_states_missing",
                    "neuron_logits_rows": [],
                    "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
                }
            seq_row_idx = int(layer_idx) + 1  # row0=embedding, row(i+1)=layer_i output
            if not (0 <= seq_row_idx < len(prefix_hidden_states)):
                return {
                    "ok": False,
                    "reason": f"prefix_hidden_unavailable_for_layer:{layer_idx}",
                    "message": f"prefix_hidden_unavailable_for_layer:{layer_idx}",
                    "neuron_logits_rows": [],
                    "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
                }
            base_sequence_hidden = torch.as_tensor(
                prefix_hidden_states[seq_row_idx],
                device=device,
                dtype=model_dtype,
            ).unsqueeze(0)
            # Perf rule: neuron intervention path reuses initial sentence attention snapshots.
            attention_reused_for_intervention = True
        except Exception as exc:  # noqa: BLE001
            reason = str(exc)
            return {
                "ok": False,
                "reason": reason,
                "message": reason,
                "neuron_logits_rows": [],
                "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
            }

    top_k = 15
    batch_size = max(1, int(return_batch_size))
    compute_batch_size = max(
        1,
        int((((cfg or {}).get("runtime") or {}).get("neuron_parallel_batch_size", 64))),
    )
    rows: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    current_batch: list[dict[str, Any]] = []

    for start in range(0, int(hidden_dim), int(compute_batch_size)):
        neuron_ids = list(range(start, min(start + int(compute_batch_size), int(hidden_dim))))
        one_hot_batch = torch.zeros((len(neuron_ids), hidden_dim), dtype=model_dtype, device=device)
        row_idx = torch.arange(len(neuron_ids), device=device, dtype=torch.long)
        col_idx = torch.tensor(neuron_ids, device=device, dtype=torch.long)
        one_hot_batch[row_idx, col_idx] = float(activation_value)
        if prefix_enabled and base_sequence_hidden is not None:
            hidden_batch = base_sequence_hidden.expand(len(neuron_ids), -1, -1).clone()
            hidden_batch[:, -1, :] = hidden_batch[:, -1, :] + one_hot_batch
        elif prefix_enabled and base_last_hidden is not None:
            hidden_batch = base_last_hidden.unsqueeze(0).expand(len(neuron_ids), -1).clone()
            hidden_batch = hidden_batch + one_hot_batch
        else:
            hidden_batch = one_hot_batch
        input_ids_batch = input_ids.expand(len(neuron_ids), -1).contiguous()
        logits_batch_rows, logits_error = run_layer_neuron_batch_to_logits_probe(
            bundle=bundle,
            config=cfg,
            start_layer_idx=layer_idx,
            input_ids=input_ids_batch,
            hidden_batch=hidden_batch,
            top_k=top_k,
        )
        if logits_batch_rows is None:
            return {
                "ok": False,
                "reason": "probe_starting_from_middle_layer_failed",
                "error": str(logits_error or "unknown"),
                "neuron_logits_rows": rows,
                "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
            }
        bsz = int(len(logits_batch_rows))
        for local_idx in range(bsz):
            neuron_id = int(neuron_ids[local_idx])
            logits_rows = logits_batch_rows[local_idx]
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
                    f"[layer_neuron_logits_table] batch_done index={len(batches)-1} range={start_id}-{end_id} processed={len(rows)}/{hidden_dim}"
                )
                current_batch = []
        if len(rows) % 128 == 0 or len(rows) == hidden_dim:
            print(
                f"[layer_neuron_logits_table] progress processed={len(rows)}/{hidden_dim} kept={len(rows)} compute_batch_size={compute_batch_size}"
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
        "intervention_layer": int(layer_number),
        "activation_value": float(activation_value),
        "use_prefix_context": bool(prefix_enabled),
        "prefix_text": str(prefix_text or ""),
        "prefix_token_count": int(len(prefix_token_ids)),
        "prefix_last_token_attention_by_layer": prefix_attention_by_layer,
        "attention_reused_for_intervention": bool(attention_reused_for_intervention),
        "threshold": 0.0,
        "top_k": int(top_k),
        "hidden_dim": int(hidden_dim),
        "returned_rows": int(len(rows)),
        "filtered_out_rows": 0,
        "return_batch_size": int(batch_size),
        "compute_batch_size": int(compute_batch_size),
        "neuron_logits_rows": rows,
        "neuron_logits_batches": batches,
        "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_batches"}],
    }
