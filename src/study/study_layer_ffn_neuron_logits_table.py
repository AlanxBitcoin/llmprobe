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
import torch.nn.functional as F

from ..config import load_config
from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api
from ..probes.probe_layer_neuron import run_layer_neuron_batch_to_logits_probe
from ..probes.probe_hidden_state import fetch_sentence_last_token_hidden_state
from ..utils.hooks import build_ffn_post_silu_neuron_output_matrix
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


def _rank_hidden_batch_by_logits(
    *,
    model,
    tokenizer,
    hidden_batch: torch.Tensor,
    top_k: int = 15,
    apply_final_norm: bool = False,
) -> list[list[dict[str, Any]]]:
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return [[] for _ in range(int(hidden_batch.shape[0]))]
    if hidden_batch.ndim != 2:
        raise ValueError(f"hidden_batch must be [B,H], got shape={tuple(hidden_batch.shape)}")

    hidden_for_logits = hidden_batch.to(dtype=lm_head.weight.dtype)
    if apply_final_norm:
        base_model = getattr(model, "model", None)
        final_norm = getattr(base_model, "norm", None)
        if final_norm is not None:
            hidden_for_logits = final_norm(hidden_for_logits)

    with torch.no_grad():
        cfg = getattr(model, "config", None)
        pretraining_tp = int(getattr(cfg, "pretraining_tp", 1) or 1)
        if pretraining_tp > 1:
            vocab_size = int(getattr(cfg, "vocab_size", lm_head.weight.shape[0]))
            lm_head_slices = lm_head.weight.split(vocab_size // pretraining_tp, dim=0)
            logits_chunks = [F.linear(hidden_for_logits, lm_head_slice) for lm_head_slice in lm_head_slices]
            logits = torch.cat(logits_chunks, dim=-1).to(dtype=torch.float32)
        else:
            logits = lm_head(hidden_for_logits).to(dtype=torch.float32)
        k = min(int(top_k), int(logits.shape[-1]))
        top_vals, top_ids = torch.topk(logits, k=k, dim=-1)

    top_ids_cpu = top_ids.detach().cpu().tolist()
    top_vals_cpu = top_vals.detach().cpu().tolist()
    flat_ids = [int(tok_id) for row in top_ids_cpu for tok_id in row]
    if tokenizer is not None and flat_ids:
        flat_tokens = tokenizer.convert_ids_to_tokens(flat_ids)
        flat_texts = [tokenizer.decode([int(tok_id)], clean_up_tokenization_spaces=False) for tok_id in flat_ids]
    else:
        flat_tokens = ["" for _ in flat_ids]
        flat_texts = ["" for _ in flat_ids]

    all_rows: list[list[dict[str, Any]]] = []
    cursor = 0
    for b, ids_row in enumerate(top_ids_cpu):
        vals_row = top_vals_cpu[b]
        rows: list[dict[str, Any]] = []
        for i, tok_id in enumerate(ids_row, start=1):
            rows.append(
                {
                    "rank": int(i),
                    "token_id": int(tok_id),
                    "token": str(flat_tokens[cursor]),
                    "text": str(flat_texts[cursor]),
                    "logit": float(vals_row[i - 1]),
                }
            )
            cursor += 1
        all_rows.append(rows)
    return all_rows


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
    model_dtype = next(model.parameters()).dtype
    token_id = _resolve_bootstrap_token_id(tokenizer)
    input_ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
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
            prefix_hidden_states = prefix_ctx.get("all_token_hidden_by_layer") or []
            if not prefix_hidden_states:
                return {
                    "ok": False,
                    "reason": "prefix_hidden_states_missing",
                    "message": "prefix_hidden_states_missing",
                    "neuron_logits_rows": [],
                    "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
                }
            seq_row_idx = int(layer_idx) + 1
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
                dtype=next(model.parameters()).dtype,
            ).unsqueeze(0)
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
        int((((cfg or {}).get("runtime") or {}).get("ffn_parallel_batch_size", 64))),
    )
    rows: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    current_batch: list[dict[str, Any]] = []

    for start in range(0, int(ffn_dim), int(compute_batch_size)):
        neuron_ids = list(range(start, min(start + int(compute_batch_size), int(ffn_dim))))
        ffn_output_batch = build_ffn_post_silu_neuron_output_matrix(
            model,
            layer_idx=int(layer_idx),
            neuron_indices=neuron_ids,
            activation_value=float(activation_value),
        )
        if prefix_enabled and base_sequence_hidden is not None:
            hidden_batch = base_sequence_hidden.expand(len(neuron_ids), -1, -1).clone()
            hidden_batch[:, -1, :] = hidden_batch[:, -1, :] + ffn_output_batch
        else:
            hidden_batch = ffn_output_batch.to(dtype=model_dtype)
        input_ids_batch = input_ids.expand(len(neuron_ids), -1).contiguous()
        logits_batch, logits_error = run_layer_neuron_batch_to_logits_probe(
            bundle=bundle,
            config=cfg,
            start_layer_idx=layer_idx,
            input_ids=input_ids_batch,
            hidden_batch=hidden_batch,
            top_k=top_k,
            include_cosine=False,
        )
        if logits_batch is None:
            return {
                "ok": False,
                "reason": "probe_starting_from_middle_layer_failed",
                "error": str(logits_error or "unknown"),
                "neuron_logits_rows": rows,
                "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
            }
        bsz = int(len(logits_batch))
        for local_idx in range(bsz):
            neuron_id = int(neuron_ids[local_idx])
            logits_rows = logits_batch[local_idx]
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
    csv_headers = ["neuron_id"]
    for rank in range(1, int(top_k) + 1):
        csv_headers.append(f"rank_{rank}_text")
        csv_headers.append(f"rank_{rank}_logit")
    write_csv(csv_path, history_rows, headers=csv_headers)
    history_rel = str(csv_path.as_posix())

    return {
        "ok": True,
        "study": "layer_ffn_neuron_single_activation_logits",
        "intervention_layer": int(layer_number),
        "activation_value": float(activation_value),
        "use_prefix_context": bool(prefix_enabled),
        "prefix_text": str(prefix_text or ""),
        "prefix_token_count": int(len(prefix_token_ids)),
        "prefix_last_token_attention_by_layer": prefix_attention_by_layer,
        "attention_reused_for_intervention": bool(attention_reused_for_intervention),
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
