from __future__ import annotations

"""逐个激活 FFN 神经元并生成 logits 排名表。

功能:
- 指定一个 decoder 层（默认 30）。
- 对该层每个 post-SiLU FFN 神经元做单神经元激活干预。
- 将激活映射回层输出后继续前向推理到末层。
- 统计 LM Head 的 top-k logits，并输出按神经元组织的表格结果。
"""

from pathlib import Path
from typing import Any
from datetime import datetime
import json
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F

from ..config import load_config
from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api
from ..probes.probe_layer_neuron import run_layer_neuron_batch_to_logits_probe
from ..probes.probe_hidden_state import fetch_sentence_last_token_hidden_state
from ..utils.hooks import build_ffn_post_silu_neuron_output_matrix
from ..utils.token_hidden_store import TokenHiddenStore, build_hidden_store_config, protocol_from_flags
from ..utils.utils import ensure_dir, write_csv

_W1_MATRIX_CACHE: dict[tuple[int, int], torch.Tensor] = {}


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


def _progress_file() -> Path:
    return ensure_dir(Path("data") / "outputs" / "layer_ffn_neuron_logits_table" / "progress") / "live_progress.json"


def _write_progress_snapshot(
    *,
    layer_number: int,
    activation_value: float,
    reverse_mode: bool,
    top_k: int,
    ffn_dim: int,
    batch_size: int,
    compute_batch_size: int,
    rows_count: int,
    batches: list[dict[str, Any]],
) -> None:
    payload = {
        "status": "ok",
        "return_code": 0,
        "stdout": "",
        "stderr": "",
        "artifacts": [],
        "csv_preview": None,
        "hidden_state_heatmap": {
            "ok": True,
            "study": "layer_ffn_neuron_single_activation_logits",
            "intervention_layer": int(layer_number),
            "activation_value": float(activation_value),
            "reverse": bool(reverse_mode),
            "top_k": int(top_k),
            "hidden_dim": int(ffn_dim),
            "returned_rows": int(rows_count),
            "filtered_out_rows": 0,
            "return_batch_size": int(batch_size),
            "compute_batch_size": int(compute_batch_size),
            "neuron_kind": "ffn_w1_reverse" if reverse_mode else "ffn_post_silu",
            "neuron_logits_rows": [],
            "neuron_logits_batches": list(batches),
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_batches"}],
        },
    }
    path = _progress_file()
    text = json.dumps(payload, ensure_ascii=False)
    last_err: Exception | None = None
    for _ in range(3):
        tmp = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
            return
        except PermissionError as exc:
            last_err = exc
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            time.sleep(0.03)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            raise
    # Fallback for Windows file-lock race: overwrite in-place when atomic replace keeps failing.
    try:
        path.write_text(text, encoding="utf-8")
        return
    except Exception:
        if last_err is not None:
            raise last_err
        raise


def _clear_progress_snapshot() -> None:
    path = _progress_file()
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


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


def _rank_w1_batch_by_embedding_dot(
    *,
    model,
    tokenizer,
    w1_batch: torch.Tensor,
    top_k: int = 15,
) -> list[list[dict[str, Any]]]:
    emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    emb_weight = getattr(emb, "weight", None) if emb is not None else None
    if emb_weight is None:
        return [[] for _ in range(int(w1_batch.shape[0]))]
    if w1_batch.ndim != 2:
        raise ValueError(f"w1_batch must be [B,H], got shape={tuple(w1_batch.shape)}")

    # Dot-product against token embeddings: logits[b, v] = <w1_b, emb[v]>.
    hidden_for_dot = w1_batch.to(dtype=emb_weight.dtype)
    with torch.no_grad():
        logits = F.linear(hidden_for_dot, emb_weight).to(dtype=torch.float32)
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


def _build_ffn_w1_neuron_input_matrix(
    model,
    *,
    layer_idx: int,
    neuron_indices: list[int],
    activation_value: float,
) -> tuple[torch.Tensor, str]:
    """
    Batched FFN w1(gate_proj) neuron directions.

    Returns:
      Tensor [B, hidden_dim], where each row is gate_proj.weight[neuron_id] * activation_value.
    """
    if not neuron_indices:
        raise ValueError("neuron_indices must not be empty")
    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("Model does not expose decoder layers via `model.layers` or `layers`.")

    n_layers = int(len(layers))
    if not (0 <= int(layer_idx) < n_layers):
        raise ValueError(f"layer_idx out of range: {layer_idx}, num_layers={n_layers}")

    layer = layers[int(layer_idx)]
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise ValueError(f"Layer {layer_idx} does not expose `mlp`.")
    gate_proj = getattr(mlp, "gate_proj", None)
    if gate_proj is None:
        raise ValueError(f"Layer {layer_idx} mlp does not expose `gate_proj`.")

    weight = getattr(gate_proj, "weight", None)
    in_features = int(getattr(gate_proj, "out_features", 0) or 0)
    hidden_size = int(getattr(gate_proj, "in_features", 0) or 0)
    expected_hidden_size = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
    if expected_hidden_size <= 0:
        lm_head = getattr(model, "lm_head", None)
        expected_hidden_size = int(getattr(lm_head, "in_features", 0) or 0)
    if in_features <= 0 or hidden_size <= 0:
        if weight is not None and getattr(weight, "ndim", 0) == 2:
            in_features = int(weight.shape[0])
            hidden_size = int(weight.shape[1])
    if in_features <= 0 or hidden_size <= 0:
        raise ValueError(f"Layer {layer_idx} gate_proj shape unavailable.")
    probe_hidden_size = int(hidden_size)
    if expected_hidden_size > 0 and expected_hidden_size != hidden_size:
        probe_hidden_size = int(expected_hidden_size)

    clean_indices = [int(x) for x in neuron_indices]
    for idx in clean_indices:
        if not (0 <= idx < in_features):
            raise ValueError(f"neuron_idx out of range: {idx}, ffn_dim={in_features}")

    model_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    # Fast path: only trust raw weight for standard dense Linear.
    if isinstance(gate_proj, torch.nn.Linear):
        if (
            weight is not None
            and getattr(weight, "ndim", 0) == 2
            and int(weight.shape[0]) >= in_features
            and int(weight.shape[1]) == int(probe_hidden_size)
        ):
            selected = weight[clean_indices, :].detach().to(device=device, dtype=model_dtype)
            return selected * float(activation_value), "direct_weight"

    # Quantized/custom module fallback: reconstruct full W once per layer via basis probing.
    # For x = e_d, gate_proj(x) gives W[:, d], so we can recover every column.
    cache_key = (id(gate_proj), int(layer_idx))
    full_w_cpu = _W1_MATRIX_CACHE.get(cache_key)
    if full_w_cpu is None:
        probe_chunk = 256
        full_w = torch.empty((in_features, probe_hidden_size), dtype=model_dtype, device=device)
        for start in range(0, probe_hidden_size, probe_chunk):
            end = min(start + probe_chunk, probe_hidden_size)
            width = int(end - start)
            basis = torch.zeros((width, probe_hidden_size), dtype=model_dtype, device=device)
            basis[:, start:end] = torch.eye(width, dtype=model_dtype, device=device)
            with torch.no_grad():
                out = gate_proj(basis)  # [width, ffn_dim]
            if int(out.ndim) != 2 or int(out.shape[1]) < in_features:
                raise ValueError(
                    f"gate_proj basis probe output shape invalid: got {tuple(out.shape)}, need [B,{in_features}]"
                )
            full_w[:, start:end] = out.transpose(0, 1).to(dtype=model_dtype)
        # Keep cache on CPU to reduce persistent GPU memory.
        full_w_cpu = full_w.detach().to(device="cpu", dtype=torch.float32)
        _W1_MATRIX_CACHE[cache_key] = full_w_cpu
        del full_w

    selected = full_w_cpu[clean_indices, :].to(device=device, dtype=model_dtype)
    return selected * float(activation_value), "basis_probe_cached"


def run_study(
    *,
    intervention_layer: int = 30,
    activation_value: float = 10.0,
    include_bos: bool = True,
    reverse: bool = False,
    use_prefix_context: bool = False,
    prefix_text: str = "",
    return_batch_size: int = 1000,
    config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/custom.yaml",
) -> dict[str, Any]:
    _clear_progress_snapshot()
    cfg = config or load_config(config_path)
    hs_cfg = dict((cfg.get("hidden_store") or {}))
    hs_cfg["protocol"] = protocol_from_flags(bos=bool(include_bos), assistant=False)
    study_cfg = dict(cfg)
    study_cfg["hidden_store"] = hs_cfg
    api = _get_or_start_runtime_api(study_cfg)
    bundle = api.execute_model_call(RuntimeRequest(config=study_cfg, force_reload=False)).bundle
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
    reverse_requested = bool(reverse)
    try:
        numeric_layer = float(raw_layer)
    except (TypeError, ValueError):
        valid_layer_min = 0 if reverse_requested else 1
        valid_layer_max = int(num_layers - 1) if reverse_requested else int(num_layers)
        return {
            "ok": False,
            "reason": "invalid_intervention_layer_type",
            "requested_layer": raw_layer,
            "valid_layer_min": int(valid_layer_min),
            "valid_layer_max": int(valid_layer_max),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }
    if not numeric_layer.is_integer():
        valid_layer_min = 0 if reverse_requested else 1
        valid_layer_max = int(num_layers - 1) if reverse_requested else int(num_layers)
        return {
            "ok": False,
            "reason": "intervention_layer_must_be_integer",
            "requested_layer": raw_layer,
            "valid_layer_min": int(valid_layer_min),
            "valid_layer_max": int(valid_layer_max),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }
    requested_layer_number = int(numeric_layer)
    if reverse_requested:
        layer_idx = int(requested_layer_number)
        is_valid = 0 <= layer_idx < num_layers
        valid_layer_min = 0
        valid_layer_max = int(num_layers - 1)
    else:
        layer_number = int(requested_layer_number)
        layer_idx = int(layer_number - 1)
        is_valid = 1 <= layer_number <= num_layers
        valid_layer_min = 1
        valid_layer_max = int(num_layers)
    if not is_valid:
        return {
            "ok": False,
            "reason": "invalid_intervention_layer",
            "requested_layer": int(requested_layer_number),
            "valid_layer_min": int(valid_layer_min),
            "valid_layer_max": int(valid_layer_max),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }

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
    layer_input_sequence_hidden = None
    layer_output_sequence_hidden = None
    hidden_source = "model_forward"
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
            config=study_cfg,
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

    if prefix_enabled:
        with torch.no_grad():
            model_outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = getattr(model_outputs, "hidden_states", None)
        if hidden_states is None:
            hidden_states = model_outputs.get("hidden_states") if isinstance(model_outputs, dict) else None
        if not hidden_states:
            return {
                "ok": False,
                "reason": "hidden_states_missing",
                "neuron_logits_rows": [],
                "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
            }
        if int(layer_idx) + 1 >= len(hidden_states):
            return {
                "ok": False,
                "reason": f"hidden_states_unavailable_for_layer:{layer_idx}",
                "neuron_logits_rows": [],
                "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
            }
        layer_input_sequence_hidden = hidden_states[int(layer_idx)].detach().to(device=device, dtype=model_dtype)
        layer_output_sequence_hidden = hidden_states[int(layer_idx) + 1].detach().to(device=device, dtype=model_dtype)
        hidden_source = "model_forward_prefix"
    else:
        # Store-first path for no-prefix context: this is where include_bos protocol selection applies.
        try:
            store_cfg = build_hidden_store_config(study_cfg, bundle=bundle)
            store = TokenHiddenStore(store_cfg, tokenizer)
            store_layers = np.asarray(store.get_or_compute_layers(bundle, int(token_id)), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "reason": "hidden_store_load_failed",
                "error": str(exc),
                "neuron_logits_rows": [],
                "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
            }
        if int(layer_idx) + 1 >= int(store_layers.shape[0]):
            return {
                "ok": False,
                "reason": f"hidden_store_rows_unavailable_for_layer:{layer_idx}",
                "rows": int(store_layers.shape[0]),
                "neuron_logits_rows": [],
                "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
            }
        layer_input_sequence_hidden = (
            torch.as_tensor(store_layers[int(layer_idx)], device=device, dtype=model_dtype)
            .view(1, 1, -1)
        )
        layer_output_sequence_hidden = (
            torch.as_tensor(store_layers[int(layer_idx) + 1], device=device, dtype=model_dtype)
            .view(1, 1, -1)
        )
        hidden_source = "hidden_store"

    top_k = 15
    batch_size = max(1, int(return_batch_size))
    compute_batch_size = max(
        1,
        int((((cfg or {}).get("runtime") or {}).get("ffn_parallel_batch_size", 64))),
    )
    rows: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    current_batch: list[dict[str, Any]] = []
    reverse_mode = bool(reverse_requested)
    reverse_embedding_dot_mode = bool(reverse_mode and int(layer_idx) == 0 and not bool(include_bos))
    w1_vector_source = "not_used"

    for start in range(0, int(ffn_dim), int(compute_batch_size)):
        neuron_ids = list(range(start, min(start + int(compute_batch_size), int(ffn_dim))))
        if reverse_mode:
            w1_batch, w1_vector_source = _build_ffn_w1_neuron_input_matrix(
                model,
                layer_idx=int(layer_idx),
                neuron_indices=neuron_ids,
                activation_value=float(activation_value),
            )
            if reverse_embedding_dot_mode:
                # Special case requested by product semantics:
                # reverse + layer0 + no BOS -> rank by W1-to-embedding dot products.
                logits_batch = _rank_w1_batch_by_embedding_dot(
                    model=model,
                    tokenizer=tokenizer,
                    w1_batch=w1_batch.to(dtype=model_dtype),
                    top_k=top_k,
                )
            else:
                # Reverse mode default: compare each neuron's W1-side vector with this layer's
                # input hidden (last token), then rank logits from the element-wise product.
                layer_input_last_hidden = layer_input_sequence_hidden[:, -1, :].expand(len(neuron_ids), -1)
                compare_hidden_batch = (w1_batch * layer_input_last_hidden).to(dtype=model_dtype)
                logits_batch = _rank_hidden_batch_by_logits(
                    model=model,
                    tokenizer=tokenizer,
                    hidden_batch=compare_hidden_batch,
                    top_k=top_k,
                    apply_final_norm=False,
                )
            logits_error = None
        else:
            ffn_output_batch = build_ffn_post_silu_neuron_output_matrix(
                model,
                layer_idx=int(layer_idx),
                neuron_indices=neuron_ids,
                activation_value=float(activation_value),
            )
            if prefix_enabled and layer_output_sequence_hidden is not None:
                hidden_batch = layer_output_sequence_hidden.expand(len(neuron_ids), -1, -1).clone()
                hidden_batch[:, -1, :] = hidden_batch[:, -1, :] + ffn_output_batch
            else:
                hidden_batch = ffn_output_batch.to(dtype=model_dtype)
            probe_start_layer_idx = int(layer_idx)
            input_ids_batch = input_ids.expand(len(neuron_ids), -1).contiguous()
            logits_batch, logits_error = run_layer_neuron_batch_to_logits_probe(
                bundle=bundle,
                config=study_cfg,
                start_layer_idx=probe_start_layer_idx,
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
                _write_progress_snapshot(
                    layer_number=int(requested_layer_number),
                    activation_value=float(activation_value),
                    reverse_mode=bool(reverse_mode),
                    top_k=int(top_k),
                    ffn_dim=int(ffn_dim),
                    batch_size=int(batch_size),
                    compute_batch_size=int(compute_batch_size),
                    rows_count=int(len(rows)),
                    batches=batches,
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
        _write_progress_snapshot(
            layer_number=int(requested_layer_number),
            activation_value=float(activation_value),
            reverse_mode=bool(reverse_mode),
            top_k=int(top_k),
            ffn_dim=int(ffn_dim),
            batch_size=int(batch_size),
            compute_batch_size=int(compute_batch_size),
            rows_count=int(len(rows)),
            batches=batches,
        )

    # Persist study result as CSV history for quick reload in UI.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reverse_tag = "_rev1" if reverse_mode else ""
    csv_name = (
        f"ffn_layer{int(requested_layer_number)}"
        f"_act{str(float(activation_value)).replace('.', 'p')}"
        f"{reverse_tag}"
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
    _clear_progress_snapshot()

    return {
        "ok": True,
        "study": "layer_ffn_neuron_single_activation_logits",
        "intervention_layer": int(requested_layer_number),
        "activation_value": float(activation_value),
        "include_bos": bool(include_bos),
        "hidden_source": str(hidden_source),
        "reverse": bool(reverse_mode),
        "reverse_embedding_dot_mode": bool(reverse_embedding_dot_mode),
        "reverse_requested": bool(reverse_requested),
        "reverse_fallback": bool(reverse_requested and not reverse_mode),
        "w1_vector_source": str(w1_vector_source),
        "use_prefix_context": bool(prefix_enabled),
        "prefix_text": str(prefix_text or ""),
        "prefix_token_count": int(len(prefix_token_ids)),
        "prefix_last_token_attention_by_layer": prefix_attention_by_layer,
        "attention_reused_for_intervention": bool(attention_reused_for_intervention),
        "threshold": 0.0,
        "top_k": int(top_k),
        "hidden_dim": int(ffn_dim),
        "neuron_kind": "ffn_w1_reverse" if reverse_mode else "ffn_post_silu",
        "returned_rows": int(len(rows)),
        "filtered_out_rows": 0,
        "return_batch_size": int(batch_size),
        "compute_batch_size": int(compute_batch_size),
        "neuron_logits_rows": [],
        "neuron_logits_batches": batches,
        "history_csv_path": history_rel,
        "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_batches"}],
    }
