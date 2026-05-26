from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..runtime_api import get_runtime_api, start_llama_api
from ..utils.hooks import starting_from_middle_layer
from ..utils.logits import rank_vector_logits_and_cosine
from ..utils.token_hidden_store import build_protocol_input_ids, parse_token_ids_with_bos_alias


def _get_or_start_runtime_api(config: dict[str, Any]):
    try:
        return get_runtime_api()
    except RuntimeError:
        return start_llama_api(config)


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

        hidden_dim_cfg = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
        if hidden_dim_cfg <= 0:
            lm_head = getattr(model, "lm_head", None)
            hidden_dim_cfg = int(getattr(lm_head, "in_features", 0) or 0)
        if hidden_dim_cfg <= 0:
            return None, "Unable to infer hidden size from model"

        # Fast path A: caller already provides full [B,S,H] start-layer hidden states.
        # This avoids repeated reference forward passes for each batch.
        if int(hidden_state.ndim) == 3:
            replacement = hidden_state.to(device=device, dtype=next(model.parameters()).dtype)
            if int(replacement.shape[-1]) != hidden_dim_cfg:
                return None, f"Hidden size mismatch: expected {hidden_dim_cfg}, got {int(replacement.shape[-1])}"
            batch_size = int(input_tensor.shape[0])
            seq_len = int(input_tensor.shape[1])
            rep_b = int(replacement.shape[0])
            rep_s = int(replacement.shape[1])
            if rep_b == 1 and batch_size > 1:
                replacement = replacement.expand(batch_size, rep_s, hidden_dim_cfg)
                rep_b = batch_size
            if rep_b != batch_size:
                return None, f"Hidden batch mismatch: expected 1 or {batch_size}, got {rep_b}"
            if rep_s != seq_len:
                return None, f"Hidden seq mismatch: expected {seq_len}, got {rep_s}"
            start_hidden_full = replacement
        elif (
            int(input_tensor.shape[0]) == 1
            and int(input_tensor.shape[1]) == 1
            and int(hidden_state.ndim) == 1
        ):
            replacement = hidden_state.to(device=device, dtype=next(model.parameters()).dtype).flatten()
            start_hidden_full = replacement.view(1, 1, -1)
        else:
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
            row_idx = int(start_layer_idx) + 1
            if not (0 <= row_idx < len(ref_hidden_states)):
                return None, f"start_layer_idx out of range for hidden_states: {start_layer_idx}"
            start_hidden_full = ref_hidden_states[row_idx].detach().clone()
            replacement = hidden_state.to(device=start_hidden_full.device, dtype=start_hidden_full.dtype)
            hidden_dim = int(start_hidden_full.shape[-1])
            batch_size = int(start_hidden_full.shape[0])
            if replacement.ndim == 1:
                flat = replacement.flatten()
                if flat.shape[0] != hidden_dim:
                    return None, f"Hidden size mismatch: expected {hidden_dim}, got {flat.shape[0]}"
                start_hidden_full[:, -1, :] = flat.unsqueeze(0)
            elif replacement.ndim == 2:
                if int(replacement.shape[1]) != hidden_dim:
                    return None, f"Hidden size mismatch: expected {hidden_dim}, got {int(replacement.shape[1])}"
                if int(replacement.shape[0]) == 1 and batch_size > 1:
                    replacement = replacement.expand(batch_size, -1)
                if int(replacement.shape[0]) != batch_size:
                    return None, (
                        f"Hidden batch mismatch: expected 1 or {batch_size}, got {int(replacement.shape[0])}"
                    )
                start_hidden_full[:, -1, :] = replacement
            elif replacement.ndim == 3:
                if int(replacement.shape[-1]) != hidden_dim:
                    return None, f"Hidden size mismatch: expected {hidden_dim}, got {int(replacement.shape[-1])}"
                rep_b = int(replacement.shape[0])
                rep_s = int(replacement.shape[1])
                if rep_b == 1 and batch_size > 1:
                    replacement = replacement.expand(batch_size, rep_s, hidden_dim)
                    rep_b = batch_size
                if rep_b != batch_size:
                    return None, f"Hidden batch mismatch: expected 1 or {batch_size}, got {rep_b}"
                copy_s = min(int(start_hidden_full.shape[1]), rep_s)
                start_hidden_full[:, :copy_s, :] = replacement[:, :copy_s, :]
            else:
                return None, f"Unsupported hidden_state ndim={replacement.ndim}, expected 1/2/3"

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
        matrix_row_idx = injection_layer_index + 1
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


def run_layer_neuron_batch_to_logits_probe(
    *,
    bundle,
    config: dict[str, Any],
    start_layer_idx: int,
    input_ids: torch.Tensor,
    hidden_batch: torch.Tensor,
    top_k: int = 15,
    include_cosine: bool = False,
) -> tuple[list[list[dict[str, Any]]] | None, str | None]:
    try:
        transit, transit_error = run_starting_from_middle_layer_probe(
            word=None,
            config=config,
            start_layer_idx=int(start_layer_idx),
            hidden_state=hidden_batch,
            input_ids_override=input_ids,
            bundle_override=bundle,
        )
        if transit is None:
            return None, str(transit_error or "starting_from_middle_layer failed")
        outputs = transit["result"]["outputs"]
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            return None, "missing_hidden_states"
        final_batch = hidden_states[-1].detach()
        tokenizer = bundle.tokenizer
        model = bundle.model
        last_hidden = final_batch[:, -1, :]
        rows = _rank_hidden_batch_logits_and_cosine(
            model=model,
            tokenizer=tokenizer,
            hidden_batch=last_hidden,
            top_k=int(top_k),
            apply_final_norm=False,
            include_cosine=bool(include_cosine),
        )
        return rows, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _rank_hidden_batch_logits_and_cosine(
    *,
    model,
    tokenizer,
    hidden_batch: torch.Tensor,
    top_k: int,
    apply_final_norm: bool,
    include_cosine: bool,
) -> list[list[dict[str, Any]]]:
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return []
    if hidden_batch.ndim != 2:
        raise ValueError(f"hidden_batch must be 2D [B,H], got shape={tuple(hidden_batch.shape)}")

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
        vocab_k = min(int(top_k), int(logits.shape[-1]))
        top_vals, top_ids = torch.topk(logits, k=vocab_k, dim=-1)

    top_ids_cpu = top_ids.detach().cpu().tolist()
    top_vals_cpu = top_vals.detach().cpu().tolist()
    flat_ids = [int(tok_id) for row in top_ids_cpu for tok_id in row]
    if tokenizer is not None and flat_ids:
        flat_tokens = tokenizer.convert_ids_to_tokens(flat_ids)
        flat_texts = [tokenizer.decode([int(tok_id)], clean_up_tokenization_spaces=False) for tok_id in flat_ids]
    else:
        flat_tokens = ["" for _ in flat_ids]
        flat_texts = ["" for _ in flat_ids]

    cosine_map: list[list[float]] | None = None
    if bool(include_cosine) and flat_ids:
        with torch.no_grad():
            hidden_norm = F.normalize(hidden_batch.to(dtype=torch.float32), dim=1)
            target = lm_head.weight[top_ids].to(dtype=torch.float32)
            target_norm = F.normalize(target, dim=2)
            cosine = torch.sum(target_norm * hidden_norm.unsqueeze(1), dim=2)
        cosine_map = cosine.detach().cpu().tolist()

    out: list[list[dict[str, Any]]] = []
    cursor = 0
    batch_size = int(len(top_ids_cpu))
    for b in range(batch_size):
        row_ids = top_ids_cpu[b]
        row_vals = top_vals_cpu[b]
        row: list[dict[str, Any]] = []
        for i, tok_id in enumerate(row_ids, start=1):
            entry: dict[str, Any] = {
                "rank": int(i),
                "token_id": int(tok_id),
                "token": str(flat_tokens[cursor]),
                "text": str(flat_texts[cursor]),
                "logit": float(row_vals[i - 1]),
            }
            if cosine_map is not None:
                entry["cosine_similarity"] = float(cosine_map[b][i - 1])
            row.append(entry)
            cursor += 1
        out.append(row)
    return out
