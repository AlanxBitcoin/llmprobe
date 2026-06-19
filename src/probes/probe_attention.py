from __future__ import annotations

# Probe: reusable per-layer per-head attention metric extraction.
# Keep study-specific token/protocol semantics outside this module.

from typing import Any

import numpy as np
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api


def _get_or_start_runtime_api(config: dict[str, Any]):
    try:
        return get_runtime_api()
    except RuntimeError:
        return start_llama_api(config)


def _error(reason: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "reason": str(reason or "unknown")}
    out.update(extra)
    return out


def _resolve_head_config(self_attn: Any) -> tuple[int, int, int]:
    head_dim = int(getattr(self_attn, "head_dim", 0))
    q_proj = getattr(self_attn, "q_proj", None)
    k_proj = getattr(self_attn, "k_proj", None)
    n_q = int(getattr(self_attn, "num_heads", 0))
    n_kv = int(getattr(self_attn, "num_key_value_heads", 0))

    if head_dim > 0 and q_proj is not None and n_q <= 0:
        q_out = int(getattr(q_proj, "out_features", 0))
        if q_out > 0 and q_out % head_dim == 0:
            n_q = int(q_out // head_dim)
    if head_dim > 0 and k_proj is not None and n_kv <= 0:
        k_out = int(getattr(k_proj, "out_features", 0))
        if k_out > 0 and k_out % head_dim == 0:
            n_kv = int(k_out // head_dim)
    if n_q > 0 and n_kv <= 0:
        n_groups = int(getattr(self_attn, "num_key_value_groups", 0))
        if n_groups > 0 and n_q % n_groups == 0:
            n_kv = int(n_q // n_groups)
    return n_q, n_kv, head_dim


def _build_strict_causal_mask(seq_len: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Keep future keys fully invisible for each query position.
    mask = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)
    upper = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
    mask[:, :, upper] = torch.finfo(dtype).min
    return mask


def fetch_head_attention_metrics_for_input_ids(
    *,
    input_ids: list[int],
    query_positions: dict[str, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generic attention probe on explicit input ids and named query positions.

    Returns per-layer/per-head full-key vectors for both pre-softmax qk and post-softmax attention.
    """
    ids = [int(x) for x in (input_ids or [])]
    if not ids:
        return _error("input_ids_required")
    if not isinstance(query_positions, dict) or not query_positions:
        return _error("query_positions_required")

    try:
        normalized_queries: dict[str, int] = {}
        seq_len = int(len(ids))
        for key, pos in query_positions.items():
            name = str(key or "").strip()
            if not name:
                return _error("query_name_required")
            p = int(pos)
            if p < 0 or p >= seq_len:
                return _error("query_position_out_of_range", query=name, query_pos=int(p), seq_len=seq_len)
            normalized_queries[name] = p

        api = _get_or_start_runtime_api(config)
        bundle = api.execute_model_call(RuntimeRequest(config=config, force_reload=False)).bundle
        model = bundle.model
        tokenizer = bundle.tokenizer

        device = next(model.parameters()).device
        input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        base_model = getattr(model, "model", None)
        if base_model is None:
            return _error("base_model_unavailable")
        model_layers = getattr(base_model, "layers", None)
        embed_tokens = getattr(base_model, "embed_tokens", None)
        rotary_emb = getattr(base_model, "rotary_emb", None)
        if model_layers is None or embed_tokens is None or rotary_emb is None:
            return _error("decoder_layers_unavailable")

        position_ids = torch.arange(input_tensor.shape[1], device=device).unsqueeze(0)
        inputs_embeds = embed_tokens(input_tensor)
        causal_mask = _build_strict_causal_mask(
            int(input_tensor.shape[1]),
            device=device,
            dtype=inputs_embeds.dtype,
        )
        hidden_states = inputs_embeds
        position_embeddings = rotary_emb(hidden_states, position_ids=position_ids)

        n_heads = None
        query_qk_to_keys: dict[str, list[list[list[float]]]] = {k: [] for k in normalized_queries}
        query_attn_to_keys: dict[str, list[list[list[float]]]] = {k: [] for k in normalized_queries}

        for layer_idx, layer in enumerate(model_layers):
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:
                return _error("self_attention_unavailable", layer=int(layer_idx + 1))

            n_q, n_kv, head_dim = _resolve_head_config(self_attn)
            if n_q <= 0 or n_kv <= 0 or head_dim <= 0:
                return _error(
                    "invalid_head_config",
                    layer=int(layer_idx + 1),
                    num_heads=n_q,
                    num_key_value_heads=n_kv,
                    head_dim=head_dim,
                )
            if n_heads is None:
                n_heads = n_q
            if n_q != n_heads:
                return _error("inconsistent_attention_heads")

            with torch.no_grad():
                hs_norm = layer.input_layernorm(hidden_states)
                input_shape = hs_norm.shape[:-1]
                hidden_shape = (*input_shape, -1, head_dim)
                q_states = self_attn.q_proj(hs_norm).view(hidden_shape).transpose(1, 2)
                k_states = self_attn.k_proj(hs_norm).view(hidden_shape).transpose(1, 2)
                cos, sin = position_embeddings
                q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)
                k_states = repeat_kv(k_states, self_attn.num_key_value_groups)
                qk_logits = torch.matmul(q_states, k_states.transpose(2, 3)) * float(self_attn.scaling)
                if causal_mask is not None:
                    qk_logits = qk_logits + causal_mask[:, :, : qk_logits.shape[2], : qk_logits.shape[3]]
                attn_probs = torch.softmax(qk_logits, dim=-1)

                for query_name, query_pos in normalized_queries.items():
                    qk_vec = qk_logits[0, :, int(query_pos), :].detach().float()
                    attn_vec = attn_probs[0, :, int(query_pos), :].detach().float()
                    query_qk_to_keys[query_name].append(qk_vec.cpu().numpy().astype(np.float32, copy=False).tolist())
                    query_attn_to_keys[query_name].append(attn_vec.cpu().numpy().astype(np.float32, copy=False).tolist())

            with torch.no_grad():
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )

        if n_heads is None:
            return _error("attention_empty")

        rows = int(len(model_layers))
        cols = int(n_heads)
        return {
            "ok": True,
            "input_ids": ids,
            "input_tokens": tokenizer.convert_ids_to_tokens(ids),
            "rows": rows,
            "cols": cols,
            "row_labels": [f"layer_{idx + 1}" for idx in range(rows)],
            "col_labels": [f"head_{idx + 1}" for idx in range(cols)],
            "query_positions": normalized_queries,
            "query_qk_to_keys": query_qk_to_keys,
            "query_attn_to_keys": query_attn_to_keys,
            "value_type": "qk_score_and_attention_probability",
        }
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc))
