from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import load_config
from ..probes.probe_attention import fetch_head_attention_metrics_for_input_ids
from ..utils.token_hidden_store import build_protocol_input_ids, protocol_from_flags, resolve_assistant_token_id


def _error_payload(reason: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": False,
        "reason": str(reason or "unknown"),
        "matrix": [],
        "heatmaps": [],
        "ui_tasks": [{"name": "render_heatmap", "value_key": "heatmaps"}],
    }
    out.update(extra)
    return out


def _locate_query_pos_from_sequences(single_ids: list[int], pair_ids: list[int]) -> int:
    if len(pair_ids) != len(single_ids) + 1:
        return int(len(pair_ids) - 1)
    for idx in range(len(pair_ids)):
        candidate = pair_ids[:idx] + pair_ids[idx + 1 :]
        if candidate == single_ids:
            return int(idx)
    return int(len(pair_ids) - 1)


def _locate_assistant_marker_position(tokenizer, input_ids: list[int]) -> int | None:
    if not input_ids:
        return None
    start_ids = tokenizer("<|start_header_id|>", add_special_tokens=False).get("input_ids") or []
    end_ids = tokenizer("<|end_header_id|>", add_special_tokens=False).get("input_ids") or []
    assistant_id = resolve_assistant_token_id(tokenizer)
    if len(start_ids) == 1 and len(end_ids) == 1:
        start_id = int(start_ids[0])
        end_id = int(end_ids[0])
        for idx in range(1, len(input_ids) - 1):
            if int(input_ids[idx]) != int(assistant_id):
                continue
            if int(input_ids[idx - 1]) == start_id and int(input_ids[idx + 1]) == end_id:
                return int(idx)
    for idx in range(len(input_ids) - 1, -1, -1):
        if int(input_ids[idx]) == int(assistant_id):
            return int(idx)
    return None


def _extract_key_column_matrix(rows_3d: list[list[list[float]]], key_pos: int) -> list[list[float]]:
    key_matrix: list[list[float]] = []
    for layer_heads in rows_3d:
        layer_row: list[float] = []
        if isinstance(layer_heads, list):
            for head_values in layer_heads:
                if isinstance(head_values, list) and 0 <= key_pos < len(head_values):
                    layer_row.append(float(head_values[key_pos]))
                else:
                    layer_row.append(0.0)
        key_matrix.append(layer_row)
    return key_matrix


def run_study(
    *,
    token_a: str,
    token_b: str,
    include_assistant: bool = False,
    config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/custom.yaml",
) -> dict[str, Any]:
    cfg = config or load_config(config_path)
    a = str(token_a or "").strip()
    b = str(token_b or "").strip()
    if not a or not b:
        return _error_payload("token_inputs_required", token_a=a, token_b=b)

    try:
        from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api

        try:
            api = get_runtime_api()
        except RuntimeError:
            api = start_llama_api(cfg)
        bundle = api.execute_model_call(RuntimeRequest(config=cfg, force_reload=False)).bundle
        tokenizer = bundle.tokenizer
    except Exception as exc:  # noqa: BLE001
        return _error_payload(str(exc), token_a=a, token_b=b)

    ids_a = [int(x) for x in (tokenizer(a, add_special_tokens=False).get("input_ids") or [])]
    ids_b = [int(x) for x in (tokenizer(b, add_special_tokens=False).get("input_ids") or [])]
    if len(ids_a) != 1:
        return _error_payload(
            "token_a_not_single_token",
            token_a=a,
            token_b=b,
            token_count=int(len(ids_a)),
            token_count_a=int(len(ids_a)),
            token_ids_a=ids_a,
        )
    if len(ids_b) != 1:
        return _error_payload(
            "token_b_not_single_token",
            token_a=a,
            token_b=b,
            token_count=int(len(ids_b)),
            token_count_b=int(len(ids_b)),
            token_ids_b=ids_b,
        )

    protocol = protocol_from_flags(bos=True, assistant=bool(include_assistant))
    input_ids = [int(x) for x in build_protocol_input_ids(tokenizer, protocol, [int(ids_a[0]), int(ids_b[0])])]
    single_input_ids = [int(x) for x in build_protocol_input_ids(tokenizer, protocol, [int(ids_a[0])])]
    if len(input_ids) < 2:
        return _error_payload("protocol_input_too_short", token_a=a, token_b=b, protocol=protocol, input_ids=input_ids)

    query_pos = _locate_query_pos_from_sequences(single_input_ids, input_ids)
    assistant_query_pos = _locate_assistant_marker_position(tokenizer, input_ids) if bool(include_assistant) else None
    query_positions = {"token_b": int(query_pos)}
    if assistant_query_pos is not None:
        query_positions["assistant"] = int(assistant_query_pos)

    metrics = fetch_head_attention_metrics_for_input_ids(
        input_ids=input_ids,
        query_positions=query_positions,
        config=cfg,
    )
    if not isinstance(metrics, dict) or not metrics.get("ok"):
        reason = str((metrics or {}).get("reason") or "unknown")
        extra = {}
        if isinstance(metrics, dict):
            extra = {k: v for k, v in metrics.items() if k != "reason"}
        return _error_payload(reason, **extra)

    input_tokens = [str(t) for t in (metrics.get("input_tokens") or [])]
    query_pos = int(query_positions.get("token_b", max(len(input_tokens) - 1, 0)))
    prev_pos = max(0, int(query_pos - 1))
    self_pos = int(query_pos)

    query_qk_to_keys = metrics.get("query_qk_to_keys") or {}
    query_attn_to_keys = metrics.get("query_attn_to_keys") or {}
    qk_token_b_to_keys = query_qk_to_keys.get("token_b") or []
    attn_token_b_to_keys = query_attn_to_keys.get("token_b") or []
    attn_assistant_query_to_keys = query_attn_to_keys.get("assistant") or []

    qk_prev = _extract_key_column_matrix(qk_token_b_to_keys, prev_pos)
    qk_self = _extract_key_column_matrix(qk_token_b_to_keys, self_pos)
    attn_prev = _extract_key_column_matrix(attn_token_b_to_keys, prev_pos)
    attn_self = _extract_key_column_matrix(attn_token_b_to_keys, self_pos)
    attn_query_to_keys = attn_token_b_to_keys
    heatmaps = [
        {
            "key": "qk_score_to_token_a",
            "title": f"QK Score (pre-softmax): {b} -> previous({a})",
            "matrix": qk_prev,
            "hover_x_label": "head",
            "hover_y_label": "layer",
            "hover_x_offset": 1,
            "hover_y_offset": 1,
        },
        {
            "key": "qk_score_to_self",
            "title": f"QK Score (pre-softmax): {b} -> self({b})",
            "matrix": qk_self,
            "hover_x_label": "head",
            "hover_y_label": "layer",
            "hover_x_offset": 1,
            "hover_y_offset": 1,
        },
    ]

    # Render attention-probability heatmaps for all keys before query (includes BOS-related tokens).
    if isinstance(attn_query_to_keys, list) and attn_query_to_keys:
        key_count = 0
        first_layer = attn_query_to_keys[0] if attn_query_to_keys else []
        if isinstance(first_layer, list) and first_layer:
            first_head = first_layer[0]
            if isinstance(first_head, list):
                key_count = int(len(first_head))
        if key_count <= 0:
            key_count = max(len(input_tokens), query_pos + 1)
        max_key_pos = min(max(key_count - 1, 0), int(query_pos))
        for key_pos in range(max_key_pos + 1):
            key_token = input_tokens[key_pos] if key_pos < len(input_tokens) else f"pos_{key_pos}"
            key_matrix: list[list[float]] = []
            key_matrix = _extract_key_column_matrix(attn_query_to_keys, key_pos)
            heatmaps.append(
                {
                    "key": f"token_b_to_key_{key_pos}",
                    "title": f"Attention Probability: {b} -> key[{key_pos}]({key_token})",
                    "matrix": key_matrix,
                    "hover_x_label": "head",
                    "hover_y_label": "layer",
                    "hover_x_offset": 1,
                    "hover_y_offset": 1,
                }
            )
    else:
        heatmaps.extend(
            [
                {
                    "key": "token_b_to_token_a",
                    "title": f"Attention Probability: {b} -> previous({a})",
                    "matrix": attn_prev,
                    "hover_x_label": "head",
                    "hover_y_label": "layer",
                    "hover_x_offset": 1,
                    "hover_y_offset": 1,
                },
                {
                    "key": "token_b_to_self",
                    "title": f"Attention Probability: {b} -> self({b})",
                    "matrix": attn_self,
                    "hover_x_label": "head",
                    "hover_y_label": "layer",
                    "hover_x_offset": 1,
                    "hover_y_offset": 1,
                },
            ]
        )

    if bool(include_assistant) and isinstance(attn_assistant_query_to_keys, list) and attn_assistant_query_to_keys:
        assistant_pos = int(assistant_query_pos) if assistant_query_pos is not None else -1
        assistant_label = (
            input_tokens[assistant_pos]
            if 0 <= assistant_pos < len(input_tokens)
            else "assistant_symbol"
        )
        key_count = 0
        first_layer = attn_assistant_query_to_keys[0] if attn_assistant_query_to_keys else []
        if isinstance(first_layer, list) and first_layer:
            first_head = first_layer[0]
            if isinstance(first_head, list):
                key_count = int(len(first_head))
        if key_count <= 0:
            key_count = max(len(input_tokens), assistant_pos + 1)
        max_key_pos = min(max(key_count - 1, 0), int(assistant_pos))
        for key_pos in range(max_key_pos + 1):
            key_token = input_tokens[key_pos] if key_pos < len(input_tokens) else f"pos_{key_pos}"
            key_matrix: list[list[float]] = []
            key_matrix = _extract_key_column_matrix(attn_assistant_query_to_keys, key_pos)
            heatmaps.append(
                {
                    "key": f"assistant_to_key_{key_pos}",
                    "title": f"Attention Probability: assistant[{assistant_pos}]({assistant_label}) -> key[{key_pos}]({key_token})",
                    "matrix": key_matrix,
                    "hover_x_label": "head",
                    "hover_y_label": "layer",
                    "hover_x_offset": 1,
                    "hover_y_offset": 1,
                }
            )
    return {
        "ok": True,
        "study": "one_on_one_attention",
        "token_a": a,
        "token_b": b,
        "include_bos": True,
        "include_assistant": bool(include_assistant),
        "protocol": protocol,
        "value_type": str(metrics.get("value_type") or "qk_score_and_attention_probability"),
        "input_ids": input_ids,
        "input_tokens": metrics.get("input_tokens") or [],
        "rows": int(metrics.get("rows") or 0),
        "cols": int(metrics.get("cols") or 0),
        "row_labels": metrics.get("row_labels") or [],
        "col_labels": metrics.get("col_labels") or [],
        "matrix": qk_prev,
        "heatmaps": heatmaps,
        "ui_tasks": [{"name": "render_heatmap", "value_key": "heatmaps"}],
    }
