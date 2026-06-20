from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import load_config
from ..probes.probe_layer_neuron import run_layer_neuron_batch_to_logits_probe
from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api
from ..utils.token_hidden_store import (
    TokenHiddenStore,
    build_hidden_store_config,
    build_protocol_input_ids,
    parse_token_ids_with_bos_alias,
    protocol_from_flags,
)


def _get_or_start_runtime_api(config: dict[str, Any]):
    try:
        return get_runtime_api()
    except RuntimeError:
        return start_llama_api(config)


def run_study(
    *,
    word: str,
    jump_to_layer: int = 32,
    include_bos: bool = True,
    config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/custom.yaml",
    return_batch_size: int = 1000,
) -> dict[str, Any]:
    cfg = config or load_config(config_path)
    hs_cfg = dict((cfg.get("hidden_store") or {}))
    hs_cfg["protocol"] = protocol_from_flags(bos=bool(include_bos), assistant=False)
    study_cfg = dict(cfg)
    study_cfg["hidden_store"] = hs_cfg

    api = _get_or_start_runtime_api(study_cfg)
    bundle = api.execute_model_call(RuntimeRequest(config=study_cfg, force_reload=False)).bundle
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    raw_word = str(word or "").strip()
    if not raw_word:
        return {
            "ok": False,
            "reason": "word_required",
            "message": "word_required",
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }

    token_ids = [int(x) for x in parse_token_ids_with_bos_alias(tokenizer, raw_word)]
    if len(token_ids) != 1:
        return {
            "ok": False,
            "reason": "single_token_required",
            "token_count": int(len(token_ids)),
            "token_ids": token_ids,
            "word": raw_word,
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }
    token_id = int(token_ids[0])

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
    if num_layers < 2:
        return {
            "ok": False,
            "reason": "model_requires_at_least_two_layers",
            "num_layers": int(num_layers),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }

    # jump_to_layer is 1-based decoder layer number where we inject as that layer's input.
    # This means start_layer_idx = jump_to_layer - 2.
    try:
        jump_layer_number = int(jump_to_layer)
    except (TypeError, ValueError):
        return {
            "ok": False,
            "reason": "invalid_jump_to_layer_type",
            "jump_to_layer": jump_to_layer,
            "valid_layer_min": 1,
            "valid_layer_max": int(num_layers),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }
    if not (1 <= int(jump_layer_number) <= int(num_layers)):
        return {
            "ok": False,
            "reason": "invalid_jump_to_layer",
            "jump_to_layer": int(jump_layer_number),
            "valid_layer_min": 1,
            "valid_layer_max": int(num_layers),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }
    start_layer_idx = int(jump_layer_number - 2)
    protocol = str(hs_cfg.get("protocol") or "bos1_assistant0")
    input_ids = [int(x) for x in build_protocol_input_ids(tokenizer, protocol, [token_id])]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    try:
        store_cfg = build_hidden_store_config(study_cfg, bundle=bundle)
        store = TokenHiddenStore(store_cfg, tokenizer)
        all_layers = np.asarray(store.get_or_compute_layers(bundle, int(token_id)), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "reason": "hidden_store_load_failed",
            "error": str(exc),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }

    max_source_layer = min(30, int(num_layers - 1))
    if int(all_layers.shape[0]) <= max_source_layer:
        max_source_layer = int(all_layers.shape[0] - 1)
    if max_source_layer < 1:
        return {
            "ok": False,
            "reason": "insufficient_hidden_rows_for_shortcut",
            "hidden_rows": int(all_layers.shape[0]),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }

    source_layers = list(range(1, int(max_source_layer) + 1))
    hidden_batch = torch.as_tensor(
        all_layers[source_layers, :],
        device=device,
        dtype=model_dtype,
    )
    input_ids_batch = input_tensor.expand(len(source_layers), -1).contiguous()

    logits_rows_batch, logits_error = run_layer_neuron_batch_to_logits_probe(
        bundle=bundle,
        config=study_cfg,
        start_layer_idx=int(start_layer_idx),
        input_ids=input_ids_batch,
        hidden_batch=hidden_batch,
        top_k=15,
        include_cosine=False,
    )
    if logits_rows_batch is None:
        return {
            "ok": False,
            "reason": "probe_starting_from_middle_layer_failed",
            "error": str(logits_error or "unknown"),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }

    rows: list[dict[str, Any]] = []
    for idx, source_layer in enumerate(source_layers):
        rows.append(
            {
                "source_layer": int(source_layer),
                "neuron_id": int(source_layer),
                "top_logits": list(logits_rows_batch[idx] or []),
            }
        )

    batch_size = max(1, int(return_batch_size))
    batches: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        if not chunk:
            continue
        batches.append(
            {
                "batch_index": int(len(batches)),
                "start_neuron_id": int(chunk[0]["source_layer"]),
                "end_neuron_id": int(chunk[-1]["source_layer"]),
                "rows": chunk,
            }
        )

    return {
        "ok": True,
        "study": "layer_shortcut",
        "word": raw_word,
        "include_bos": bool(include_bos),
        "protocol": protocol,
        "token_id": int(token_id),
        "jump_to_layer": int(jump_layer_number),
        "shortcut_target_layer": int(jump_layer_number),
        "shortcut_target_layer_input": int(max(0, jump_layer_number - 1)),
        "remaining_layers_run": int(max(0, num_layers - jump_layer_number + 1)),
        "source_layer_min": 1,
        "source_layer_max": int(max_source_layer),
        "top_k": 15,
        "row_label_key": "source_layer",
        "row_label_title": "source_layer",
        "table_title": "Layer Shortcut -> Top 15 Logits Table",
        "returned_rows": int(len(rows)),
        "return_batch_size": int(batch_size),
        "neuron_logits_rows": rows,
        "neuron_logits_batches": batches,
        "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_batches"}],
    }
