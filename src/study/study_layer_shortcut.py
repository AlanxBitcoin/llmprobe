from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import load_config
from ..probes.probe_layer_neuron import run_layer_neuron_batch_to_logits_probe
from ..probes.probe_layer_shortcut import validate_jump_to_layer_zero_based
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
    jump_to_layer: int = 31,
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

    try:
        jump_layer_number, start_layer_idx = validate_jump_to_layer_zero_based(
            jump_to_layer=int(jump_to_layer),
            layer_count=int(num_layers),
        )
    except (TypeError, ValueError):
        return {
            "ok": False,
            "reason": "invalid_jump_to_layer",
            "jump_to_layer": int(jump_to_layer) if str(jump_to_layer).strip().lstrip("-").isdigit() else jump_to_layer,
            "valid_layer_min": 0,
            "valid_layer_max": int(num_layers - 1),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }
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

    max_source_layer = min(30, int(num_layers - 2), int(jump_layer_number - 1))
    if int(all_layers.shape[0]) <= int(max_source_layer + 1):
        max_source_layer = int(all_layers.shape[0] - 2)
    if max_source_layer < 0:
        return {
            "ok": False,
            "reason": "insufficient_hidden_rows_for_shortcut",
            "hidden_rows": int(all_layers.shape[0]),
            "neuron_logits_rows": [],
            "ui_tasks": [{"name": "render_neuron_logits_table", "value_key": "neuron_logits_rows"}],
        }

    source_layers = list(range(0, int(max_source_layer) + 1))
    source_rows = [int(x + 1) for x in source_layers]
    hidden_batch = torch.as_tensor(
        all_layers[source_rows, :],
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
        "shortcut_target_layer_input": int(max(0, jump_layer_number)),
        "remaining_layers_run": int(max(0, num_layers - jump_layer_number)),
        "source_layer_min": 0,
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


def register_cli(subparsers: argparse._SubParsersAction, bool_parser) -> None:
    parser = subparsers.add_parser(
        "run-layer-shortcut",
        help="Single-token layer shortcut: layer 0..30 hidden -> configured target layer input, then rank top-15 logits.",
    )
    parser.add_argument("word", help="Single-token word")
    parser.add_argument(
        "--include-bos",
        type=bool_parser,
        default=True,
        help="Whether to prepend BOS/chat-prefix symbols (true/false). false means only this word token is used.",
    )
    parser.add_argument(
        "--jump-to-layer",
        type=int,
        default=31,
        help="0-based decoder layer number to inject hidden state into (default: 31).",
    )


def try_execute_cli(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any] | None:
    if args.command != "run-layer-shortcut":
        return None
    heatmap = run_study(
        word=str(args.word or ""),
        jump_to_layer=int(args.jump_to_layer),
        include_bos=bool(args.include_bos),
        config=config,
        config_path=args.config,
    )
    return {"hidden_state_heatmap": heatmap}
