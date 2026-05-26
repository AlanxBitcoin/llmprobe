from __future__ import annotations

# Study: Layer Neurons
# - Input: JSON list payload describing one layer and multiple neuron overrides.
# - Validate JSON in backend; if valid, persist to data/cache/layer_neuron_list.json.
# - Build one intervention vector (all listed neurons applied once), continue from middle layer once.
# - Return heatmaps + top logits through existing ui_tasks (server-driven UI rendering).

from pathlib import Path
from typing import Any
import json

import numpy as np
import torch

from ..config import load_config
from ..probes.probe_hidden_state import (
    fetch_sentence_last_token_hidden_state,
    get_or_build_random_token_mean_matrix,
)
from ..probes.probe_layer_neuron import run_layer_neurons_once_to_logits_probe
from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api
from ..utils.layer_neurons_list_file import ensure_layer_neurons_list_file


def _get_or_start_runtime_api(config: dict[str, Any]):
    try:
        return get_runtime_api()
    except RuntimeError:
        return start_llama_api(config)


def _validate_one_entry(item: Any, *, num_layers: int, hidden_dim: int, tag: str) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(item, dict):
        return None, f"invalid_json_field:{tag}_must_be_object"
    list_name = item.get("list_name")
    if not isinstance(list_name, str) or not str(list_name).strip():
        return None, f"invalid_json_field:{tag}.list_name_must_be_nonempty_string"
    n_layer = item.get("nLayer")
    if not isinstance(n_layer, int):
        return None, f"invalid_json_field:{tag}.nLayer_must_be_integer"
    if not (1 <= int(n_layer) <= int(num_layers)):
        return None, f"invalid_json_field:{tag}.nLayer_out_of_range_valid_1_to_{int(num_layers)}"
    neurons = item.get("neurons")
    if not isinstance(neurons, list):
        return None, f"invalid_json_field:{tag}.neurons_must_be_array"
    clean: list[dict[str, Any]] = []
    for idx, n in enumerate(neurons):
        if not isinstance(n, dict):
            return None, f"invalid_json_field:{tag}.neurons[{idx}]_must_be_object"
        n_neuron = n.get("nNeuron")
        value = n.get("value")
        if not isinstance(n_neuron, int):
            return None, f"invalid_json_field:{tag}.neurons[{idx}].nNeuron_must_be_integer"
        if not isinstance(value, (int, float)):
            return None, f"invalid_json_field:{tag}.neurons[{idx}].value_must_be_number"
        if not (0 <= int(n_neuron) < int(hidden_dim)):
            return None, f"invalid_json_field:{tag}.neurons[{idx}].nNeuron_out_of_range_valid_0_to_{int(hidden_dim)-1}"
        clean.append({"nNeuron": int(n_neuron), "value": float(value)})
    if not clean:
        return None, f"invalid_json_field:{tag}.neurons_must_not_be_empty"
    return {"list_name": str(list_name).strip(), "nLayer": int(n_layer), "neurons": clean}, None


def _parse_and_select_list_json(
    raw_json: str,
    *,
    selected_list_name: str,
    num_layers: int,
    hidden_dim: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str], str | None]:
    try:
        payload = json.loads(str(raw_json or "").strip())
    except json.JSONDecodeError as exc:
        return None, None, [], f"invalid_json:{exc.msg}"

    raw_lists: list[Any]
    if isinstance(payload, dict) and isinstance(payload.get("lists"), list):
        raw_lists = list(payload.get("lists") or [])
    elif isinstance(payload, list):
        raw_lists = list(payload)
        payload = {"lists": raw_lists}
    elif isinstance(payload, dict):
        raw_lists = [payload]
        payload = {"lists": raw_lists}
    else:
        return None, None, [], "invalid_json_type:root_must_be_object_or_array"

    clean_lists: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_lists):
        clean, err = _validate_one_entry(item, num_layers=num_layers, hidden_dim=hidden_dim, tag=f"lists[{idx}]")
        if clean is None:
            return None, None, [], err
        clean_lists.append(clean)
    if not clean_lists:
        return None, None, [], "invalid_json_field:lists_must_not_be_empty"

    names = [str(x["list_name"]) for x in clean_lists]
    if len(set(names)) != len(names):
        return None, None, names, "invalid_json_field:list_name_must_be_unique"

    selected = str(selected_list_name or "").strip()
    if selected:
        chosen = next((x for x in clean_lists if str(x["list_name"]) == selected), None)
        if chosen is None:
            return None, None, names, "selected_list_name_not_found"
    elif len(clean_lists) == 1:
        chosen = clean_lists[0]
    else:
        return None, None, names, "selected_list_name_required_when_multiple_lists"

    return {"lists": clean_lists}, chosen, names, None


def _protocol_flags(config: dict[str, Any]) -> tuple[bool, bool]:
    protocol = str((((config or {}).get("hidden_store") or {}).get("protocol")) or "bos1_assistant0")
    if protocol == "bos0_assistant0":
        return False, False
    if protocol == "bos1_assistant1":
        return True, True
    return True, False


def run_study(
    *,
    layer_neuron_list_json: str,
    selected_list_name: str = "",
    use_prefix_context: bool = False,
    prefix_text: str = "",
    config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/custom.yaml",
) -> dict[str, Any]:
    cfg = config or load_config(config_path)
    project_root = Path(__file__).resolve().parents[2]
    list_file = ensure_layer_neurons_list_file(project_root)

    api = _get_or_start_runtime_api(cfg)
    bundle = api.execute_model_call(RuntimeRequest(config=cfg, force_reload=False)).bundle
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    num_layers = int(getattr(getattr(model, "config", None), "num_hidden_layers", 0) or 0)
    hidden_dim = int(getattr(getattr(model, "config", None), "hidden_size", 0) or 0)
    if num_layers <= 0 or hidden_dim <= 0:
        return {
            "ok": False,
            "reason": "model_shape_unavailable",
            "matrix": [],
            "heatmaps": [],
            "top_logits": [],
            "ui_tasks": [
                {"name": "render_heatmap", "value_key": "heatmaps"},
                {"name": "render_logits", "value_key": "top_logits"},
            ],
        }

    parsed_payload, parsed, list_names, err = _parse_and_select_list_json(
        layer_neuron_list_json,
        selected_list_name=str(selected_list_name or ""),
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )
    if parsed is None or parsed_payload is None:
        return {
            "ok": False,
            "reason": str(err or "invalid_json"),
            "list_file": str(list_file),
            "available_list_names": list_names,
            "matrix": [],
            "heatmaps": [],
            "top_logits": [],
            "ui_tasks": [
                {"name": "render_heatmap", "value_key": "heatmaps"},
                {"name": "render_logits", "value_key": "top_logits"},
            ],
        }

    list_file.write_text(json.dumps(parsed_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    layer_number = int(parsed["nLayer"])
    layer_idx = int(layer_number - 1)
    neurons = list(parsed["neurons"])

    prefix_enabled = bool(use_prefix_context)
    prefix_token_count = 0
    input_ids: torch.Tensor
    base_vector: torch.Tensor
    matrix_ref: np.ndarray

    if prefix_enabled:
        text = str(prefix_text or "").strip()
        if not text:
            return {
                "ok": False,
                "reason": "prefix_text_required",
                "list_file": str(list_file),
                "matrix": [],
                "heatmaps": [],
                "top_logits": [],
                "ui_tasks": [
                    {"name": "render_heatmap", "value_key": "heatmaps"},
                    {"name": "render_logits", "value_key": "top_logits"},
                ],
            }
        prefix_ctx = fetch_sentence_last_token_hidden_state(sentence=text, config=cfg)
        if not isinstance(prefix_ctx, dict) or not prefix_ctx.get("ok"):
            return {
                "ok": False,
                "reason": str((prefix_ctx or {}).get("reason") if isinstance(prefix_ctx, dict) else "prefix_failed"),
                "list_file": str(list_file),
                "matrix": [],
                "heatmaps": [],
                "top_logits": [],
                "ui_tasks": [
                    {"name": "render_heatmap", "value_key": "heatmaps"},
                    {"name": "render_logits", "value_key": "top_logits"},
                ],
            }
        matrix_ref = np.asarray(prefix_ctx.get("matrix") or [], dtype=np.float32)
        if matrix_ref.ndim != 2 or matrix_ref.shape[0] < (layer_idx + 2):
            return {
                "ok": False,
                "reason": "prefix_matrix_unavailable_for_layer",
                "list_file": str(list_file),
                "matrix": [],
                "heatmaps": [],
                "top_logits": [],
                "ui_tasks": [
                    {"name": "render_heatmap", "value_key": "heatmaps"},
                    {"name": "render_logits", "value_key": "top_logits"},
                ],
            }
        base_vector = torch.as_tensor(matrix_ref[layer_idx + 1], dtype=model_dtype, device=device).flatten()
        input_ids_list = [int(x) for x in (prefix_ctx.get("input_ids") or [])]
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        prefix_token_count = int(len(prefix_ctx.get("prefix_token_ids") or []))
    else:
        matrix_ref = np.zeros((num_layers + 1, hidden_dim), dtype=np.float32)
        base_vector = torch.zeros((hidden_dim,), dtype=model_dtype, device=device)
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            enc = tokenizer.encode("", add_special_tokens=True)
            if not enc:
                return {
                    "ok": False,
                    "reason": "bootstrap_token_unavailable",
                    "list_file": str(list_file),
                    "matrix": [],
                    "heatmaps": [],
                    "top_logits": [],
                    "ui_tasks": [
                        {"name": "render_heatmap", "value_key": "heatmaps"},
                        {"name": "render_logits", "value_key": "top_logits"},
                    ],
                }
            bos_id = int(enc[0])
        input_ids = torch.tensor([[int(bos_id)]], dtype=torch.long, device=device)

    modified = base_vector.clone()
    for item in neurons:
        modified[int(item["nNeuron"])] = float(item["value"])

    logits_rows, matrix_from_start, logits_error = run_layer_neurons_once_to_logits_probe(
        bundle=bundle,
        config=cfg,
        start_layer_idx=int(layer_idx),
        input_ids=input_ids,
        hidden_state=modified,
        top_k=15,
        include_cosine=False,
    )
    if logits_rows is None:
        return {
            "ok": False,
            "reason": "probe_starting_from_middle_layer_failed",
            "error": str(logits_error or "unknown"),
            "list_file": str(list_file),
            "matrix": [],
            "heatmaps": [],
            "top_logits": [],
            "ui_tasks": [
                {"name": "render_heatmap", "value_key": "heatmaps"},
                {"name": "render_logits", "value_key": "top_logits"},
            ],
        }

    main_matrix = np.asarray(matrix_ref, dtype=np.float32).copy()
    if main_matrix.shape != (num_layers + 1, hidden_dim):
        main_matrix = np.zeros((num_layers + 1, hidden_dim), dtype=np.float32)
    main_matrix[layer_idx + 1, :] = modified.detach().float().cpu().numpy().astype(np.float32, copy=False)
    if matrix_from_start is not None:
        tail = np.asarray(matrix_from_start, dtype=np.float32)
        # tail rows map to full matrix starting at layer_idx+1.
        start_row = int(layer_idx) + 1
        copy_rows = min(int(tail.shape[0]), int(main_matrix.shape[0] - start_row))
        if copy_rows > 0:
            main_matrix[start_row : start_row + copy_rows, :] = tail[:copy_rows, :]

    include_bos, include_assistant = _protocol_flags(cfg)
    random_ref = get_or_build_random_token_mean_matrix(
        config=cfg,
        include_bos=bool(include_bos),
        include_assistant=bool(include_assistant),
        sample_size=1000,
        seed=20260526,
    )
    random_matrix = None
    diff_matrix = None
    random_source = "error"
    random_error = None
    if isinstance(random_ref, dict) and random_ref.get("ok"):
        candidate = np.asarray(random_ref.get("matrix"), dtype=np.float32)
        if candidate.shape == main_matrix.shape:
            random_matrix = candidate
            diff_matrix = main_matrix - random_matrix
            random_source = str(random_ref.get("cache_source") or "unknown")
        else:
            random_error = f"random_ref_shape_mismatch:{list(candidate.shape)} != {list(main_matrix.shape)}"
    else:
        random_error = str((random_ref or {}).get("reason") if isinstance(random_ref, dict) else "unknown")

    heatmaps = [
        {"key": "layer_neurons", "title": "Layer Neurons Heatmap", "matrix": main_matrix.tolist()},
    ]
    if random_matrix is not None:
        heatmaps.append({"key": "random1000_mean", "title": "Random 1000 Tokens Mean Heatmap", "matrix": random_matrix.tolist()})
    if diff_matrix is not None:
        heatmaps.append({"key": "layer_neurons_minus_random1000", "title": "Layer Neurons - Random1000 Mean Heatmap", "matrix": diff_matrix.tolist()})

    return {
        "ok": True,
        "study": "layer_neurons",
        "word": str(parsed.get("list_name") or "layer_neurons"),
        "rows": int(main_matrix.shape[0]),
        "cols": int(main_matrix.shape[1]),
        "matrix": main_matrix.tolist(),
        "heatmaps": heatmaps,
        "top_logits": logits_rows,
        "logits_source": "probe_intervention",
        "logits_error": None,
        "list_file": str(list_file),
        "available_list_names": list_names,
        "selected_list_name": str(parsed.get("list_name") or ""),
        "list_name": str(parsed.get("list_name") or ""),
        "intervention_layer": int(layer_number),
        "applied_neurons": int(len(neurons)),
        "use_prefix_context": bool(prefix_enabled),
        "prefix_text": str(prefix_text or ""),
        "prefix_token_count": int(prefix_token_count),
        "random_ref_source": random_source,
        "random_ref_error": random_error,
        "ui_tasks": [
            {"name": "render_heatmap", "value_key": "heatmaps"},
            {"name": "render_logits", "value_key": "top_logits"},
        ],
    }
