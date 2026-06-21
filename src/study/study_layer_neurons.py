from __future__ import annotations

"""多神经元联合干预 study 入口。

功能:
- 接收并校验某一层的多神经元干预配置（JSON）。
- 持久化配置到缓存文件供后续复用。
- 一次性构建联合干预向量并从中间层继续推理。
- 返回热力图与 top logits 的前端渲染任务数据。
"""

import argparse
from pathlib import Path
from typing import Any
import json

import numpy as np
import torch

from ..config import load_config
from ..probes.probe_hidden_state import (
    fetch_sentence_last_token_hidden_state,
    get_or_build_random_token_prefix_attention_baseline_matrix,
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


def _validate_one_entry(
    item: Any,
    *,
    num_layers: int,
    hidden_dim: int,
    tag: str,
    default_list_name: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(item, dict):
        return None, f"invalid_json_field:{tag}_must_be_object"
    list_name = item.get("list_name")
    if list_name is None or str(list_name).strip() == "":
        list_name = str(default_list_name)
    if not isinstance(list_name, str) or not str(list_name).strip():
        return None, f"invalid_json_field:{tag}.list_name_invalid"
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
        n_neuron: Any = None
        value: Any = None
        # Support both formats:
        # 1) object: {"nNeuron": 45, "value": 20.0}
        # 2) compact pair: [45, 20.0]
        if isinstance(n, dict):
            if "nNeuron" in n or "value" in n:
                n_neuron = n.get("nNeuron")
                value = n.get("value")
            elif "neuron" in n or "v" in n or "n" in n:
                n_neuron = n.get("neuron", n.get("n"))
                value = n.get("value", n.get("v"))
            elif len(n) == 1:
                # Support compact object form: {"23": 1.0}
                key, val = next(iter(n.items()))
                n_neuron = key
                value = val
            else:
                return None, f"invalid_json_field:{tag}.neurons[{idx}]_object_missing_neuron_or_value"
        elif isinstance(n, (list, tuple)) and len(n) == 2:
            n_neuron, value = n[0], n[1]
        elif isinstance(n, str) and "," in n:
            # Support compact string form: "23,1.0"
            parts = [p.strip() for p in n.split(",", 1)]
            n_neuron, value = parts[0], parts[1]
        else:
            return None, f"invalid_json_field:{tag}.neurons[{idx}]_must_be_object_or_pair"
        # neuron id must be an integer (not float-ish like 3603.2)
        if isinstance(n_neuron, float) and not float(n_neuron).is_integer():
            return None, f"invalid_json_field:{tag}.neurons[{idx}].nNeuron_must_be_integer"
        try:
            n_neuron = int(n_neuron)
        except (TypeError, ValueError):
            return None, f"invalid_json_field:{tag}.neurons[{idx}].nNeuron_must_be_integer"
        try:
            value = float(value)
        except (TypeError, ValueError):
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
        clean, err = _validate_one_entry(
            item,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            tag=f"lists[{idx}]",
            default_list_name=f"list_{idx + 1}",
        )
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


def _build_no_prefix_context_input_ids(tokenizer, *, include_bos: bool, include_assistant: bool) -> list[int]:
    # Build symbolic context sequence so continuation uses attention over BOS/chat markers.
    if include_bos:
        try:
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=True,
                add_generation_prompt=bool(include_assistant),
            )
            ids = chat.get("input_ids") if isinstance(chat, dict) else chat
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            ids = [int(x) for x in (ids or [])]
            if ids:
                return ids
        except Exception:
            pass
    # Fallback: at least one token to keep shape valid.
    bos_id = tokenizer.bos_token_id
    if bos_id is not None:
        return [int(bos_id)]
    enc = tokenizer.encode("", add_special_tokens=True)
    if enc:
        return [int(enc[0])]
    return []


def _normalize_template_ids(payload: Any) -> list[int]:
    ids = payload.get("input_ids") if isinstance(payload, dict) else payload
    if ids is None:
        return []
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list):
        return []
    return [int(x) for x in ids]


def _build_assistant_suffix_ids(tokenizer) -> list[int]:
    """Build assistant generation-prompt suffix ids functionally via chat template."""
    try:
        with_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=True,
            add_generation_prompt=True,
        )
        without_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=True,
            add_generation_prompt=False,
        )
        with_ids = _normalize_template_ids(with_prompt)
        without_ids = _normalize_template_ids(without_prompt)
        if len(with_ids) > len(without_ids):
            suffix = with_ids[len(without_ids) :]
            return [int(x) for x in suffix]
    except Exception:
        pass

    # Fallback: explicit assistant marker symbols as real tokens (not literal display prefix).
    fallback_text = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ids = tokenizer(fallback_text, add_special_tokens=False).get("input_ids") or []
    return [int(x) for x in ids]


def _generate_with_one_time_layer_neurons_intervention(
    *,
    bundle,
    input_ids: torch.Tensor,
    layer_idx: int,
    neurons: list[dict[str, Any]],
    max_new_tokens: int = 256,
) -> tuple[str | None, str | None]:
    model = bundle.model
    tokenizer = bundle.tokenizer
    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        return None, "model_layers_not_found"
    if not (0 <= int(layer_idx) < int(len(layers))):
        return None, "layer_idx_out_of_range"

    device = next(model.parameters()).device
    target_layer = layers[int(layer_idx)]
    base_prompt_len = int(input_ids.shape[1])
    intervention_pos = max(0, base_prompt_len - 1)
    run_input_ids = input_ids
    assistant_suffix = _build_assistant_suffix_ids(tokenizer)
    if assistant_suffix:
        existing = [int(x) for x in input_ids[0].detach().cpu().tolist()]
        if len(existing) < len(assistant_suffix) or existing[-len(assistant_suffix) :] != assistant_suffix:
            appended = existing + [int(x) for x in assistant_suffix]
            run_input_ids = torch.tensor([appended], dtype=torch.long, device=device)

    state = {"applied": False}

    def _one_time_prefill_hook(_module, _inputs, output):
        if state["applied"]:
            return output
        if isinstance(output, torch.Tensor):
            if int(output.shape[1]) >= base_prompt_len and int(output.shape[1]) > intervention_pos:
                for item in neurons:
                    output[:, intervention_pos, int(item["nNeuron"])] = float(item["value"])
                state["applied"] = True
            return output
        if isinstance(output, tuple) and output:
            first = output[0]
            if isinstance(first, torch.Tensor) and int(first.shape[1]) >= base_prompt_len and int(first.shape[1]) > intervention_pos:
                for item in neurons:
                    first[:, intervention_pos, int(item["nNeuron"])] = float(item["value"])
                state["applied"] = True
                return (first, *output[1:])
        return output

    hook_handle = target_layer.register_forward_hook(_one_time_prefill_hook)
    try:
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
        with torch.inference_mode():
            out_ids = model.generate(
                input_ids=run_input_ids.to(device=device, dtype=torch.long),
                max_new_tokens=int(max(1, min(int(max_new_tokens), 256))),
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        prompt_len = int(run_input_ids.shape[1])
        new_ids = out_ids[0, prompt_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        if not text:
            text = tokenizer.decode(new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
        return str(text), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    finally:
        hook_handle.remove()


def run_study(
    *,
    layer_neuron_list_json: str,
    selected_list_name: str = "",
    use_prefix_context: bool = False,
    prefix_text: str = "",
    use_random1000_baseline_no_prefix: bool = True,
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
    include_bos, include_assistant = _protocol_flags(cfg)
    random_ref = get_or_build_random_token_mean_matrix(
        config=cfg,
        include_bos=bool(include_bos),
        include_assistant=bool(include_assistant),
        sample_size=1000,
        seed=20260526,
    )
    random_matrix = None
    random_source = "error"
    random_error = None
    if isinstance(random_ref, dict) and random_ref.get("ok"):
        candidate = np.asarray(random_ref.get("matrix"), dtype=np.float32)
        if candidate.shape == (num_layers + 1, hidden_dim):
            random_matrix = candidate
            random_source = str(random_ref.get("cache_source") or "unknown")
        else:
            random_error = f"random_ref_shape_mismatch:{list(candidate.shape)} != {[num_layers + 1, hidden_dim]}"
    else:
        random_error = str((random_ref or {}).get("reason") if isinstance(random_ref, dict) else "unknown")

    prefix_attn_ref = get_or_build_random_token_prefix_attention_baseline_matrix(
        config=cfg,
        include_bos=bool(include_bos),
        include_assistant=bool(include_assistant),
        sample_size=1000,
        seed=20260526,
    )
    prefix_attn_matrix = None
    prefix_attn_source = "error"
    prefix_attn_error = None
    if isinstance(prefix_attn_ref, dict) and prefix_attn_ref.get("ok"):
        candidate = np.asarray(prefix_attn_ref.get("matrix"), dtype=np.float32)
        if candidate.shape == (num_layers + 1, hidden_dim):
            prefix_attn_matrix = candidate
            prefix_attn_source = str(prefix_attn_ref.get("cache_source") or "unknown")
        else:
            prefix_attn_error = f"prefix_attn_shape_mismatch:{list(candidate.shape)} != {[num_layers + 1, hidden_dim]}"
    else:
        prefix_attn_error = str((prefix_attn_ref or {}).get("reason") if isinstance(prefix_attn_ref, dict) else "unknown")

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
        use_random1000_baseline_flag = bool(use_random1000_baseline_no_prefix)
        if use_random1000_baseline_flag:
            if prefix_attn_matrix is None:
                return {
                    "ok": False,
                    "reason": "prefix_attn_ref_unavailable_for_no_prefix",
                    "error": str(prefix_attn_error or "unknown"),
                    "list_file": str(list_file),
                    "matrix": [],
                    "heatmaps": [],
                    "top_logits": [],
                    "ui_tasks": [
                        {"name": "render_heatmap", "value_key": "heatmaps"},
                        {"name": "render_logits", "value_key": "top_logits"},
                    ],
                }
            # No-prefix mode baseline (optional):
            # start from 1000-token averaged prefix-symbol attention baseline.
            matrix_ref = np.asarray(prefix_attn_matrix, dtype=np.float32).copy()
        else:
            # No-prefix mode without 1000-token baseline:
            # start from pure zeros, still keep context token attention in continuation.
            matrix_ref = np.zeros((num_layers + 1, hidden_dim), dtype=np.float32)
        base_vector = torch.as_tensor(matrix_ref[layer_idx + 1], dtype=model_dtype, device=device).flatten()
        context_ids = _build_no_prefix_context_input_ids(
            tokenizer,
            include_bos=bool(include_bos),
            include_assistant=bool(include_assistant),
        )
        if not context_ids:
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
        input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)

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

    generated_text, generated_text_error = _generate_with_one_time_layer_neurons_intervention(
        bundle=bundle,
        input_ids=input_ids,
        layer_idx=int(layer_idx),
        neurons=neurons,
        max_new_tokens=256,
    )

    original_matrix = np.asarray(matrix_ref, dtype=np.float32).copy()
    if original_matrix.shape != (num_layers + 1, hidden_dim):
        original_matrix = np.zeros((num_layers + 1, hidden_dim), dtype=np.float32)
    main_matrix = np.asarray(original_matrix, dtype=np.float32).copy()
    main_matrix[layer_idx + 1, :] = modified.detach().float().cpu().numpy().astype(np.float32, copy=False)
    if matrix_from_start is not None:
        tail = np.asarray(matrix_from_start, dtype=np.float32)
        # tail rows map to full matrix starting at layer_idx+1.
        start_row = int(layer_idx) + 1
        copy_rows = min(int(tail.shape[0]), int(main_matrix.shape[0] - start_row))
        if copy_rows > 0:
            main_matrix[start_row : start_row + copy_rows, :] = tail[:copy_rows, :]

    diff_matrix = None
    if random_matrix is not None:
        if random_matrix.shape == main_matrix.shape:
            diff_matrix = main_matrix - random_matrix
        else:
            random_error = f"random_ref_shape_mismatch:{list(random_matrix.shape)} != {list(main_matrix.shape)}"

    heatmaps = [
        {"key": "original_hidden_state", "title": "Original Hidden State Heatmap", "matrix": original_matrix.tolist()},
        {"key": "layer_neurons", "title": "Layer Neurons Heatmap", "matrix": main_matrix.tolist()},
    ]
    if random_matrix is not None:
        heatmaps.append({"key": "random1000_mean", "title": "Random 1000 Tokens Mean Heatmap", "matrix": random_matrix.tolist()})
    if prefix_attn_matrix is not None:
        heatmaps.append(
            {
                "key": "random1000_prefix_attn_baseline",
                "title": "Random 1000 Prefix-Attention Baseline Heatmap",
                "matrix": prefix_attn_matrix.tolist(),
            }
        )
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
        "use_random1000_baseline_no_prefix": bool(use_random1000_baseline_no_prefix),
        "prefix_text": str(prefix_text or ""),
        "prefix_token_count": int(prefix_token_count),
        "random_ref_source": random_source,
        "random_ref_error": random_error,
        "prefix_attn_ref_source": prefix_attn_source,
        "prefix_attn_ref_error": prefix_attn_error,
        "generated_text": str(generated_text or ""),
        "generated_text_error": generated_text_error,
        "generated_max_new_tokens": 256,
        "ui_tasks": [
            {"name": "render_heatmap", "value_key": "heatmaps"},
            {"name": "render_logits", "value_key": "top_logits"},
            {"name": "render_text_output", "value_key": "generated_text"},
        ],
    }


def register_cli(subparsers: argparse._SubParsersAction, bool_parser) -> None:
    parser = subparsers.add_parser(
        "run-layer-neurons",
        help="Apply multiple neuron overrides from JSON once and return heatmap + top-15 logits.",
    )
    parser.add_argument(
        "--use-prefix-context",
        type=bool_parser,
        default=False,
        help="If true, run prefix text first and apply multi-neuron override on that layer hidden state.",
    )
    parser.add_argument(
        "--prefix-text",
        type=str,
        default="The apple is red.",
        help="Prefix sentence used when --use-prefix-context=true.",
    )
    parser.add_argument(
        "--use-random1000-baseline-no-prefix",
        type=bool_parser,
        default=True,
        help="When --use-prefix-context=false, use 1000-token no-prefix baseline as starting hidden state.",
    )
    parser.add_argument(
        "--selected-list-name",
        type=str,
        default="",
        help="Select one list_name when JSON contains multiple lists.",
    )
    parser.add_argument(
        "--layer-neuron-list-json",
        type=str,
        default="",
        help="JSON payload (compact preferred): {list_name,nLayer,neurons:[[nNeuron,value],...]}",
    )


def try_execute_cli(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any] | None:
    if args.command != "run-layer-neurons":
        return None
    heatmap = run_study(
        layer_neuron_list_json=str(args.layer_neuron_list_json or ""),
        selected_list_name=str(args.selected_list_name or ""),
        use_prefix_context=bool(args.use_prefix_context),
        prefix_text=str(args.prefix_text or ""),
        use_random1000_baseline_no_prefix=bool(args.use_random1000_baseline_no_prefix),
        config=config,
        config_path=args.config,
    )
    return {"hidden_state_heatmap": heatmap}
