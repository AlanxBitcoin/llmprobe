from __future__ import annotations

# Study: Attribute Group Neurons
# - Input: JSON text with named groups and token lists.
# - Select one group, read each token's hidden-state matrix through token hidden-store (cache-first).
# - Apply configurable filtering over neurons (algorithm placeholder, to be refined later).
# - Export selected-neuron statistics + per-token values to CSV and return popup-friendly payload.

from pathlib import Path
from typing import Any
import csv
import json
import re
from datetime import datetime

import numpy as np

from ..config import load_config
from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api
from ..probes.probe_hidden_state import get_or_build_random_token_mean_matrix
from ..utils.attribute_groups_file import ensure_attribute_groups_file
from ..utils.token_hidden_store import (
    TokenHiddenStore,
    build_hidden_store_config,
    parse_token_ids_with_bos_alias,
)


def _get_or_start_runtime_api(config: dict[str, Any]):
    try:
        return get_runtime_api()
    except RuntimeError:
        return start_llama_api(config)


def _safe_name(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip())
    text = text.strip("._-")
    return text or "group"


def _normalize_groups_payload(raw_json: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(str(raw_json or "").strip())
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc.msg}"

    if isinstance(payload, dict) and isinstance(payload.get("groups"), list):
        groups = list(payload.get("groups") or [])
    elif isinstance(payload, list):
        groups = list(payload)
        payload = {"groups": groups}
    elif isinstance(payload, dict):
        groups = [payload]
        payload = {"groups": groups}
    else:
        return None, "invalid_json_type:root_must_be_object_or_array"

    clean_groups: list[dict[str, Any]] = []
    for idx, group in enumerate(groups):
        if not isinstance(group, dict):
            return None, f"invalid_json_field:groups[{idx}]_must_be_object"
        group_name = str(group.get("group_name") or group.get("name") or f"group_{idx + 1}").strip()
        if not group_name:
            return None, f"invalid_json_field:groups[{idx}].group_name_invalid"
        tokens_raw = group.get("tokens")
        if isinstance(tokens_raw, str):
            # Support comma-separated tokens for easier manual editing.
            tokens = [x.strip() for x in tokens_raw.split(",") if str(x).strip()]
        elif isinstance(tokens_raw, list):
            tokens = list(tokens_raw)
        else:
            tokens = []
        if not tokens:
            return None, f"invalid_json_field:groups[{idx}].tokens_must_be_non_empty_array_or_csv_string"
        filter_cfg = group.get("filter")
        if filter_cfg is None:
            filter_cfg = {}
        if not isinstance(filter_cfg, dict):
            return None, f"invalid_json_field:groups[{idx}].filter_must_be_object"
        clean_groups.append(
            {
                "group_name": group_name,
                "tokens": list(tokens),
                "filter": dict(filter_cfg),
            }
        )

    names = [str(x["group_name"]) for x in clean_groups]
    if len(set(names)) != len(names):
        return None, "invalid_json_field:group_name_must_be_unique"
    return {"groups": clean_groups}, None


def _pick_group(payload: dict[str, Any], selected_group: str) -> tuple[dict[str, Any] | None, str | None]:
    groups = list((payload or {}).get("groups") or [])
    selected = str(selected_group or "").strip()
    if not groups:
        return None, "groups_empty"
    if selected:
        found = next((x for x in groups if str(x.get("group_name") or "") == selected), None)
        if found is None:
            return None, "selected_group_not_found"
        return found, None
    if len(groups) == 1:
        return groups[0], None
    return None, "selected_group_required_when_multiple_groups"


def _resolve_token_id(tokenizer, item: Any) -> tuple[int | None, str | None]:
    # Accept explicit token id (int / numeric string), or single-token text.
    if isinstance(item, int):
        return int(item), None
    text = str(item).strip()
    if not text:
        return None, "empty_token"
    if re.fullmatch(r"-?\d+", text):
        return int(text), None
    try:
        ids = [int(x) for x in parse_token_ids_with_bos_alias(tokenizer, text)]
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    if len(ids) != 1:
        return None, "token_not_single"
    return int(ids[0]), None


def _collect_token_layers(
    *,
    bundle,
    store: TokenHiddenStore,
    token_items: list[Any],
) -> tuple[list[int], dict[int, np.ndarray], list[dict[str, Any]]]:
    token_ids: list[int] = []
    errors: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in token_items:
        token_id, err = _resolve_token_id(bundle.tokenizer, item)
        if token_id is None:
            errors.append({"token": str(item), "reason": str(err or "invalid_token")})
            continue
        if token_id in seen:
            continue
        seen.add(token_id)
        token_ids.append(int(token_id))

    if not token_ids:
        return [], {}, errors

    batch = store.get_or_compute_layers_batch(bundle, token_ids, flush=True)
    out: dict[int, np.ndarray] = {}
    for token_id in token_ids:
        arr = np.asarray(batch.get(int(token_id)), dtype=np.float32)
        if arr.ndim != 2 or arr.shape != (store.cfg.n_layers, store.cfg.hidden_dim):
            errors.append(
                {
                    "token_id": int(token_id),
                    "reason": "shape_mismatch",
                    "actual_shape": list(arr.shape) if hasattr(arr, "shape") else [],
                    "expected_shape": [int(store.cfg.n_layers), int(store.cfg.hidden_dim)],
                }
            )
            continue
        out[int(token_id)] = arr
    token_ids_ok = [tid for tid in token_ids if tid in out]
    return token_ids_ok, out, errors


def _collect_group_context(
    *,
    group: dict[str, Any],
    bundle,
    store: TokenHiddenStore,
    config: dict[str, Any],
) -> dict[str, Any]:
    token_ids, layers_by_token, token_errors = _collect_token_layers(
        bundle=bundle,
        store=store,
        token_items=list(group.get("tokens") or []),
    )
    if token_errors:
        return {
            "ok": False,
            "reason": "token_resolution_failed",
            "token_ids": token_ids,
            "layers_by_token": layers_by_token,
            "token_errors": token_errors,
        }
    if not token_ids:
        return {
            "ok": False,
            "reason": "no_valid_tokens_after_resolution",
            "token_ids": [],
            "layers_by_token": {},
            "token_errors": token_errors,
        }

    filter_cfg = dict(group.get("filter") or {})
    baseline_ref = get_or_build_random_token_mean_matrix(
        config=config,
        include_bos=True,
        include_assistant=False,
        sample_size=int(filter_cfg.get("random1000_sample_size", 1000)),
        seed=int(filter_cfg.get("random1000_seed", 20260526)),
    )

    return {
        "ok": True,
        "token_ids": token_ids,
        "layers_by_token": layers_by_token,
        "token_errors": token_errors,
        "random1000_mean_ref": baseline_ref,
    }


def _compute_group_statistics(
    *,
    token_ids: list[int],
    layers_by_token: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-neuron statistics from resolved token hidden states."""
    stacked = np.stack([layers_by_token[int(tid)] for tid in token_ids], axis=0)  # [N, L, H]
    mean = np.mean(stacked, axis=0, dtype=np.float32)
    std = np.std(stacked, axis=0, dtype=np.float32)
    abs_mean = np.abs(mean, dtype=np.float32)
    return mean, std, abs_mean


def _select_neurons(
    *,
    group: dict[str, Any],
    context: dict[str, Any],
    mean: np.ndarray,
    abs_mean: np.ndarray,
) -> dict[int, np.ndarray]:
    filter_cfg = dict(group.get("filter") or {})
    algo_raw = filter_cfg.get("algorithm", 1)
    try:
        algorithm = int(algo_raw)
    except (TypeError, ValueError):
        algorithm = 1
    if algorithm == 1:
        return _filter_algorithm_1(group=group, context=context, mean=mean, abs_mean=abs_mean)
    if algorithm == 2:
        return _filter_algorithm_2(group=group, abs_mean=abs_mean)
    if algorithm == 3:
        return _filter_algorithm_3(group=group, abs_mean=abs_mean)
    # Unknown algorithm -> fallback to algorithm 1.
    return _filter_algorithm_1(group=group, context=context, mean=mean, abs_mean=abs_mean)


def _filter_algorithm_1(
    *,
    group: dict[str, Any],
    context: dict[str, Any],
    mean: np.ndarray,
    abs_mean: np.ndarray,
) -> dict[int, np.ndarray]:
    """Filter algorithm v1 (current requested behavior).

    Step 1 (candidate1):
    - Fixed layer = 8.
    - Compare group mean hidden-state vs random-1000-token mean baseline at layer 8.
    - Keep neurons where abs(diff) > 0.2.
    - Keep neurons where abs(group_mean) >= 0.1.

    Step 2 (candidate2):
    - For a candidate neuron, there exists at least one token such that:
      abs(token_value) >= 0.2 and abs(token_value - group_mean) >= 0.2.
    """
    mean = np.asarray(mean, dtype=np.float32)
    abs_mean = np.asarray(abs_mean, dtype=np.float32)
    if abs_mean.ndim != 2 or mean.ndim != 2:
        return {}
    layers = int(mean.shape[0])
    hidden_dim = int(mean.shape[1])
    selected_by_layer: dict[int, np.ndarray] = {int(i): np.array([], dtype=np.int32) for i in range(layers)}

    target_layer = 8
    if target_layer < 0 or target_layer >= layers:
        return selected_by_layer

    baseline_ref = dict((context or {}).get("random1000_mean_ref") or {})
    baseline_matrix = np.asarray(baseline_ref.get("matrix"), dtype=np.float32)
    if baseline_matrix.ndim != 2 or baseline_matrix.shape != mean.shape:
        return selected_by_layer

    g = np.asarray(mean[target_layer], dtype=np.float32)
    b = np.asarray(baseline_matrix[target_layer], dtype=np.float32)
    diff = np.abs(g - b)
    baseline_zero = np.abs(b) <= 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(g, b, out=np.full_like(g, np.nan, dtype=np.float32), where=~baseline_zero)
    ratio_cond = np.logical_or(
        baseline_zero,
        np.logical_or(ratio < 0.0, ratio > 3.0),
    )
    mean_abs_cond = np.abs(g) >= 0.1
    candidate1 = np.where(np.logical_and(np.logical_and(diff > 0.2, ratio_cond), mean_abs_cond))[0].astype(np.int32)

    layers_by_token = dict((context or {}).get("layers_by_token") or {})
    token_ids = [int(x) for x in ((context or {}).get("token_ids") or [])]
    candidate2 = np.array([], dtype=np.int32)
    if token_ids and layers_by_token:
        # token_matrix shape: [N, H] at target_layer
        token_rows: list[np.ndarray] = []
        for tid in token_ids:
            arr = np.asarray(layers_by_token.get(int(tid)), dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] <= target_layer:
                continue
            token_rows.append(np.asarray(arr[target_layer], dtype=np.float32))
        if token_rows:
            token_matrix = np.stack(token_rows, axis=0)
            selected: list[int] = []
            for nid in range(hidden_dim):
                nv = token_matrix[:, int(nid)]
                m = float(mean[target_layer, int(nid)])
                m_zero = abs(m) <= 1e-12
                if m_zero:
                    ratio_cond = np.ones_like(nv, dtype=bool)
                else:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = nv / m
                    ratio_cond = np.logical_or(ratio < 0.0, ratio > 3.0)
                cond = np.logical_and(np.abs(nv) >= 0.2, ratio_cond)
                if bool(np.any(cond)):
                    selected.append(int(nid))
            candidate2 = np.asarray(selected, dtype=np.int32)

    # candidate1 OR candidate2, keep boundary/order and keep duplicates.
    filtered_neurons = np.asarray(candidate1.tolist() + candidate2.tolist(), dtype=np.int32)
    selected_by_layer[int(target_layer)] = filtered_neurons
    return selected_by_layer


def _filter_algorithm_2(
    *,
    group: dict[str, Any],
    abs_mean: np.ndarray,
) -> dict[int, np.ndarray]:
    """Placeholder for filter algorithm v2."""
    return _filter_algorithm_3(group=group, abs_mean=abs_mean)


def _filter_algorithm_3(
    *,
    group: dict[str, Any],
    abs_mean: np.ndarray,
) -> dict[int, np.ndarray]:
    """Legacy selection behavior (old default before algorithm split)."""
    abs_mean = np.asarray(abs_mean, dtype=np.float32)
    if abs_mean.ndim != 2:
        return {}
    layers = int(abs_mean.shape[0])
    hidden_dim = int(abs_mean.shape[1])
    filter_cfg = dict(group.get("filter") or {})
    k = int(filter_cfg.get("top_k_per_layer", 0))
    min_abs_mean = float(filter_cfg.get("min_abs_mean", 0.0))

    selected_by_layer: dict[int, np.ndarray] = {}
    for layer_idx in range(layers):
        row_abs = abs_mean[layer_idx]
        if k > 0 and k < hidden_dim:
            idx = np.argpartition(-row_abs, k - 1)[:k]
            selected = np.sort(idx.astype(np.int32))
        else:
            selected = np.arange(hidden_dim, dtype=np.int32)
        if min_abs_mean > 0:
            mask = row_abs[selected] >= min_abs_mean
            selected = selected[mask]
        selected_by_layer[int(layer_idx)] = selected
    return selected_by_layer


def _build_token_text_columns(tokenizer, token_ids: list[int]) -> list[str]:
    """Build unique UTF-8 token-text column labels for CSV header."""
    labels: list[str] = []
    used: dict[str, int] = {}
    for tid in token_ids:
        text = str(tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False))
        if not text:
            text = str(tokenizer.convert_ids_to_tokens([int(tid)])[0] or "")
        if not text:
            text = f"<token_{int(tid)}>"
        base = text
        count = used.get(base, 0)
        if count > 0:
            text = f"{base}#{count + 1}"
        used[base] = count + 1
        labels.append(text)
    return labels


def _build_layer_matrix_rows(
    *,
    target_layer: int,
    neuron_list: list[int],
    token_ids: list[int],
    token_text_columns: list[str],
    layers_by_token: dict[int, np.ndarray],
    baseline_matrix: np.ndarray | None,
    mean_matrix: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Matrix rows: y-axis neuron_id, x-axis token text, cell = layer value."""
    if not neuron_list:
        return []
    baseline = None
    if baseline_matrix is not None:
        arr = np.asarray(baseline_matrix, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] > int(target_layer):
            baseline = arr
    mean_ref = None
    if mean_matrix is not None:
        arr = np.asarray(mean_matrix, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] > int(target_layer):
            mean_ref = arr
    rows: list[dict[str, Any]] = []
    for neuron_id in neuron_list:
        baseline_value = float(baseline[int(target_layer), int(neuron_id)]) if baseline is not None else 0.0
        average_value = float(mean_ref[int(target_layer), int(neuron_id)]) if mean_ref is not None else 0.0
        row: dict[str, Any] = {"neuron_id": int(neuron_id), "baseline": baseline_value, "average": average_value}
        for tid, col in zip(token_ids, token_text_columns):
            arr = np.asarray(layers_by_token[int(tid)], dtype=np.float32)
            value = float(arr[int(target_layer), int(neuron_id)])
            row[col] = value
        rows.append(row)
    return rows


def _write_csv(
    *,
    project_root: Path,
    group_name: str,
    target_layer: int,
    neuron_list: list[int],
    matrix_rows: list[dict[str, Any]],
) -> Path:
    out_dir = (project_root / "data" / "outputs" / "attribute_group_neurons").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = (
        f"{_safe_name(group_name)}__layer{int(target_layer)}"
        f"__neurons{int(len(neuron_list))}__{stamp}.csv"
    )
    csv_path = (out_dir / file_name).resolve()

    if matrix_rows:
        headers = list(matrix_rows[0].keys())
    else:
        headers = ["neuron_id"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        if matrix_rows:
            writer.writerows(matrix_rows)
    return csv_path


def run_study(
    *,
    attribute_groups_json: str,
    selected_attribute_group: str = "",
    config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/custom.yaml",
) -> dict[str, Any]:
    cfg = config or load_config(config_path)
    project_root = Path(__file__).resolve().parents[2]
    groups_file = ensure_attribute_groups_file(project_root)

    normalized_payload, parse_err = _normalize_groups_payload(attribute_groups_json)
    if normalized_payload is None:
        return {
            "ok": False,
            "reason": str(parse_err or "invalid_json"),
            "groups_file": str(groups_file),
            "ui_tasks": [{"name": "render_text_output", "value_key": "summary_text"}],
            "summary_text": f"Attribute group JSON invalid: {parse_err}",
        }
    groups_file.write_text(
        json.dumps(normalized_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    group, pick_err = _pick_group(normalized_payload, str(selected_attribute_group or ""))
    if group is None:
        names = [str(x.get("group_name") or "") for x in (normalized_payload.get("groups") or [])]
        return {
            "ok": False,
            "reason": str(pick_err or "group_select_failed"),
            "groups_file": str(groups_file),
            "available_groups": names,
            "ui_tasks": [{"name": "render_text_output", "value_key": "summary_text"}],
            "summary_text": f"Group selection failed: {pick_err}. available={names}",
        }

    api = _get_or_start_runtime_api(cfg)
    bundle = api.execute_model_call(RuntimeRequest(config=cfg, force_reload=False)).bundle
    store_cfg = build_hidden_store_config(cfg, bundle=bundle)
    store = TokenHiddenStore(store_cfg, bundle.tokenizer)

    context = _collect_group_context(
        group=group,
        bundle=bundle,
        store=store,
        config=cfg,
    )
    token_ids = [int(x) for x in (context.get("token_ids") or [])]
    layers_by_token = dict(context.get("layers_by_token") or {})
    token_errors = list(context.get("token_errors") or [])
    if not bool(context.get("ok")) or not token_ids:
        reason = str(context.get("reason") or "no_valid_tokens_after_resolution")
        return {
            "ok": False,
            "reason": reason,
            "groups_file": str(groups_file),
            "selected_attribute_group": str(group.get("group_name") or ""),
            "token_errors": token_errors,
            "ui_tasks": [{"name": "render_text_output", "value_key": "summary_text"}],
            "summary_text": (
                f"Token validation failed for group {group.get('group_name')}. "
                f"reason={reason}. token_errors={token_errors}"
            ),
        }

    filter_cfg = dict(group.get("filter") or {})
    min_abs_mean = float(filter_cfg.get("min_abs_mean", 0.0))
    top_k_per_layer = int(filter_cfg.get("top_k_per_layer", 0))
    try:
        filter_algorithm = int(filter_cfg.get("algorithm", 1))
    except (TypeError, ValueError):
        filter_algorithm = 1
    mean, std, abs_mean = _compute_group_statistics(
        token_ids=token_ids,
        layers_by_token=layers_by_token,
    )
    selected_by_layer = _select_neurons(
        group=group,
        context=context,
        mean=mean,
        abs_mean=abs_mean,
    )
    target_layer = 8
    neuron_list = [
        int(x) for x in np.asarray(selected_by_layer.get(int(target_layer), np.array([], dtype=np.int32))).tolist()
    ]
    token_text_columns = _build_token_text_columns(bundle.tokenizer, token_ids)
    baseline_ref = dict((context.get("random1000_mean_ref") or {}))
    baseline_matrix = baseline_ref.get("matrix")
    matrix_rows = _build_layer_matrix_rows(
        target_layer=int(target_layer),
        neuron_list=neuron_list,
        token_ids=token_ids,
        token_text_columns=token_text_columns,
        layers_by_token=layers_by_token,
        baseline_matrix=baseline_matrix if isinstance(baseline_matrix, (list, np.ndarray)) else None,
        mean_matrix=mean,
    )
    csv_path = _write_csv(
        project_root=project_root,
        group_name=str(group.get("group_name") or ""),
        target_layer=int(target_layer),
        neuron_list=neuron_list,
        matrix_rows=matrix_rows,
    )

    summary = (
        f"study=attribute_group_neurons\n"
        f"group={group.get('group_name')}\n"
        f"layer={target_layer}\n"
        f"token_count={len(token_ids)}\n"
        f"selected_neurons={len(neuron_list)}\n"
        f"csv_rows={len(matrix_rows)}\n"
        f"csv={csv_path.as_posix()}\n"
        f"filter.algorithm={filter_algorithm}\n"
        f"filter.min_abs_mean={min_abs_mean}\n"
        f"filter.top_k_per_layer={top_k_per_layer}\n"
        f"random1000_cache_source={((context.get('random1000_mean_ref') or {}).get('cache_source', 'none'))}\n"
        f"token_errors={json.dumps(token_errors, ensure_ascii=False)}"
    )
    return {
        "ok": True,
        "study": "attribute_group_neurons",
        "selected_attribute_group": str(group.get("group_name") or ""),
        "groups_file": str(groups_file),
        "token_ids": [int(x) for x in token_ids],
        "token_text_columns": [str(x) for x in token_text_columns],
        "target_layer": int(target_layer),
        "neuron_list": [int(x) for x in neuron_list],
        "token_errors": token_errors,
        "rows_written": int(len(matrix_rows)),
        "filter_algorithm": int(filter_algorithm),
        "csv_path": csv_path.as_posix(),
        "summary_text": summary,
        "ui_tasks": [{"name": "render_text_output", "value_key": "summary_text"}],
    }
