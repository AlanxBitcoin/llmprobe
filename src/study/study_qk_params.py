from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

from ..config import load_config

_Q_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$")
_K_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$")


def _error_payload(reason: str, **extra: Any) -> dict[str, Any]:
    out = {
        "ok": False,
        "reason": str(reason or "unknown"),
        "matrix": [],
        "heatmaps": [],
        "ui_tasks": [{"name": "render_heatmap", "value_key": "heatmaps"}],
    }
    out.update(extra)
    return out


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_index(index_path: Path) -> dict[str, str]:
    payload = _read_json(index_path)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("invalid_index_weight_map")
    out: dict[str, str] = {}
    for k, v in weight_map.items():
        key = str(k or "").strip()
        val = str(v or "").strip()
        if key and val:
            out[key] = val
    return out


def _build_layer_tensor_map(weight_map: dict[str, str]) -> tuple[dict[int, tuple[str, Path]], dict[int, tuple[str, Path]]]:
    q_map: dict[int, tuple[str, Path]] = {}
    k_map: dict[int, tuple[str, Path]] = {}
    for tensor_name, rel_file in weight_map.items():
        mq = _Q_KEY_RE.match(tensor_name)
        if mq:
            q_map[int(mq.group(1))] = (tensor_name, Path(rel_file))
            continue
        mk = _K_KEY_RE.match(tensor_name)
        if mk:
            k_map[int(mk.group(1))] = (tensor_name, Path(rel_file))
    return q_map, k_map


def _read_arch_config(model_dir: Path) -> dict[str, Any]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return {}
    try:
        payload = _read_json(cfg_path)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def run_study(
    *,
    view_by_layer: bool = True,
    view_by_head: bool = False,
    selected_layer: int = 1,
    config: dict[str, Any] | None = None,
    config_path: str | Path = "configs/custom.yaml",
) -> dict[str, Any]:
    cfg = config or load_config(config_path)
    model_dir = Path(str(((cfg or {}).get("model") or {}).get("model_name_or_path") or "")).resolve()
    if not str(model_dir):
        return _error_payload("model_name_or_path_missing")
    if not model_dir.exists():
        return _error_payload("model_dir_not_found", model_dir=str(model_dir))

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return _error_payload("safetensors_index_not_found", index_path=str(index_path))

    if not bool(view_by_layer):
        return _error_payload(
            "view_by_layer_required_for_current_qk_headmaps",
            view_by_layer=bool(view_by_layer),
            view_by_head=bool(view_by_head),
        )

    try:
        weight_map = _read_index(index_path)
        q_layer_map, k_layer_map = _build_layer_tensor_map(weight_map)
    except Exception as exc:  # noqa: BLE001
        return _error_payload(f"failed_to_read_index:{exc}", index_path=str(index_path))

    if not q_layer_map or not k_layer_map:
        return _error_payload("qk_tensors_not_found_in_index", index_path=str(index_path))

    common_layers = sorted(set(q_layer_map.keys()) & set(k_layer_map.keys()))
    if not common_layers:
        return _error_payload("qk_layer_intersection_empty")

    layer_number = int(selected_layer or 1)
    layer_idx = int(layer_number - 1)
    if layer_idx not in q_layer_map or layer_idx not in k_layer_map:
        return _error_payload(
            "selected_layer_out_of_range",
            selected_layer=int(layer_number),
            valid_layer_min=1,
            valid_layer_max=int(max(common_layers) + 1),
        )

    q_name, q_rel = q_layer_map[layer_idx]
    k_name, k_rel = k_layer_map[layer_idx]
    q_file = (model_dir / q_rel).resolve()
    k_file = (model_dir / k_rel).resolve()
    if not q_file.exists() or not k_file.exists():
        return _error_payload(
            "safetensors_shard_missing",
            layer=int(layer_number),
            q_file=str(q_file),
            k_file=str(k_file),
        )

    try:
        q_sf = safe_open(str(q_file), framework="pt", device="cpu")
        k_sf = safe_open(str(k_file), framework="pt", device="cpu")
        q_tensor = q_sf.get_tensor(q_name)
        k_tensor = k_sf.get_tensor(k_name)
    except Exception as exc:  # noqa: BLE001
        return _error_payload("failed_to_read_qk_tensors", error=str(exc), layer=int(layer_number))

    if q_tensor.ndim != 2 or k_tensor.ndim != 2:
        return _error_payload(
            "invalid_qk_rank",
            layer=int(layer_number),
            q_shape=[int(x) for x in q_tensor.shape],
            k_shape=[int(x) for x in k_tensor.shape],
        )
    if int(q_tensor.shape[1]) != int(k_tensor.shape[1]):
        return _error_payload(
            "qk_in_features_mismatch",
            layer=int(layer_number),
            q_shape=[int(x) for x in q_tensor.shape],
            k_shape=[int(x) for x in k_tensor.shape],
        )

    arch_cfg = _read_arch_config(model_dir)
    hidden_size = int(arch_cfg.get("hidden_size") or q_tensor.shape[1])
    num_q_heads = int(arch_cfg.get("num_attention_heads") or 0)
    if num_q_heads <= 0:
        num_q_heads = int(q_tensor.shape[0] // 128)
    if num_q_heads <= 0:
        return _error_payload("invalid_num_attention_heads", num_attention_heads=int(num_q_heads))

    head_dim = int(hidden_size // num_q_heads)
    if head_dim <= 0:
        return _error_payload("invalid_head_dim", hidden_size=int(hidden_size), num_q_heads=int(num_q_heads))
    if int(q_tensor.shape[0]) != int(num_q_heads * head_dim):
        return _error_payload(
            "q_shape_head_mismatch",
            q_shape=[int(x) for x in q_tensor.shape],
            num_q_heads=int(num_q_heads),
            head_dim=int(head_dim),
        )

    num_kv_heads = int(arch_cfg.get("num_key_value_heads") or 0)
    if num_kv_heads <= 0:
        num_kv_heads = int(k_tensor.shape[0] // head_dim)
    if num_kv_heads <= 0:
        return _error_payload("invalid_num_kv_heads", num_kv_heads=int(num_kv_heads))
    if int(k_tensor.shape[0]) != int(num_kv_heads * head_dim):
        return _error_payload(
            "k_shape_head_mismatch",
            k_shape=[int(x) for x in k_tensor.shape],
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
        )

    q_per_k = max(1, int(num_q_heads // num_kv_heads))
    q_np = q_tensor.float().cpu().numpy().astype(np.float32, copy=False)
    k_np = k_tensor.float().cpu().numpy().astype(np.float32, copy=False)

    heatmaps: list[dict[str, Any]] = []
    for q_head in range(int(num_q_heads)):
        kv_head = min(int(num_kv_heads - 1), int(q_head // q_per_k))
        q_start = int(q_head * head_dim)
        q_end = int(q_start + head_dim)
        k_start = int(kv_head * head_dim)
        k_end = int(k_start + head_dim)
        wq_h = q_np[q_start:q_end, :]
        wk_h = k_np[k_start:k_end, :]
        # 128x4096 @ 4096x128 -> 128x128
        qk_head = (wq_h @ wk_h.T).astype(np.float32, copy=False)
        heatmaps.append(
            {
                "key": f"head_{int(q_head)}",
                "title": f"Layer {int(layer_number)} Head {int(q_head)} WQ*WK (rows={head_dim}, cols={head_dim})",
                "matrix": qk_head.tolist(),
                "hover_x_label": "neuron(global WK)",
                "hover_y_label": "neuron(global WQ)",
                "hover_x_offset": int(k_start),
                "hover_y_offset": int(q_start),
            }
        )

    first_matrix = np.asarray((heatmaps[0] or {}).get("matrix") if heatmaps else [], dtype=np.float32)
    first_rows = int(first_matrix.shape[0]) if first_matrix.ndim == 2 else 0
    first_cols = int(first_matrix.shape[1]) if first_matrix.ndim == 2 else 0

    return {
        "ok": True,
        "study": "qk_params",
        "source": "model_files_safetensors",
        "model_dir": str(model_dir),
        "index_path": str(index_path),
        "metric": "head_local_wq_mul_wk_transpose",
        "selected_layer": int(layer_number),
        "num_layers": int(max(common_layers) + 1),
        "num_q_heads": int(num_q_heads),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
        "view_by_layer": True,
        "view_by_head": bool(view_by_head),
        "rows": first_rows,
        "cols": first_cols,
        "hidden_size": int(hidden_size),
        "q_proj_shape": [int(q_tensor.shape[0]), int(q_tensor.shape[1])],
        "k_proj_shape": [int(k_tensor.shape[0]), int(k_tensor.shape[1])],
        "matrix": (heatmaps[0].get("matrix") if heatmaps else []),
        "heatmaps": heatmaps,
        "ui_tasks": [{"name": "render_heatmap", "value_key": "heatmaps"}],
    }
