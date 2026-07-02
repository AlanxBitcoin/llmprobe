from __future__ import annotations

import argparse
import re
import unicodedata
import copy
import sys
from types import SimpleNamespace
from typing import Any

import torch
from ..probes.probe_attention import fetch_head_attention_metrics_for_input_ids
from ..probes.probe_chat_common import get_assistant_ids
from ..probes.probe_layer_neuron import run_starting_from_middle_layer_probe
from ..probes.probe_layer_shortcut import validate_shortcut_layers_zero_based
from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api
from ..utils.token_hidden_store import resolve_assistant_token_id



def _replace_target_in_user_messages(
    messages: list[dict[str, str]],
    *,
    target: str,
    replacement: str,
) -> tuple[list[dict[str, str]], int]:
    replaced_count = 0
    out: list[dict[str, str]] = []
    for item in messages:
        role = str(item.get("role") or "")
        content = str(item.get("content") or "")
        if role == "user":
            count = content.count(target)
            replaced_count += int(count)
            if count > 0:
                content = content.replace(target, replacement)
        out.append({"role": role, "content": content})
    return out, replaced_count


def _as_layer_sequence(past_key_values: Any) -> tuple[Any, ...]:
    if past_key_values is None:
        raise ValueError("past_key_values is None")
    if isinstance(past_key_values, (list, tuple)):
        return tuple(past_key_values)
    layers = getattr(past_key_values, "layers", None)
    if isinstance(layers, (list, tuple)):
        return tuple(layers)
    try:
        return tuple(iter(past_key_values))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unsupported past_key_values type: {type(past_key_values)!r}") from exc


def _merge_past_key_values(
    *,
    original: Any,
    replaced: Any,
    replace_layer_indices_zero_based: set[int],
    replace_k: bool,
    kv_replace_mode: int = 3,
    target_token_index: int | None = None,
) -> Any:
    # Preferred path for transformers>=5 Cache objects (e.g. DynamicCache):
    # mutate original.layers in place to keep Cache type and metadata intact.
    original_layers_obj = getattr(original, "layers", None)
    replaced_layers_obj = getattr(replaced, "layers", None)
    if isinstance(original_layers_obj, (list, tuple)) and isinstance(replaced_layers_obj, (list, tuple)):
        if len(original_layers_obj) != len(replaced_layers_obj):
            raise ValueError(
                f"Cannot merge cache layers with different counts: {len(original_layers_obj)} vs {len(replaced_layers_obj)}"
            )
        if int(kv_replace_mode) in (1, 2) and target_token_index is None:
            raise ValueError("kv_replace_mode=1/2 requires target_token_index")
        mode2_boundary_layer = max(replace_layer_indices_zero_based) if replace_layer_indices_zero_based else -1
        for i in range(len(original_layers_obj)):
            in_replace_layer = int(i) in replace_layer_indices_zero_based
            mode2_after_replace_layer = int(kv_replace_mode) == 2 and int(i) > int(mode2_boundary_layer)
            if not in_replace_layer and not mode2_after_replace_layer:
                continue
            original_layer = original_layers_obj[i]
            replaced_layer = replaced_layers_obj[i]
            # CacheLayer path (transformers>=5): replace V always; replace K only when requested.
            if hasattr(original_layer, "values") and hasattr(replaced_layer, "values"):
                if int(kv_replace_mode) == 1 or (int(kv_replace_mode) == 2 and in_replace_layer):
                    pos = int(target_token_index)
                    if int(getattr(original_layer.values, "shape", [0, 0, 0, 0])[2]) <= pos:
                        raise ValueError(f"target token index out of range for original V cache: idx={pos}")
                    if int(getattr(replaced_layer.values, "shape", [0, 0, 0, 0])[2]) <= pos:
                        raise ValueError(f"target token index out of range for replaced V cache: idx={pos}")
                    original_layer.values[:, :, pos, :] = replaced_layer.values[:, :, pos, :]
                elif int(kv_replace_mode) == 2 and mode2_after_replace_layer:
                    start = int(target_token_index) + 1
                    seq_len = int(getattr(original_layer.values, "shape", [0, 0, 0, 0])[2])
                    if start < seq_len:
                        original_layer.values[:, :, start:, :] = replaced_layer.values[:, :, start:, :]
                else:
                    original_layer.values = replaced_layer.values
            else:
                if isinstance(original_layers_obj, list):
                    original_layers_obj[i] = replaced_layer
                continue
            if bool(replace_k):
                if hasattr(original_layer, "keys") and hasattr(replaced_layer, "keys"):
                    if int(kv_replace_mode) == 1 or (int(kv_replace_mode) == 2 and in_replace_layer):
                        pos = int(target_token_index)
                        if int(getattr(original_layer.keys, "shape", [0, 0, 0, 0])[2]) <= pos:
                            raise ValueError(f"target token index out of range for original K cache: idx={pos}")
                        if int(getattr(replaced_layer.keys, "shape", [0, 0, 0, 0])[2]) <= pos:
                            raise ValueError(f"target token index out of range for replaced K cache: idx={pos}")
                        original_layer.keys[:, :, pos, :] = replaced_layer.keys[:, :, pos, :]
                    elif int(kv_replace_mode) == 2 and mode2_after_replace_layer:
                        start = int(target_token_index) + 1
                        seq_len = int(getattr(original_layer.keys, "shape", [0, 0, 0, 0])[2])
                        if start < seq_len:
                            original_layer.keys[:, :, start:, :] = replaced_layer.keys[:, :, start:, :]
                    else:
                        original_layer.keys = replaced_layer.keys
                else:
                    if isinstance(original_layers_obj, list):
                        original_layers_obj[i] = replaced_layer
        return original

    # Legacy tuple/list fallback.
    original_layers = _as_layer_sequence(original)
    replaced_layers = _as_layer_sequence(replaced)
    if len(original_layers) != len(replaced_layers):
        raise ValueError(
            f"Cannot merge caches with different layer counts: {len(original_layers)} vs {len(replaced_layers)}"
        )
    if int(kv_replace_mode) in (1, 2) and target_token_index is None:
        raise ValueError("kv_replace_mode=1/2 requires target_token_index")
    mode2_boundary_layer = max(replace_layer_indices_zero_based) if replace_layer_indices_zero_based else -1
    out: list[Any] = []
    for i in range(len(original_layers)):
        in_replace_layer = int(i) in replace_layer_indices_zero_based
        mode2_after_replace_layer = int(kv_replace_mode) == 2 and int(i) > int(mode2_boundary_layer)
        if not in_replace_layer and not mode2_after_replace_layer:
            out.append(original_layers[i])
            continue
        if int(kv_replace_mode) == 3 and bool(replace_k):
            out.append(replaced_layers[i])
            continue
        orig_item = original_layers[i]
        repl_item = replaced_layers[i]
        if isinstance(orig_item, (list, tuple)) and isinstance(repl_item, (list, tuple)):
            if len(orig_item) >= 2 and len(repl_item) >= 2:
                merged = list(orig_item)
                if int(kv_replace_mode) == 1 or (int(kv_replace_mode) == 2 and in_replace_layer):
                    pos = int(target_token_index)
                    if torch.is_tensor(orig_item[1]) and torch.is_tensor(repl_item[1]):
                        merged_v = orig_item[1].clone()
                        merged_v[:, :, pos, :] = repl_item[1][:, :, pos, :]
                        merged[1] = merged_v
                    if bool(replace_k) and torch.is_tensor(orig_item[0]) and torch.is_tensor(repl_item[0]):
                        merged_k = orig_item[0].clone()
                        merged_k[:, :, pos, :] = repl_item[0][:, :, pos, :]
                        merged[0] = merged_k
                elif int(kv_replace_mode) == 2 and mode2_after_replace_layer:
                    start = int(target_token_index) + 1
                    if torch.is_tensor(orig_item[1]) and torch.is_tensor(repl_item[1]):
                        merged_v = orig_item[1].clone()
                        if start < int(merged_v.shape[2]):
                            merged_v[:, :, start:, :] = repl_item[1][:, :, start:, :]
                        merged[1] = merged_v
                    if bool(replace_k) and torch.is_tensor(orig_item[0]) and torch.is_tensor(repl_item[0]):
                        merged_k = orig_item[0].clone()
                        if start < int(merged_k.shape[2]):
                            merged_k[:, :, start:, :] = repl_item[0][:, :, start:, :]
                        merged[0] = merged_k
                else:
                    merged[1] = repl_item[1]  # replace V only
                out.append(tuple(merged))
                continue
        out.append(orig_item)
    return tuple(out)


def _parse_replace_layers_spec(spec: str, *, layer_count: int) -> tuple[set[int], list[int]]:
    raw = str(spec or "").strip()
    if not raw:
        raise ValueError("replace_layers is required.")
    normalized = unicodedata.normalize('NFKC', raw).strip()
    parts = [p for p in normalized.split(",") if p]
    if not parts:
        raise ValueError("replace_layers is empty.")
    selected_zero_based: set[int] = set()
    for part in parts:
        if "-" in part:
            if part.count("-") != 1:
                raise ValueError(f"Invalid replace_layers segment: {part!r}")
            left, right = part.split("-", 1)
            if not left:
                raise ValueError(f"Invalid range start in segment: {part!r}")
            try:
                start = int(left)
            except ValueError as exc:
                raise ValueError(f"Invalid range start in segment: {part!r}") from exc
            if not right:
                end = int(layer_count) - 1
            else:
                try:
                    end = int(right)
                except ValueError as exc:
                    raise ValueError(f"Invalid range end in segment: {part!r}") from exc
            if start < 0 or end < 0 or start > end:
                raise ValueError(f"Invalid layer range segment: {part!r}")
            for v in range(start, end + 1):
                selected_zero_based.add(int(v))
        else:
            try:
                v = int(part)
            except ValueError as exc:
                raise ValueError(f"Invalid layer index segment: {part!r}") from exc
            if v < 0:
                raise ValueError(f"Layer index must be >=0: {part!r}")
            selected_zero_based.add(int(v))

    if not selected_zero_based:
        raise ValueError("replace_layers selects no layers.")
    max_sel = max(selected_zero_based)
    if max_sel >= int(layer_count):
        raise ValueError(f"replace_layers out of range: max={max_sel}, valid=[0,{int(layer_count) - 1}]")
    sorted_zero_based = sorted(selected_zero_based)
    return set(sorted_zero_based), sorted_zero_based


def _sample_next_id(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> int:
    if not do_sample:
        return int(torch.argmax(logits, dim=-1).item())

    scaled = logits / max(float(temperature), 1e-5)
    probs = torch.softmax(scaled, dim=-1)
    if float(top_p) < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > float(top_p)
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        denom = torch.sum(sorted_probs, dim=-1, keepdim=True).clamp_min(1e-12)
        sorted_probs = sorted_probs / denom
        sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
        sampled_id = torch.gather(sorted_indices, dim=-1, index=sampled_sorted_idx)
        return int(sampled_id.item())
    sampled = torch.multinomial(probs, num_samples=1)
    return int(sampled.item())


def _parse_kv_replace_mode(value: Any) -> int:
    try:
        mode = int(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"kv_replace_mode must be 1/2/3, got: {value!r}") from exc
    if mode not in (1, 2, 3):
        raise ValueError(f"kv_replace_mode must be 1/2/3, got: {mode}")
    return int(mode)


def _single_diff_index(ids_a: list[int], ids_b: list[int]) -> int:
    if len(ids_a) != len(ids_b):
        raise ValueError("kv_replace_mode 1/2 requires equal tokenized prefix length.")
    diffs = [i for i, (a, b) in enumerate(zip(ids_a, ids_b)) if int(a) != int(b)]
    if len(diffs) != 1:
        raise ValueError(
            f"kv_replace_mode 1/2 requires exactly one token difference in prefix, got {len(diffs)}."
        )
    return int(diffs[0])

def _resolve_single_token_candidates(tokenizer, text: str) -> list[int]:
    raw = str(text or "").strip()
    if not raw:
        return []
    # Normalize full-width punctuation and trim quote wrappers.
    normalized = unicodedata.normalize('NFKC', raw).strip()
    candidate_texts: list[str] = [raw]
    if normalized and normalized != raw:
        candidate_texts.append(normalized)
    # Explicitly support both half/full-width question marks.
    if raw == '?' or normalized == '?':
        candidate_texts.append('\uFF1F')
    if raw == '\uFF1F' or normalized == '\uFF1F':
        candidate_texts.append('?')
    # Deduplicate while preserving order.
    dedup_texts: list[str] = []
    seen_texts: set[str] = set()
    for txt in candidate_texts:
        if not txt or txt in seen_texts:
            continue
        seen_texts.add(txt)
        dedup_texts.append(txt)
    # Try both forms because many BPE vocabularies encode in-context words
    # as leading-space tokens (e.g. " book"), while standalone input may be "book".
    candidates: list[list[int]] = []
    for txt in dedup_texts:
        ids_plain = [int(x) for x in (tokenizer(txt, add_special_tokens=False).get("input_ids") or [])]
        ids_space = [int(x) for x in (tokenizer(f" {txt}", add_special_tokens=False).get("input_ids") or [])]
        if ids_plain:
            candidates.append(ids_plain)
        if ids_space:
            candidates.append(ids_space)
    out: list[int] = []
    seen: set[int] = set()
    for seq in candidates:
        if len(seq) != 1:
            continue
        tid = int(seq[0])
        if tid in seen:
            continue
        seen.add(tid)
        out.append(tid)
    return out


def _resolve_token_id_sequences(tokenizer, text: str) -> list[list[int]]:
    raw = str(text or "").strip()
    if not raw:
        return []
    seq_plain = [int(x) for x in (tokenizer(raw, add_special_tokens=False).get("input_ids") or [])]
    seq_space = [int(x) for x in (tokenizer(f" {raw}", add_special_tokens=False).get("input_ids") or [])]
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for seq in (seq_plain, seq_space):
        if not seq:
            continue
        key = tuple(int(x) for x in seq)
        if key in seen:
            continue
        seen.add(key)
        out.append([int(x) for x in seq])
    return out


def _parse_ignore_token_id_list(tokenizer, text: str) -> tuple[list[str], list[int]]:
    raw = str(text or "").strip()
    if not raw:
        return [], []
    parts = [p for p in re.split(r"[\s,]+", raw) if p]
    token_ids: list[int] = []
    seen: set[int] = set()
    for part in parts:
        token_id_from_hash: int | None = None
        if str(part).startswith("#"):
            id_text = str(part)[1:].strip()
            if not id_text or not id_text.isdigit():
                raise ValueError(f"ignore token id must be '#<non-negative-int>', got: {part!r}")
            token_id_from_hash = int(id_text)
        if token_id_from_hash is not None:
            if token_id_from_hash not in seen:
                seen.add(token_id_from_hash)
                token_ids.append(token_id_from_hash)
            continue

        candidates = _resolve_single_token_candidates(tokenizer, part)
        if not candidates:
            raise ValueError(
                f"ignore token must map to one token (or one leading-space token), or use '#<token_id>': {part!r}"
            )
        for tid in candidates:
            tid_int = int(tid)
            if tid_int in seen:
                continue
            seen.add(tid_int)
            token_ids.append(tid_int)
    return parts, token_ids


def _prepare_replaced_cache_mode_1(
    *,
    model,
    device,
    prefix_ids: list[int],
    prefix_ids_replaced: list[int],
    replacement_is_zero: bool = False,
    target_token_index_override: int | None = None,
) -> tuple[Any, int]:
    target_token_index = (
        int(target_token_index_override)
        if target_token_index_override is not None
        else _single_diff_index(prefix_ids, prefix_ids_replaced)
    )
    target_token_id = (
        int(prefix_ids[int(target_token_index)])
        if bool(replacement_is_zero)
        else int(prefix_ids_replaced[int(target_token_index)])
    )
    replaced_target_prefix = prefix_ids[:int(target_token_index)] + [int(target_token_id)]
    replaced_target_tensor = torch.tensor([replaced_target_prefix], dtype=torch.long, device=device)
    replaced_target_prefill = model(
        input_ids=replaced_target_tensor,
        use_cache=True,
        return_dict=True,
    )
    out_cache = replaced_target_prefill.past_key_values
    if bool(replacement_is_zero):
        out_cache = _zero_token_state_in_cache(copy.deepcopy(out_cache), int(target_token_index))
    return out_cache, int(target_token_index)


def _prepare_replaced_cache_mode_2(
    *,
    model,
    device,
    tokenizer,
    prefix_ids: list[int],
    prefix_ids_replaced: list[int],
    replace_layer_indices_zero_based: set[int],
    replace_k: bool,
    replacement_is_zero: bool = False,
    target_token_index_override: int | None = None,
) -> tuple[Any, int, list[dict[str, Any]]]:
    target_token_index = (
        int(target_token_index_override)
        if target_token_index_override is not None
        else _single_diff_index(prefix_ids, prefix_ids_replaced)
    )
    input_token_top_logits: list[dict[str, Any]] = []
    prefix_before_target = prefix_ids[:int(target_token_index)]
    cache_before = None
    if prefix_before_target:
        before_tensor = torch.tensor([prefix_before_target], dtype=torch.long, device=device)
        out_before = model(
            input_ids=before_tensor,
            use_cache=True,
            return_dict=True,
        )
        cache_before = out_before.past_key_values
        input_token_top_logits.extend(
            _rows_from_prefill_logits(
                tokenizer=tokenizer,
                token_ids=prefix_before_target,
                logits_3d=getattr(out_before, "logits", None),
                start_step=0,
                top_k=15,
            )
        )

    original_target_token = int(prefix_ids[int(target_token_index)])
    replaced_target_token = (
        int(prefix_ids[int(target_token_index)])
        if bool(replacement_is_zero)
        else int(prefix_ids_replaced[int(target_token_index)])
    )
    # DynamicCache can be mutated in-place by model forward; use isolated cache
    # copies for original/replaced target-token passes to keep token position aligned.
    past_for_original = copy.deepcopy(cache_before) if cache_before is not None else None
    past_for_replaced = copy.deepcopy(cache_before) if cache_before is not None else None
    out_target_original = model(
        input_ids=torch.tensor([[original_target_token]], dtype=torch.long, device=device),
        past_key_values=past_for_original,
        use_cache=True,
        return_dict=True,
    )
    out_target_replaced = model(
        input_ids=torch.tensor([[replaced_target_token]], dtype=torch.long, device=device),
        past_key_values=past_for_replaced,
        use_cache=True,
        return_dict=True,
    )
    if bool(replacement_is_zero):
        out_target_replaced.past_key_values = _zero_token_state_in_cache(
            copy.deepcopy(out_target_replaced.past_key_values),
            int(target_token_index),
        )
    input_token_top_logits.append(
        _build_input_token_logits_row(
            tokenizer=tokenizer,
            step_index=int(len(input_token_top_logits)),
            token_id=int(original_target_token),
            logits=out_target_original.logits[:, -1, :],
            top_k=15,
        )
    )
    # mode2 core: at the target step, each selected layer takes replaced stream KV
    # while unselected layers keep original stream KV.
    cache = _merge_past_key_values(
        original=copy.deepcopy(out_target_original.past_key_values),
        replaced=out_target_replaced.past_key_values,
        replace_layer_indices_zero_based=replace_layer_indices_zero_based,
        replace_k=bool(replace_k),
        kv_replace_mode=3,
        target_token_index=int(target_token_index),
    )
    for tok_id in prefix_ids[int(target_token_index) + 1 :]:
        out_step = model(
            input_ids=torch.tensor([[int(tok_id)]], dtype=torch.long, device=device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = out_step.past_key_values
        input_token_top_logits.append(
            _build_input_token_logits_row(
                tokenizer=tokenizer,
                step_index=int(len(input_token_top_logits)),
                token_id=int(tok_id),
                logits=out_step.logits[:, -1, :],
                top_k=15,
            )
        )
    return cache, int(target_token_index), input_token_top_logits


def _prepare_replaced_cache_mode_3(
    *,
    model,
    replaced_prefix_tensor: torch.Tensor,
) -> tuple[Any, None, Any]:
    replaced_prefill = model(
        input_ids=replaced_prefix_tensor,
        use_cache=True,
        return_dict=True,
    )
    return replaced_prefill.past_key_values, None, getattr(replaced_prefill, "logits", None)


def _prepare_replaced_cache_ignore_mode_2(
    *,
    model,
    device,
    tokenizer,
    prefix_ids: list[int],
    prefix_ids_replaced: list[int],
    replace_layer_indices_zero_based: set[int],
    replace_k: bool,
    ignore_token_ids: list[int],
    replacement_is_zero: bool = False,
    target_token_index_override: int | None = None,
    ) -> tuple[Any, int, list[dict[str, Any]]]:
    # 1) Run the normal stream once to build baseline cache kv_cache0.
    if not prefix_ids:
        raise ValueError("ignore_mode2 requires non-empty prefix tokens.")
    out_normal = model(
        input_ids=torch.tensor([prefix_ids], dtype=torch.long, device=device),
        use_cache=True,
        return_dict=True,
    )
    kv_cache0 = out_normal.past_key_values
    input_token_top_logits: list[dict[str, Any]] = []

    # 2) Build kv_cache_replaced at target token by layer-wise mixing.
    target_token_index = (
        int(target_token_index_override)
        if target_token_index_override is not None
        else _single_diff_index(prefix_ids, prefix_ids_replaced)
    )
    prefix_before_target = prefix_ids[:int(target_token_index)]
    cache_before = None

    def _truncate_cache_to_prefix_len(cache_obj: Any, seq_len: int) -> Any:
        keep = max(0, int(seq_len))
        layers_obj = getattr(cache_obj, "layers", None)
        if isinstance(layers_obj, list):
            for layer in layers_obj:
                if hasattr(layer, "values") and torch.is_tensor(layer.values):
                    layer.values = layer.values[:, :, :keep, :]
                if hasattr(layer, "keys") and torch.is_tensor(layer.keys):
                    layer.keys = layer.keys[:, :, :keep, :]
            return cache_obj
        legacy_layers = _as_layer_sequence(cache_obj)
        sliced: list[Any] = []
        for item in legacy_layers:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and torch.is_tensor(item[1]):
                k = item[0]
                v = item[1]
                k2 = k[:, :, :keep, :] if torch.is_tensor(k) else k
                v2 = v[:, :, :keep, :]
                sliced.append((k2, v2))
            else:
                sliced.append(item)
        return tuple(sliced)

    if prefix_before_target:
        # Do not rerun prefix: directly copy/truncate from kv_cache0.
        cache_before = _truncate_cache_to_prefix_len(copy.deepcopy(kv_cache0), int(target_token_index))
        input_token_top_logits.extend(
            _rows_from_prefill_logits(
                tokenizer=tokenizer,
                token_ids=prefix_before_target,
                logits_3d=getattr(out_normal, "logits", None),
                start_step=0,
                top_k=15,
            )
        )

    original_target_token = int(prefix_ids[int(target_token_index)])
    replaced_target_token = (
        int(prefix_ids[int(target_token_index)])
        if bool(replacement_is_zero)
        else int(prefix_ids_replaced[int(target_token_index)])
    )
    out_target_original = _truncate_cache_to_prefix_len(
        copy.deepcopy(kv_cache0),
        int(target_token_index) + 1,
    )
    out_target_replaced = model(
        input_ids=torch.tensor([[replaced_target_token]], dtype=torch.long, device=device),
        past_key_values=(copy.deepcopy(cache_before) if cache_before is not None else None),
        use_cache=True,
        return_dict=True,
    )
    if bool(replacement_is_zero):
        out_target_replaced.past_key_values = _zero_token_state_in_cache(
            copy.deepcopy(out_target_replaced.past_key_values),
            int(target_token_index),
        )
    input_token_top_logits.append(
        _build_input_token_logits_row(
            tokenizer=tokenizer,
            step_index=int(len(input_token_top_logits)),
            token_id=int(original_target_token),
            logits=out_target_replaced.logits[:, -1, :],
            top_k=15,
        )
    )

    # For the target token: replaced layers use replacement-token KV,
    # non-replaced layers keep original-token KV.
    merged_target_cache = _merge_past_key_values(
        original=copy.deepcopy(out_target_original),
        replaced=out_target_replaced.past_key_values,
        replace_layer_indices_zero_based=replace_layer_indices_zero_based,
        replace_k=bool(replace_k),
        kv_replace_mode=1,
        target_token_index=int(target_token_index),
    )

    # Copy one absolute token slot from source cache into target cache (all layers, K+V).
    def _copy_token_from_source(*, target_cache: Any, source_cache: Any, token_pos: int) -> Any:
        pos = int(token_pos)
        target_layers_obj = getattr(target_cache, "layers", None)
        source_layers_obj = getattr(source_cache, "layers", None)
        if isinstance(target_layers_obj, list) and isinstance(source_layers_obj, list):
            if len(target_layers_obj) != len(source_layers_obj):
                raise ValueError(
                    f"Cannot copy cache layers with different counts: {len(target_layers_obj)} vs {len(source_layers_obj)}"
                )
            for i in range(len(target_layers_obj)):
                target_layer = target_layers_obj[i]
                source_layer = source_layers_obj[i]
                if hasattr(target_layer, "values") and hasattr(source_layer, "values"):
                    tgt_v = target_layer.values
                    src_v = source_layer.values
                    tgt_seq = int(getattr(tgt_v, "shape", [0, 0, 0, 0])[2])
                    src_seq = int(getattr(src_v, "shape", [0, 0, 0, 0])[2])
                    if src_seq <= pos:
                        raise ValueError(f"target token index out of range for source V cache: idx={pos}")
                    if tgt_seq <= pos:
                        target_layer.values = torch.cat([tgt_v, src_v[:, :, tgt_seq : pos + 1, :]], dim=2)
                    target_layer.values[:, :, pos, :] = src_v[:, :, pos, :]
                    if hasattr(target_layer, "keys") and hasattr(source_layer, "keys"):
                        tgt_k = target_layer.keys
                        src_k = source_layer.keys
                        tgt_k_seq = int(getattr(tgt_k, "shape", [0, 0, 0, 0])[2])
                        src_k_seq = int(getattr(src_k, "shape", [0, 0, 0, 0])[2])
                        if src_k_seq <= pos:
                            raise ValueError(f"target token index out of range for source K cache: idx={pos}")
                        if tgt_k_seq <= pos:
                            target_layer.keys = torch.cat([tgt_k, src_k[:, :, tgt_k_seq : pos + 1, :]], dim=2)
                        target_layer.keys[:, :, pos, :] = src_k[:, :, pos, :]
                else:
                    target_layers_obj[i] = source_layer
            return target_cache

        target_layers = _as_layer_sequence(target_cache)
        source_layers = _as_layer_sequence(source_cache)
        if len(target_layers) != len(source_layers):
            raise ValueError(
                f"Cannot copy caches with different layer counts: {len(target_layers)} vs {len(source_layers)}"
            )
        out_layers: list[Any] = []
        for target_item, source_item in zip(target_layers, source_layers):
            if isinstance(target_item, (list, tuple)) and isinstance(source_item, (list, tuple)):
                if len(target_item) >= 2 and len(source_item) >= 2:
                    tgt_k, tgt_v = target_item[0], target_item[1]
                    src_k, src_v = source_item[0], source_item[1]
                    if torch.is_tensor(tgt_v) and torch.is_tensor(src_v):
                        merged_v = tgt_v.clone()
                        tgt_seq = int(merged_v.shape[2])
                        src_seq = int(src_v.shape[2])
                        if src_seq <= pos:
                            raise ValueError(f"target token index out of range for source V cache: idx={pos}")
                        if tgt_seq <= pos:
                            merged_v = torch.cat([merged_v, src_v[:, :, tgt_seq : pos + 1, :]], dim=2)
                        merged_v[:, :, pos, :] = src_v[:, :, pos, :]
                        merged_k = tgt_k
                        if torch.is_tensor(tgt_k) and torch.is_tensor(src_k):
                            merged_k = tgt_k.clone()
                            tgt_k_seq = int(merged_k.shape[2])
                            src_k_seq = int(src_k.shape[2])
                            if src_k_seq <= pos:
                                raise ValueError(f"target token index out of range for source K cache: idx={pos}")
                            if tgt_k_seq <= pos:
                                merged_k = torch.cat([merged_k, src_k[:, :, tgt_k_seq : pos + 1, :]], dim=2)
                            merged_k[:, :, pos, :] = src_k[:, :, pos, :]
                        out_layers.append((merged_k, merged_v))
                        continue
            out_layers.append(source_item)
        return tuple(out_layers)

    # Build kv_cache_replaced in two steps:
    # 1) create merged target-token KV;
    # 2) append/copy only this token onto prefix cache.
    base_prefix_cache = (
        copy.deepcopy(cache_before)
        if cache_before is not None
        else _truncate_cache_to_prefix_len(copy.deepcopy(kv_cache0), 0)
    )
    kv_cache_replaced = _copy_token_from_source(
        target_cache=base_prefix_cache,
        source_cache=merged_target_cache,
        token_pos=int(target_token_index),
    )
   # kv_cache0=_copy_token_from_source(
   #     target_cache=kv_cache0,
   #     source_cache=merged_target_cache,
   #     token_pos=int(target_token_index),
   # )

    # 3) From token right after target:
    #    - if token is in ignore list: directly copy this token from kv_cache0
    #    - otherwise: run directly on previous kv_cache_replaced
    ignore_id_set = {int(x) for x in (ignore_token_ids or [])}
    for abs_pos in range(int(target_token_index) + 1, len(prefix_ids)):
        tok_id = int(prefix_ids[abs_pos])

        if tok_id in ignore_id_set:
            print(
                f"[ignore-mode2] ignored token pos={int(abs_pos)} id={int(tok_id)} "
                f"text={_safe_decode_token(tokenizer, int(tok_id))!r}",
                file=sys.stderr,
                flush=True,
            )
            kv_cache_replaced = _copy_token_from_source(
                target_cache=kv_cache_replaced,
                source_cache=kv_cache0,
                token_pos=int(abs_pos),
            )
            logits_row = None
            normal_logits = getattr(out_normal, "logits", None)
            if torch.is_tensor(normal_logits) and normal_logits.ndim == 3 and int(abs_pos) < int(normal_logits.shape[1]):
                logits_row = normal_logits[:, int(abs_pos), :]
            if torch.is_tensor(logits_row):
                input_token_top_logits.append(
                    _build_input_token_logits_row(
                        tokenizer=tokenizer,
                        step_index=int(len(input_token_top_logits)),
                        token_id=int(tok_id),
                        logits=logits_row,
                        top_k=15,
                    )
                )
            continue

        # Non-ignore token: run one step from previous kv_cache_replaced context.
        out_step_replaced = model(
            input_ids=torch.tensor([[tok_id]], dtype=torch.long, device=device),
            past_key_values=kv_cache_replaced,
            use_cache=True,
            return_dict=True,
        )
        kv_cache_replaced = out_step_replaced.past_key_values
        input_token_top_logits.append(
            _build_input_token_logits_row(
                tokenizer=tokenizer,
                step_index=int(len(input_token_top_logits)),
                token_id=int(tok_id),
                logits=out_step_replaced.logits[:, -1, :],
                top_k=15,
            )
        )
    return kv_cache_replaced, int(target_token_index), input_token_top_logits   
    #return kv_cache0, int(target_token_index), input_token_top_logits   


def _prepare_replaced_cache_ignore_mode_3(
    *,
    model,
    device,
    tokenizer,
    prefix_ids: list[int],
    prefix_ids_replaced: list[int],
    replace_layer_indices_zero_based: set[int],
    replace_k: bool,
    ignore_token_ids: list[int],
    replacement_is_zero: bool = False,
    target_token_index_override: int | None = None,
) -> tuple[Any, int, Any]:
    # Copy mode2 setup skeleton: keep baseline kv_cache0 path unchanged.
    if not prefix_ids:
        raise ValueError("ignore_mode3 requires non-empty prefix tokens.")
    out_normal = model(
        input_ids=torch.tensor([prefix_ids], dtype=torch.long, device=device),
        use_cache=True,
        return_dict=True,
    )
    kv_cache0 = out_normal.past_key_values

    # Replace target word and run full replaced prefix to get kv_cache1.
    target_token_index = (
        int(target_token_index_override)
        if target_token_index_override is not None
        else _single_diff_index(prefix_ids, prefix_ids_replaced)
    )
    out_replaced_full = model(
        input_ids=torch.tensor([prefix_ids_replaced], dtype=torch.long, device=device),
        use_cache=True,
        return_dict=True,
    )
    kv_cache1 = out_replaced_full.past_key_values
    if bool(replacement_is_zero):
        kv_cache1 = _zero_token_state_in_cache(copy.deepcopy(kv_cache1), int(target_token_index))
    replaced_logits = getattr(out_replaced_full, "logits", None)

    # Build kv_cache_replaced token-by-token from kv_cache0 and kv_cache1.
    kv_cache_replaced = copy.deepcopy(kv_cache0)
    ignore_id_set = {int(x) for x in (ignore_token_ids or [])}

    layers_obj = getattr(kv_cache_replaced, "layers", None)
    if isinstance(layers_obj, list):
        all_layers = set(range(len(layers_obj)))
    else:
        all_layers = set(range(len(_as_layer_sequence(kv_cache_replaced))))

    for pos, tok_id in enumerate(prefix_ids):
        # Step A: always copy this token slot from kv_cache0.
        kv_cache_replaced = _merge_past_key_values(
            original=kv_cache_replaced,
            replaced=kv_cache0,
            replace_layer_indices_zero_based=all_layers,
            replace_k=True,
            kv_replace_mode=1,
            target_token_index=int(pos),
        )

        # Step B: if token is ignored, skip replacement for this token.
        if int(tok_id) in ignore_id_set:
            print(
                f"[ignore-mode3] ignored token pos={int(pos)} id={int(tok_id)} "
                f"text={_safe_decode_token(tokenizer, int(tok_id))!r}",
                file=sys.stderr,
                flush=True,
            )
            continue

        # Step C: for non-ignore token, only selected layers use kv_cache1.
        kv_cache_replaced = _merge_past_key_values(
            original=kv_cache_replaced,
            replaced=kv_cache1,
            replace_layer_indices_zero_based=replace_layer_indices_zero_based,
            replace_k=bool(replace_k),
            kv_replace_mode=1,
            target_token_index=int(pos),
        )

    return kv_cache_replaced, int(target_token_index), replaced_logits



def _apply_ignore_token_mode_3(
    *,
    mixed_past: Any,
    original_past: Any,
    ignore_positions: list[int],
    replace_layer_indices_zero_based: set[int],
    replace_k: bool,
) -> Any:
    """Restore ignored token positions from original cache on top of a mixed mode3 cache."""
    out = mixed_past
    for pos in ignore_positions or []:
        out = _merge_past_key_values(
            original=out,
            replaced=original_past,
            replace_layer_indices_zero_based=replace_layer_indices_zero_based,
            replace_k=bool(replace_k),
            kv_replace_mode=1,
            target_token_index=int(pos),
        )
    return out

def _mix_assistant_cache_mode_1(
    *,
    original_past: Any,
    replaced_past: Any,
    replace_layer_indices_zero_based: set[int],
    replace_k: bool,
    target_token_index: int,
) -> tuple[Any, Any]:
    # Mode1: mix only target token KV at assistant step.
    if replaced_past is None:
        raise ValueError("mode1 requires replaced_past")
    mixed = _merge_past_key_values(
        original=copy.deepcopy(original_past),
        replaced=replaced_past,
        replace_layer_indices_zero_based=replace_layer_indices_zero_based,
        replace_k=bool(replace_k),
        kv_replace_mode=1,
        target_token_index=int(target_token_index),
    )
    return mixed, copy.deepcopy(original_past)


def _mix_assistant_cache_mode_2(
    *,
    original_past: Any,
    replaced_past: Any,
    replace_layer_indices_zero_based: set[int],
    replace_k: bool,
    target_token_index: int,
) -> tuple[Any, Any]:
    # Mode2 cache is already prepared before assistant boundary.
    _ = replaced_past
    _ = replace_layer_indices_zero_based
    _ = bool(replace_k)
    _ = int(target_token_index)
    return original_past, copy.deepcopy(original_past)


def _mix_assistant_cache_mode_3(
    *,
    original_past: Any,
    replaced_past: Any,
    replace_layer_indices_zero_based: set[int],
    replace_k: bool,
) -> tuple[Any, Any]:
    # Mode3: full-layer replacement behavior at assistant step.
    if replaced_past is None:
        raise ValueError("mode3 requires replaced_past")
    mixed = _merge_past_key_values(
        original=copy.deepcopy(original_past),
        replaced=replaced_past,
        replace_layer_indices_zero_based=replace_layer_indices_zero_based,
        replace_k=bool(replace_k),
        kv_replace_mode=3,
        target_token_index=None,
    )
    return mixed, copy.deepcopy(original_past)


def _truncate_cache_to_prefix_len(cache_obj: Any, seq_len: int) -> Any:
    keep = max(0, int(seq_len))
    layers_obj = getattr(cache_obj, "layers", None)
    if isinstance(layers_obj, list):
        for layer in layers_obj:
            if hasattr(layer, "values") and torch.is_tensor(layer.values):
                layer.values = layer.values[:, :, :keep, :]
            if hasattr(layer, "keys") and torch.is_tensor(layer.keys):
                layer.keys = layer.keys[:, :, :keep, :]
        return cache_obj
    legacy_layers = _as_layer_sequence(cache_obj)
    sliced: list[Any] = []
    for item in legacy_layers:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and torch.is_tensor(item[1]):
            k = item[0]
            v = item[1]
            k2 = k[:, :, :keep, :] if torch.is_tensor(k) else k
            v2 = v[:, :, :keep, :]
            sliced.append((k2, v2))
        else:
            sliced.append(item)
    return tuple(sliced)


def _zero_token_state_in_cache(cache_obj: Any, token_index: int) -> Any:
    pos = int(token_index)
    layers_obj = getattr(cache_obj, "layers", None)
    if isinstance(layers_obj, list):
        for layer in layers_obj:
            if hasattr(layer, "values") and torch.is_tensor(layer.values):
                if int(layer.values.shape[2]) > pos:
                    layer.values[:, :, pos, :] = 0
            if hasattr(layer, "keys") and torch.is_tensor(layer.keys):
                if int(layer.keys.shape[2]) > pos:
                    layer.keys[:, :, pos, :] = 0
        return cache_obj
    legacy_layers = _as_layer_sequence(cache_obj)
    out_layers: list[Any] = []
    for item in legacy_layers:
        if isinstance(item, (list, tuple)) and len(item) >= 2 and torch.is_tensor(item[1]):
            k = item[0]
            v = item[1].clone()
            if int(v.shape[2]) > pos:
                v[:, :, pos, :] = 0
            if torch.is_tensor(k):
                k2 = k.clone()
                if int(k2.shape[2]) > pos:
                    k2[:, :, pos, :] = 0
            else:
                k2 = k
            out_layers.append((k2, v))
        else:
            out_layers.append(item)
    return tuple(out_layers)


def _find_single_target_token_index(
    tokenizer,
    *,
    prefix_ids: list[int],
    target_word: str,
    search_end_exclusive: int | None = None,
) -> int:
    candidate_ids = _resolve_single_token_candidates(tokenizer, target_word)
    if not candidate_ids:
        raise ValueError(f"target_word cannot be resolved as single token candidate: {target_word!r}")
    end = int(len(prefix_ids) if search_end_exclusive is None else max(0, min(search_end_exclusive, len(prefix_ids))))
    cset = {int(x) for x in candidate_ids}
    hits = [i for i in range(end) if int(prefix_ids[i]) in cset]
    if len(hits) != 1:
        raise ValueError(
            f"cannot locate unique target token index in prefix for {target_word!r}; "
            f"candidate_ids={sorted(cset)}, hits={hits}"
        )
    return int(hits[0])


def _safe_decode_token(tokenizer, token_id: int) -> str:
    try:
        text = tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except Exception:  # noqa: BLE001
        text = str(int(token_id))
    return str(text)


def _top_logits_rows_from_tensor(
    *,
    tokenizer,
    logits: torch.Tensor,
    top_k: int = 100,
) -> list[dict[str, Any]]:
    if not torch.is_tensor(logits):
        return []
    if logits.ndim == 2:
        row = logits[0]
    elif logits.ndim == 1:
        row = logits
    else:
        row = logits.reshape(-1)
    k = max(1, min(int(top_k), int(row.shape[-1])))
    values, indices = torch.topk(row, k=k)
    out: list[dict[str, Any]] = []
    for i in range(k):
        tid = int(indices[i].item())
        tok_text = _safe_decode_token(tokenizer, tid)
        out.append(
            {
                "token_id": tid,
                "token": tok_text,
                "text": tok_text,
                "logit": float(values[i].item()),
            }
        )
    return out


def _build_input_token_logits_row(
    *,
    tokenizer,
    step_index: int,
    token_id: int,
    logits: torch.Tensor,
    top_k: int = 15,
) -> dict[str, Any]:
    row_text = _safe_decode_token(tokenizer, int(token_id))
    return {
        "step": int(step_index),
        "token_id": int(token_id),
        "token": row_text,
        "text": row_text,
        "top_logits": _top_logits_rows_from_tensor(tokenizer=tokenizer, logits=logits, top_k=int(top_k)),
    }


def _rows_from_prefill_logits(
    *,
    tokenizer,
    token_ids: list[int],
    logits_3d: Any,
    start_step: int = 0,
    top_k: int = 15,
) -> list[dict[str, Any]]:
    if not torch.is_tensor(logits_3d) or logits_3d.ndim != 3:
        return []
    seq_len = min(int(logits_3d.shape[1]), int(len(token_ids)))
    rows: list[dict[str, Any]] = []
    for pos in range(seq_len):
        rows.append(
            _build_input_token_logits_row(
                tokenizer=tokenizer,
                step_index=int(start_step + pos),
                token_id=int(token_ids[pos]),
                logits=logits_3d[:, pos, :],
                top_k=int(top_k),
            )
        )
    return rows


def _find_last_subseq_end_pos(haystack: list[int], needle: list[int], *, stop_before: int | None = None) -> int | None:
    if not haystack or not needle:
        return None
    n = int(len(needle))
    end_limit = int(len(haystack)) if stop_before is None else max(0, int(stop_before))
    if n > end_limit:
        return None
    for start in range(end_limit - n, -1, -1):
        if haystack[start:start + n] == needle:
            return int(start + n - 1)
    return None


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


def _build_assistant_target_heatmaps(
    *,
    tokenizer,
    input_ids_with_assistant: list[int],
    target_word: str,
    config: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta: dict[str, Any] = {}
    try:
        assistant_pos = _locate_assistant_marker_position(tokenizer, input_ids_with_assistant)
        if assistant_pos is None:
            meta["assistant_target_heatmap_error"] = "assistant_marker_not_found"
            return [], meta
        target_text = str(target_word or "")
        target_ids_primary = [int(x) for x in (tokenizer(target_text, add_special_tokens=False).get("input_ids") or [])]
        target_ids_space = [int(x) for x in (tokenizer(f" {target_text}", add_special_tokens=False).get("input_ids") or [])]
        candidate_target_id_seqs: list[list[int]] = []
        if target_ids_primary:
            candidate_target_id_seqs.append(target_ids_primary)
        if target_ids_space:
            candidate_target_id_seqs.append(target_ids_space)
        # Deduplicate
        uniq: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for seq in candidate_target_id_seqs:
            key = tuple(int(x) for x in seq)
            if key in seen:
                continue
            seen.add(key)
            uniq.append([int(x) for x in seq])
        candidate_target_id_seqs = uniq
        if not candidate_target_id_seqs:
            meta["assistant_target_heatmap_error"] = "target_tokenize_failed"
            return [], meta
        target_pos: int | None = None
        for cand in candidate_target_id_seqs:
            hit = _find_last_subseq_end_pos(input_ids_with_assistant, cand, stop_before=int(assistant_pos))
            if hit is None:
                continue
            if target_pos is None or int(hit) > int(target_pos):
                target_pos = int(hit)
        if target_pos is None:
            meta["assistant_target_heatmap_error"] = "target_not_found_before_assistant"
            return [], meta
        if config is None:
            meta["assistant_target_heatmap_error"] = "missing_config_for_attention_probe"
            return [], meta

        metrics = fetch_head_attention_metrics_for_input_ids(
            input_ids=[int(x) for x in input_ids_with_assistant],
            query_positions={"assistant": int(assistant_pos)},
            config=config,
        )
        if not isinstance(metrics, dict) or not metrics.get("ok"):
            meta["assistant_target_heatmap_error"] = str((metrics or {}).get("reason") or "attention_probe_failed")
            return [], meta
        query_qk_to_keys = metrics.get("query_qk_to_keys") or {}
        query_attn_to_keys = metrics.get("query_attn_to_keys") or {}
        qk_rows = query_qk_to_keys.get("assistant") or []
        attn_rows = query_attn_to_keys.get("assistant") or []
        qk_target = _extract_key_column_matrix(qk_rows, int(target_pos))
        attn_target = _extract_key_column_matrix(attn_rows, int(target_pos))
        tokens = [str(t) for t in (metrics.get("input_tokens") or [])]
        assistant_token = tokens[int(assistant_pos)] if 0 <= int(assistant_pos) < len(tokens) else "assistant_symbol"
        target_token = tokens[int(target_pos)] if 0 <= int(target_pos) < len(tokens) else str(target_word or "")
        heatmaps = [
            {
                "key": "assistant_to_target_qk",
                "title": f"QK Score (pre-softmax): assistant[{assistant_pos}]({assistant_token}) -> target[{target_pos}]({target_token})",
                "matrix": qk_target,
                "hover_x_label": "head",
                "hover_y_label": "layer",
                "hover_x_offset": 1,
                "hover_y_offset": 1,
            },
            {
                "key": "assistant_to_target_attn",
                "title": f"Attention Probability: assistant[{assistant_pos}]({assistant_token}) -> target[{target_pos}]({target_token})",
                "matrix": attn_target,
                "hover_x_label": "head",
                "hover_y_label": "layer",
                "hover_x_offset": 1,
                "hover_y_offset": 1,
            },
        ]
        meta["assistant_query_pos"] = int(assistant_pos)
        meta["target_key_pos"] = int(target_pos)
        meta["assistant_target_heatmap_error"] = None
        return heatmaps, meta
    except Exception as exc:  # noqa: BLE001
        meta["assistant_target_heatmap_error"] = str(exc)
        return [], meta


def _maybe_apply_layer_shortcut_logits(
    *,
    model,
    tokenizer,
    config: dict[str, Any] | None,
    input_ids: torch.Tensor,
    step_hidden_states: Any,
    shortcut_start_layer_zero_based: int,
    shortcut_target_layer_zero_based: int,
) -> torch.Tensor:
    if config is None:
        raise ValueError("Layer shortcut requires config.")
    if not isinstance(step_hidden_states, (list, tuple)):
        raise ValueError("Layer shortcut requires output_hidden_states from model forward.")
    source_idx = int(shortcut_start_layer_zero_based) + 1
    if source_idx < 0 or source_idx >= len(step_hidden_states):
        raise ValueError(
            f"Layer shortcut source row out of range: source={int(shortcut_start_layer_zero_based)}, "
            f"available_rows={int(len(step_hidden_states))}"
        )
    source_hidden = step_hidden_states[source_idx]
    bundle = SimpleNamespace(model=model, tokenizer=tokenizer)
    transit, transit_error = run_starting_from_middle_layer_probe(
        word=None,
        config=config,
        start_layer_idx=int(shortcut_target_layer_zero_based - 1),
        hidden_state=source_hidden,
        input_ids_override=input_ids,
        bundle_override=bundle,
    )
    if transit is None:
        raise ValueError(f"layer_shortcut_failed: {str(transit_error or 'unknown')}")
    transit_result = transit.get("result") if isinstance(transit, dict) else None
    logits = None
    if isinstance(transit_result, dict):
        logits = transit_result.get("logits")
        if logits is None:
            outputs = transit_result.get("outputs")
            logits = getattr(outputs, "logits", None) if outputs is not None else None
    if not torch.is_tensor(logits):
        raise ValueError("layer_shortcut_failed: missing logits")
    if logits.ndim == 3:
        return logits[:, -1, :]
    if logits.ndim == 2:
        return logits
    raise ValueError(f"layer_shortcut_failed: unexpected logits ndim={int(logits.ndim)}")


def run_study(
    *,
    model,
    tokenizer,
    messages: list[dict[str, str]],
    target_word: str,
    replacement_word: str,
    replace_layers: str,
    kv_replace_mode: int = 3,
    enable_ignore_replacement_token: bool = False,
    ignore_replacement_token: str = "",
    replace_k: bool = True,
    enable_layer_shortcut: bool = False,
    shortcut_start_layer: int = 24,
    shortcut_target_layer: int = 31,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    eos_token_id: int | None,
    include_assistant_marker: bool = True,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = str(target_word or "").strip()
    replacement = str(replacement_word or "").strip()
    if not target:
        raise ValueError("target_word is required.")
    replacement_is_zero = (replacement == "")
    target_ids = [int(x) for x in (tokenizer(target, add_special_tokens=False).get("input_ids") or [])]
    replacement_ids = [int(x) for x in (tokenizer(replacement, add_special_tokens=False).get("input_ids") or [])]
    if len(target_ids) != 1:
        raise ValueError(
            f"target_word must be single-token, got {len(target_ids)} tokens: {target!r}, ids={target_ids}"
        )
    if (not replacement_is_zero) and len(replacement_ids) != 1:
        raise ValueError(
            f"replacement_word must be single-token or empty, got {len(replacement_ids)} tokens: {replacement!r}, ids={replacement_ids}"
        )
    ignore_token_text = str(ignore_replacement_token or "").strip()
    ignore_enabled = bool(enable_ignore_replacement_token) or bool(ignore_token_text)
    ignore_token_parts: list[str] = []
    ignore_token_ids: list[int] = []
    if ignore_token_text:
        ignore_token_parts, ignore_token_ids = _parse_ignore_token_id_list(tokenizer, ignore_token_text)

    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("Model does not expose decoder layers.")
    layer_count = int(len(layers))
    parsed_kv_replace_mode = _parse_kv_replace_mode(kv_replace_mode)
    replace_layer_indices_zero_based, replace_layers_zero_based = _parse_replace_layers_spec(
        str(replace_layers or ""),
        layer_count=layer_count,
    )
    layer_shortcut_enabled = bool(enable_layer_shortcut)
    shortcut_start_layer_zero_based: int | None = None
    shortcut_target_layer_zero_based: int | None = None
    if layer_shortcut_enabled:
        src, tgt, _ = validate_shortcut_layers_zero_based(
            source_layer=int(shortcut_start_layer),
            target_layer=int(shortcut_target_layer),
            layer_count=layer_count,
        )
        shortcut_start_layer_zero_based = int(src)
        shortcut_target_layer_zero_based = int(tgt)

    replaced_messages_candidate, replaced_count = _replace_target_in_user_messages(
        messages,
        target=target,
        replacement=replacement,
    )
    replaced_messages = (
        [{"role": str(x.get("role") or ""), "content": str(x.get("content") or "")} for x in messages]
        if bool(replacement_is_zero)
        else replaced_messages_candidate
    )
    if replaced_count <= 0:
        raise ValueError("Target word was not found in user messages.")
    if replaced_count > 1:
        raise ValueError(
            f"target_word must appear exactly once in user messages, got {int(replaced_count)} occurrences: {target!r}"
        )

    # Unified queue build: start from last-user tokens; when assistant is enabled,
    # append assistant suffix ids, then keep downstream flow identical.
    last_user_text = ""
    last_user_text_replaced = ""
    for item in messages:
        if str(item.get("role") or "") == "user":
            last_user_text = str(item.get("content") or "")
    for item in replaced_messages:
        if str(item.get("role") or "") == "user":
            last_user_text_replaced = str(item.get("content") or "")

    full_ids = [int(x) for x in (tokenizer(last_user_text, add_special_tokens=False).get("input_ids") or [])]
    full_ids_replaced = [
        int(x) for x in (tokenizer(last_user_text_replaced, add_special_tokens=False).get("input_ids") or [])
    ]
    if len(full_ids) < 1 or len(full_ids_replaced) < 1:
        raise ValueError("Last user message must contain at least one token.")

    if bool(include_assistant_marker):
        assistant_suffix_ids = get_assistant_ids(tokenizer, messages)
        assistant_suffix_replaced_ids = get_assistant_ids(tokenizer, replaced_messages)
        if assistant_suffix_ids != assistant_suffix_replaced_ids:
            raise ValueError("Assistant suffix differs between original and replaced prompts.")
        full_ids = list(full_ids) + list(assistant_suffix_ids)
        full_ids_replaced = list(full_ids_replaced) + list(assistant_suffix_replaced_ids)
    else:
        assistant_suffix_ids = [int(full_ids[-1])]
        assistant_suffix_replaced_ids = [int(full_ids_replaced[-1])]

    target_token_index_override: int | None = None
    if bool(replacement_is_zero):
        search_end = int(len(full_ids) - len(assistant_suffix_ids)) if bool(include_assistant_marker) else int(len(full_ids))
        target_token_index_override = _find_single_target_token_index(
            tokenizer,
            prefix_ids=list(full_ids),
            target_word=target,
            search_end_exclusive=search_end,
        )
        full_ids_replaced = list(full_ids)

    if len(full_ids) != len(full_ids_replaced):
        if parsed_kv_replace_mode in (1, 2):
            raise ValueError(
                "kv_replace_mode 1/2 requires equal tokenized prompt length."
            )
        if not bool(replace_k):
            raise ValueError(
                "V-only mode requires equal prompt length."
            )

    # Assistant ids (when enabled) are already appended into full_ids above,
    # so treat the whole queue as the input sequence for downstream flow.
    prefix_ids = list(full_ids)
    prefix_ids_replaced = list(full_ids_replaced)
    if len(prefix_ids) != len(prefix_ids_replaced):
        if parsed_kv_replace_mode in (1, 2):
            raise ValueError(
                "kv_replace_mode 1/2 requires equal tokenized prefix length. "
                "Please use replacement with same tokenization length or switch to mode 3."
            )
        if not bool(replace_k):
            raise ValueError(
                "Replacement changed prompt token length before assistant marker. "
                "V-only mode requires equal prefix length; enable replace_k=true (KV mode), "
                "or use a replacement with the same tokenized length."
            )

    if bool(include_assistant_marker):
        assistant_token_id = int(resolve_assistant_token_id(tokenizer))
    else:
        assistant_token_id = int(assistant_suffix_ids[0])

    device = next(model.parameters()).device
    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    replaced_prefix_tensor = torch.tensor([prefix_ids_replaced], dtype=torch.long, device=device)

    with torch.inference_mode():
        original_prefill = model(
            input_ids=prefix_tensor,
            use_cache=True,
            return_dict=True,
        )
        original_past = original_prefill.past_key_values
        baseline_prompt_past = copy.deepcopy(original_past)
        input_token_top_logits: list[dict[str, Any]] = []
        if ignore_enabled:
            if parsed_kv_replace_mode == 1:
                replaced_past, target_token_index = _prepare_replaced_cache_mode_1(
                    model=model,
                    device=device,
                    prefix_ids=prefix_ids,
                    prefix_ids_replaced=prefix_ids_replaced,
                    replacement_is_zero=bool(replacement_is_zero),
                    target_token_index_override=target_token_index_override,
                )
                input_token_top_logits = _rows_from_prefill_logits(
                    tokenizer=tokenizer,
                    token_ids=prefix_ids,
                    logits_3d=getattr(original_prefill, "logits", None),
                    start_step=0,
                    top_k=15,
                )
            elif parsed_kv_replace_mode == 2:
                mode2_cache, target_token_index, input_token_top_logits = _prepare_replaced_cache_ignore_mode_2(
                    model=model,
                    device=device,
                    tokenizer=tokenizer,
                    prefix_ids=prefix_ids,
                    prefix_ids_replaced=prefix_ids_replaced,
                    replace_layer_indices_zero_based=replace_layer_indices_zero_based,
                    replace_k=bool(replace_k),
                    ignore_token_ids=ignore_token_ids,
                    replacement_is_zero=bool(replacement_is_zero),
                    target_token_index_override=target_token_index_override,
                )
                original_past = mode2_cache
                replaced_past = mode2_cache
            else:
                replaced_past, target_token_index, replaced_prefix_logits = _prepare_replaced_cache_ignore_mode_3(
                    model=model,
                    device=device,
                    tokenizer=tokenizer,
                    prefix_ids=prefix_ids,
                    prefix_ids_replaced=prefix_ids_replaced,
                    replace_layer_indices_zero_based=replace_layer_indices_zero_based,
                    replace_k=bool(replace_k),
                    ignore_token_ids=ignore_token_ids,
                    replacement_is_zero=bool(replacement_is_zero),
                    target_token_index_override=target_token_index_override,
                )
                use_replaced_logits = int(layer_count - 1) in replace_layer_indices_zero_based
                input_token_top_logits = _rows_from_prefill_logits(
                    tokenizer=tokenizer,
                    token_ids=prefix_ids_replaced if use_replaced_logits else prefix_ids,
                    logits_3d=(
                        replaced_prefix_logits
                        if use_replaced_logits
                        else getattr(original_prefill, "logits", None)
                    ),
                    start_step=0,
                    top_k=15,
                )
        else:
            if parsed_kv_replace_mode == 1:
                replaced_past, target_token_index = _prepare_replaced_cache_mode_1(
                    model=model,
                    device=device,
                    prefix_ids=prefix_ids,
                    prefix_ids_replaced=prefix_ids_replaced,
                    replacement_is_zero=bool(replacement_is_zero),
                    target_token_index_override=target_token_index_override,
                )
                input_token_top_logits = _rows_from_prefill_logits(
                    tokenizer=tokenizer,
                    token_ids=prefix_ids,
                    logits_3d=getattr(original_prefill, "logits", None),
                    start_step=0,
                    top_k=15,
                )
            elif parsed_kv_replace_mode == 2:
                mode2_cache, target_token_index, input_token_top_logits = _prepare_replaced_cache_mode_2(
                    model=model,
                    device=device,
                    tokenizer=tokenizer,
                    prefix_ids=prefix_ids,
                    prefix_ids_replaced=prefix_ids_replaced,
                    replace_layer_indices_zero_based=replace_layer_indices_zero_based,
                    replace_k=bool(replace_k),
                    replacement_is_zero=bool(replacement_is_zero),
                    target_token_index_override=target_token_index_override,
                )
                original_past = mode2_cache
                replaced_past = None
            else:
                replaced_past, target_token_index, replaced_prefix_logits = _prepare_replaced_cache_mode_3(
                    model=model,
                    replaced_prefix_tensor=replaced_prefix_tensor,
                )
                if bool(replacement_is_zero):
                    target_token_index = int(target_token_index_override) if target_token_index_override is not None else int(target_token_index or 0)
                    replaced_past = _zero_token_state_in_cache(copy.deepcopy(replaced_past), int(target_token_index))
                use_replaced_logits = int(layer_count - 1) in replace_layer_indices_zero_based
                input_token_top_logits = _rows_from_prefill_logits(
                    tokenizer=tokenizer,
                    token_ids=prefix_ids_replaced if use_replaced_logits else prefix_ids,
                    logits_3d=(
                        replaced_prefix_logits
                        if use_replaced_logits
                        else getattr(original_prefill, "logits", None)
                    ),
                    start_step=0,
                    top_k=15,
                )

        # Unified post-flow: still honor mode-specific cache mix before generation.
        assistant_mixed = bool(include_assistant_marker)
        if int(parsed_kv_replace_mode) == 1:
            if replaced_past is None:
                raise ValueError("Internal error: missing replaced cache before mode1 merge.")
            original_past = _merge_past_key_values(
                original=copy.deepcopy(original_past),
                replaced=replaced_past,
                replace_layer_indices_zero_based=replace_layer_indices_zero_based,
                replace_k=bool(replace_k),
                kv_replace_mode=1,
                target_token_index=int(target_token_index),
            )
        elif int(parsed_kv_replace_mode) == 3:
            if replaced_past is None:
                raise ValueError("Internal error: missing replaced cache before mode3 merge.")
            original_past = _merge_past_key_values(
                original=copy.deepcopy(original_past),
                replaced=replaced_past,
                replace_layer_indices_zero_based=replace_layer_indices_zero_based,
                replace_k=bool(replace_k),
                kv_replace_mode=3,
                target_token_index=None,
            )

        if len(prefix_ids) < 1:
            raise ValueError("Prompt must contain at least one token.")
        last_token_id = int(prefix_ids[-1])
        cache_before_last = None
        if len(prefix_ids) > 1:
            cache_before_last = _truncate_cache_to_prefix_len(copy.deepcopy(original_past), len(prefix_ids) - 1)
        out_last = model(
            input_ids=torch.tensor([[last_token_id]], dtype=torch.long, device=device),
            past_key_values=cache_before_last,
            use_cache=True,
            return_dict=True,
            output_hidden_states=bool(layer_shortcut_enabled),
        )
        original_past = out_last.past_key_values
        last_logits: torch.Tensor | None = out_last.logits[:, -1, :]
        if layer_shortcut_enabled:
            last_logits = _maybe_apply_layer_shortcut_logits(
                model=model,
                tokenizer=tokenizer,
                config=config,
                input_ids=torch.tensor([[last_token_id]], dtype=torch.long, device=device),
                step_hidden_states=getattr(out_last, "hidden_states", None),
                shortcut_start_layer_zero_based=int(shortcut_start_layer_zero_based or 0),
                shortcut_target_layer_zero_based=int(shortcut_target_layer_zero_based or 0),
            )

        generated: list[int] = []
        max_steps = int(max_new_tokens)
        if max_steps <= 0:
            top100_source = last_logits
            top_logits = _top_logits_rows_from_tensor(tokenizer=tokenizer, logits=top100_source, top_k=100)
            return {
                "status": "ok",
                "assistant_message": "",
                "generated_token_count": 0,
                "top_logits": top_logits,
                "input_token_top_logits": input_token_top_logits,
                "attention_word_replacement": {
                    "enabled": True,
                    "target_word": target,
                    "replacement_word": replacement,
                    "replace_layers": str(replace_layers or ""),
                    "replace_layers_zero_based": list(replace_layers_zero_based),
                    "kv_replace_mode": int(parsed_kv_replace_mode),
                    "ignore_enabled": bool(ignore_enabled),
                    "ignore_replacement_token": ignore_token_text,
                    "ignore_replacement_tokens": [str(x) for x in ignore_token_parts],
                    "ignore_replacement_token_ids": [int(x) for x in ignore_token_ids],
                    "replace_k": bool(replace_k),
                    "replace_mode": "kv" if bool(replace_k) else "v_only",
                    "replaced_occurrences": int(replaced_count),
                    "assistant_token_id": int(assistant_token_id),
                    "assistant_suffix_length": int(len(assistant_suffix_ids)),
                    "assistant_mixed": bool(assistant_mixed),
                    "layer_shortcut_enabled": bool(layer_shortcut_enabled),
                    "shortcut_start_layer_zero_based": (
                        int(shortcut_start_layer_zero_based) if shortcut_start_layer_zero_based is not None else None
                    ),
                    "shortcut_target_layer_zero_based": (
                        int(shortcut_target_layer_zero_based) if shortcut_target_layer_zero_based is not None else None
                    ),
                },
            }
        next_id = _sample_next_id(
            last_logits,
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
        )
        generated.append(int(next_id))
        if eos_token_id is None or int(next_id) != int(eos_token_id):
            for _ in range(max_steps - 1):
                next_input = torch.tensor([[int(next_id)]], dtype=torch.long, device=device)
                outputs = model(
                    input_ids=next_input,
                    past_key_values=original_past,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=False,
                )
                original_past = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                next_id = _sample_next_id(
                    logits,
                    do_sample=bool(do_sample),
                    temperature=float(temperature),
                    top_p=float(top_p),
                )
                generated.append(int(next_id))
                if eos_token_id is not None and int(next_id) == int(eos_token_id):
                    break

    text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    if not text:
        text = tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
    top100_source = last_logits
    top_logits = _top_logits_rows_from_tensor(tokenizer=tokenizer, logits=top100_source, top_k=100)
    assistant_target_heatmaps, heatmap_meta = _build_assistant_target_heatmaps(
        tokenizer=tokenizer,
        input_ids_with_assistant=[int(x) for x in full_ids],
        target_word=target,
        config=config,
    )
    return {
        "status": "ok",
        "assistant_message": text,
        "generated_token_count": int(len(generated)),
        "top_logits": top_logits,
        "input_token_top_logits": input_token_top_logits,
        "heatmaps": assistant_target_heatmaps,
        "attention_word_replacement": {
            "enabled": True,
            "target_word": target,
            "replacement_word": replacement,
            "replace_layers": str(replace_layers or ""),
            "replace_layers_zero_based": list(replace_layers_zero_based),
            "kv_replace_mode": int(parsed_kv_replace_mode),
            "ignore_enabled": bool(ignore_enabled),
            "ignore_replacement_token": ignore_token_text,
            "ignore_replacement_tokens": [str(x) for x in ignore_token_parts],
            "ignore_replacement_token_ids": [int(x) for x in ignore_token_ids],
            "replace_k": bool(replace_k),
            "replace_mode": "kv" if bool(replace_k) else "v_only",
            "replaced_occurrences": int(replaced_count),
            "assistant_token_id": int(assistant_token_id),
            "assistant_suffix_length": int(len(assistant_suffix_ids)),
            "assistant_mixed": bool(assistant_mixed),
            "layer_shortcut_enabled": bool(layer_shortcut_enabled),
            "shortcut_start_layer_zero_based": (
                int(shortcut_start_layer_zero_based) if shortcut_start_layer_zero_based is not None else None
            ),
            "shortcut_target_layer_zero_based": (
                int(shortcut_target_layer_zero_based) if shortcut_target_layer_zero_based is not None else None
            ),
            **heatmap_meta,
        },
    }


def register_cli(subparsers: argparse._SubParsersAction, bool_parser) -> None:
    parser = subparsers.add_parser(
        "run-chat-attention-word-replacement",
        help="Chat study: replace selected user-word attention cache from a start layer at assistant boundary, then generate.",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="Alice gave Bob a book. Who has the book? Please directly give the name",
        help="User prompt text for this one-turn chat study.",
    )
    parser.add_argument(
        "--target-word",
        type=str,
        default="Bob",
        help="Word to replace in the user prompt.",
    )
    parser.add_argument(
        "--replacement-word",
        type=str,
        default="Bill",
        help="Replacement word used to build substitute cache.",
    )
    parser.add_argument(
        "--enable-ignore-replacement-token",
        type=bool_parser,
        default=False,
        help="Whether to enable ignore-replacement routing for mode 1/2/3.",
    )
    parser.add_argument(
        "--ignore-replacement-token",
        type=str,
        default="",
        help="Ignore token text used by ignore mode routing.",
    )
    parser.add_argument(
        "--replace-layers",
        type=str,
        default="0-",
        help="Layers to replace (0-based). Examples: 0- , 0-3 , 0,3",
    )
    parser.add_argument(
        "--kv-replace-mode",
        type=int,
        default=3,
        help="KV replacement mode: 1=only target token KV, 2=target token KV + following tokens only for layers after replace_layers, 3=full replaced prompt KV.",
    )
    parser.add_argument(
        "--replace-k",
        type=bool_parser,
        default=True,
        help="Whether to replace K together with V (true/false). false means V-only replacement.",
    )
    parser.add_argument(
        "--enable-layer-shortcut",
        type=bool_parser,
        default=False,
        help="Whether to enable layer shortcut during assistant suffix and generation.",
    )
    parser.add_argument(
        "--shortcut-start-layer",
        type=int,
        default=24,
        help="Shortcut start layer index (0-based).",
    )
    parser.add_argument(
        "--shortcut-target-layer",
        type=int,
        default=31,
        help="Shortcut target layer index (0-based).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (>0 enables sampling).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p.",
    )
    parser.add_argument(
        "--include-assistant-marker",
        type=bool_parser,
        default=True,
        help="Whether to include assistant generation marker; false means process/mix on the last prompt token.",
    )


def try_execute_cli(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any] | None:
    if args.command != "run-chat-attention-word-replacement":
        return None

    try:
        api = get_runtime_api()
    except RuntimeError:
        api = start_llama_api(config)
    bundle = api.execute_model_call(RuntimeRequest(config=config, force_reload=False)).bundle
    tokenizer = bundle.tokenizer
    model = bundle.model

    safe_max_new_tokens = int(max(1, min(int(args.max_new_tokens), 1024)))
    safe_temperature = float(args.temperature)
    safe_top_p = float(args.top_p)
    do_sample = safe_temperature > 0.0
    if do_sample:
        safe_temperature = max(0.05, min(safe_temperature, 5.0))
        safe_top_p = max(0.05, min(safe_top_p, 1.0))

    result = run_study(
        model=model,
        tokenizer=tokenizer,
        messages=[{"role": "user", "content": str(args.prompt_text or "")}],
        target_word=str(args.target_word or ""),
        replacement_word=str(args.replacement_word or ""),
        enable_ignore_replacement_token=bool(args.enable_ignore_replacement_token),
        ignore_replacement_token=str(args.ignore_replacement_token or ""),
        replace_layers=str(args.replace_layers or ""),
        kv_replace_mode=int(args.kv_replace_mode),
        replace_k=bool(args.replace_k),
        enable_layer_shortcut=bool(args.enable_layer_shortcut),
        shortcut_start_layer=int(args.shortcut_start_layer),
        shortcut_target_layer=int(args.shortcut_target_layer),
        max_new_tokens=safe_max_new_tokens,
        do_sample=bool(do_sample),
        temperature=float(safe_temperature),
        top_p=float(safe_top_p),
        eos_token_id=int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
        include_assistant_marker=bool(args.include_assistant_marker),
        config=config,
    )
    if str(result.get("status") or "") != "ok":
        reason = str(result.get("error") or "chat_attention_word_replacement_failed")
        print(f"[chat_awr] error: {reason}")
        return {
            "hidden_state_heatmap": {
                "ok": False,
                "study": "chat_attention_word_replacement",
                "reason": reason,
                "summary_text": reason,
                "ui_tasks": [],
            }
        }
    assistant_message = str(result.get("assistant_message") or "").strip()
    meta = result.get("attention_word_replacement") or {}
    top_logits = result.get("top_logits") if isinstance(result.get("top_logits"), list) else []
    input_token_top_logits = (
        result.get("input_token_top_logits") if isinstance(result.get("input_token_top_logits"), list) else []
    )
    # Heatmap popup rendering is temporarily disabled by request.
    # heatmaps = result.get("heatmaps") if isinstance(result.get("heatmaps"), list) else []
    # valid_heatmaps = [
    #     hm
    #     for hm in heatmaps
    #     if isinstance(hm, dict)
    #     and isinstance(hm.get("matrix"), list)
    #     and len(hm.get("matrix") or []) > 0
    #     and isinstance((hm.get("matrix") or [None])[0], list)
    #     and len(((hm.get("matrix") or [[]])[0] or [])) > 0
    # ]
    valid_heatmaps: list[dict[str, Any]] = []
    summary_text = (
        f"Prompt: {str(args.prompt_text or '')}\n"
        f"Target -> Replacement: {str(args.target_word or '')} -> {str(args.replacement_word or '')}\n"
        f"Ignore enabled: {bool(args.enable_ignore_replacement_token) or bool(str(args.ignore_replacement_token or '').strip())}\n"
        f"Ignore token: {str(args.ignore_replacement_token or '')}\n"
        f"Include assistant marker: {bool(args.include_assistant_marker)}\n"
        f"Replace layers: {str(args.replace_layers or '')}\n"
        f"KV replace mode: {int(args.kv_replace_mode)} "
        "(1=target-only, 2=target+following(after replace layers), 3=full-replaced-prompt)\n"
        f"Replace K: {bool(args.replace_k)} ({'KV' if bool(args.replace_k) else 'V-only'})\n"
        f"Layer shortcut: {bool(args.enable_layer_shortcut)} "
        f"(start={int(args.shortcut_start_layer)}, target={int(args.shortcut_target_layer)}, 0-based)\n"
        f"Generated tokens: {int(result.get('generated_token_count') or 0)}\n"
        f"{'Assistant' if bool(args.include_assistant_marker) else 'Post-last-token output'}: {assistant_message}"
    )
    # Heatmap-related summary text is also disabled while popup heatmap is disabled.
    # if not valid_heatmaps:
    #     heatmap_err = str((meta or {}).get("assistant_target_heatmap_error") or "").strip()
    #     if heatmap_err:
    #         summary_text += f"\nAssistant->target heatmap unavailable: {heatmap_err}"
    print(summary_text)
    ui_tasks: list[dict[str, str]] = []
    # Disable popup heatmap task without removing code.
    # if valid_heatmaps:
    #     ui_tasks.append({"name": "render_heatmap", "value_key": "heatmaps"})
    if top_logits:
        ui_tasks.append({"name": "render_logits", "value_key": "top_logits"})
    ui_tasks.append({"name": "render_text_output", "value_key": "summary_text"})
    if input_token_top_logits:
        ui_tasks.append({"name": "render_input_token_logits_table", "value_key": "input_token_top_logits"})
    return {
        "hidden_state_heatmap": {
            "ok": True,
            "study": "chat_attention_word_replacement",
            "prompt_text": str(args.prompt_text or ""),
            "assistant_message": assistant_message,
            "generated_token_count": int(result.get("generated_token_count") or 0),
            "attention_word_replacement": meta,
            "top_logits": top_logits,
            "input_token_top_logits": input_token_top_logits,
            "heatmaps": valid_heatmaps,
            "summary_text": summary_text,
            "ui_tasks": ui_tasks,
        }
    }









