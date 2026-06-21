from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any

import torch
from ..probes.probe_attention import fetch_head_attention_metrics_for_input_ids
from ..probes.probe_layer_neuron import run_starting_from_middle_layer_probe
from ..probes.probe_layer_shortcut import validate_shortcut_layers_zero_based
from ..runtime_api import RuntimeRequest, get_runtime_api, start_llama_api
from ..utils.token_hidden_store import resolve_assistant_token_id


def _normalize_chat_template_ids(payload: Any) -> list[int]:
    if payload is None:
        return []
    if hasattr(payload, "get"):
        ids = payload.get("input_ids")
    else:
        ids = payload
    if torch.is_tensor(ids):
        if ids.ndim == 2:
            ids = ids[0]
        return [int(x) for x in ids.detach().to(device="cpu", dtype=torch.long).tolist()]
    if isinstance(ids, (list, tuple)):
        if ids and isinstance(ids[0], (list, tuple)):
            return [int(x) for x in ids[0]]
        return [int(x) for x in ids]
    return []


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
    if isinstance(layers, list):
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
) -> Any:
    # Preferred path for transformers>=5 Cache objects (e.g. DynamicCache):
    # mutate original.layers in place to keep Cache type and metadata intact.
    original_layers_obj = getattr(original, "layers", None)
    replaced_layers_obj = getattr(replaced, "layers", None)
    if isinstance(original_layers_obj, list) and isinstance(replaced_layers_obj, list):
        if len(original_layers_obj) != len(replaced_layers_obj):
            raise ValueError(
                f"Cannot merge cache layers with different counts: {len(original_layers_obj)} vs {len(replaced_layers_obj)}"
            )
        for i in range(len(original_layers_obj)):
            if int(i) not in replace_layer_indices_zero_based:
                continue
            original_layer = original_layers_obj[i]
            replaced_layer = replaced_layers_obj[i]
            # CacheLayer path (transformers>=5): replace V always; replace K only when requested.
            if hasattr(original_layer, "values") and hasattr(replaced_layer, "values"):
                original_layer.values = replaced_layer.values
            else:
                original_layers_obj[i] = replaced_layer
                continue
            if bool(replace_k):
                if hasattr(original_layer, "keys") and hasattr(replaced_layer, "keys"):
                    original_layer.keys = replaced_layer.keys
                else:
                    original_layers_obj[i] = replaced_layer
        return original

    # Legacy tuple/list fallback.
    original_layers = _as_layer_sequence(original)
    replaced_layers = _as_layer_sequence(replaced)
    if len(original_layers) != len(replaced_layers):
        raise ValueError(
            f"Cannot merge caches with different layer counts: {len(original_layers)} vs {len(replaced_layers)}"
        )
    out: list[Any] = []
    for i in range(len(original_layers)):
        if int(i) not in replace_layer_indices_zero_based:
            out.append(original_layers[i])
            continue
        if bool(replace_k):
            out.append(replaced_layers[i])
            continue
        orig_item = original_layers[i]
        repl_item = replaced_layers[i]
        if isinstance(orig_item, (list, tuple)) and isinstance(repl_item, (list, tuple)):
            if len(orig_item) >= 2 and len(repl_item) >= 2:
                merged = list(orig_item)
                merged[1] = repl_item[1]  # replace V only
                out.append(tuple(merged))
                continue
        out.append(orig_item)
    return tuple(out)


def _parse_replace_layers_spec(spec: str, *, layer_count: int) -> tuple[set[int], list[int]]:
    raw = str(spec or "").strip()
    if not raw:
        raise ValueError("replace_layers is required.")
    normalized = raw.replace("，", ",").replace("、", ",").replace(" ", "")
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


def _top_logits_rows_from_tensor(
    *,
    tokenizer,
    logits: torch.Tensor,
    top_k: int = 100,
) -> list[dict[str, Any]]:
    vec = logits.detach().to(dtype=torch.float32)
    if vec.ndim == 2:
        vec = vec[0]
    if vec.ndim != 1:
        return []
    k = int(min(max(1, int(top_k)), int(vec.shape[0])))
    top_vals, top_ids = torch.topk(vec, k=k)
    ids = top_ids.detach().cpu().tolist()
    vals = top_vals.detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids) if tokenizer is not None else ["" for _ in ids]
    texts = (
        [tokenizer.decode([int(tok_id)], clean_up_tokenization_spaces=False) for tok_id in ids]
        if tokenizer is not None
        else ["" for _ in ids]
    )
    rows: list[dict[str, Any]] = []
    for i, tok_id in enumerate(ids, start=1):
        rows.append(
            {
                "rank": int(i),
                "token_id": int(tok_id),
                "token": str(tokens[i - 1]),
                "text": str(texts[i - 1]),
                "logit": float(vals[i - 1]),
            }
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
        raise ValueError("Layer jump requires config.")
    if not isinstance(step_hidden_states, (list, tuple)):
        raise ValueError("Layer jump requires output_hidden_states from model forward.")
    source_idx = int(shortcut_start_layer_zero_based) + 1
    if source_idx < 0 or source_idx >= len(step_hidden_states):
        raise ValueError(
            f"Layer jump source row out of range: source={int(shortcut_start_layer_zero_based)}, "
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
        raise ValueError(f"layer_jump_failed: {str(transit_error or 'unknown')}")
    transit_result = transit.get("result") if isinstance(transit, dict) else None
    logits = None
    if isinstance(transit_result, dict):
        logits = transit_result.get("logits")
        if logits is None:
            outputs = transit_result.get("outputs")
            logits = getattr(outputs, "logits", None) if outputs is not None else None
    if not torch.is_tensor(logits):
        raise ValueError("layer_jump_failed: missing logits")
    if logits.ndim == 3:
        return logits[:, -1, :]
    if logits.ndim == 2:
        return logits
    raise ValueError(f"layer_jump_failed: unexpected logits ndim={int(logits.ndim)}")


def run_study(
    *,
    model,
    tokenizer,
    messages: list[dict[str, str]],
    target_word: str,
    replacement_word: str,
    replace_layers: str,
    replace_k: bool = False,
    enable_layer_jump: bool = False,
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
    if not include_assistant_marker:
        raise ValueError("Attention word replacement requires include_assistant_marker=true.")

    target = str(target_word or "").strip()
    replacement = str(replacement_word or "").strip()
    if not target:
        raise ValueError("target_word is required.")
    if not replacement:
        raise ValueError("replacement_word is required.")
    target_ids = [int(x) for x in (tokenizer(target, add_special_tokens=False).get("input_ids") or [])]
    replacement_ids = [int(x) for x in (tokenizer(replacement, add_special_tokens=False).get("input_ids") or [])]
    if len(target_ids) != 1:
        raise ValueError(
            f"target_word must be single-token, got {len(target_ids)} tokens: {target!r}, ids={target_ids}"
        )
    if len(replacement_ids) != 1:
        raise ValueError(
            f"replacement_word must be single-token, got {len(replacement_ids)} tokens: {replacement!r}, ids={replacement_ids}"
        )

    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("Model does not expose decoder layers.")
    layer_count = int(len(layers))
    replace_layer_indices_zero_based, replace_layers_zero_based = _parse_replace_layers_spec(
        str(replace_layers or ""),
        layer_count=layer_count,
    )
    layer_jump_enabled = bool(enable_layer_jump)
    shortcut_start_layer_zero_based: int | None = None
    shortcut_target_layer_zero_based: int | None = None
    if layer_jump_enabled:
        src, tgt, _ = validate_shortcut_layers_zero_based(
            source_layer=int(shortcut_start_layer),
            target_layer=int(shortcut_target_layer),
            layer_count=layer_count,
        )
        shortcut_start_layer_zero_based = int(src)
        shortcut_target_layer_zero_based = int(tgt)

    replaced_messages, replaced_count = _replace_target_in_user_messages(
        messages,
        target=target,
        replacement=replacement,
    )
    if replaced_count <= 0:
        raise ValueError("Target word was not found in user messages.")
    if replaced_count > 1:
        raise ValueError(
            f"target_word must appear exactly once in user messages, got {int(replaced_count)} occurrences: {target!r}"
        )

    chat_with_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    chat_without_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    replaced_with_prompt = tokenizer.apply_chat_template(
        replaced_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    replaced_without_prompt = tokenizer.apply_chat_template(
        replaced_messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    full_ids = _normalize_chat_template_ids(chat_with_prompt)
    without_ids = _normalize_chat_template_ids(chat_without_prompt)
    full_ids_replaced = _normalize_chat_template_ids(replaced_with_prompt)
    without_ids_replaced = _normalize_chat_template_ids(replaced_without_prompt)
    if not full_ids or not full_ids_replaced:
        raise ValueError("Chat template returned empty ids.")
    if len(without_ids) >= len(full_ids):
        raise ValueError("Unable to locate assistant generation suffix for original prompt.")
    if len(without_ids_replaced) >= len(full_ids_replaced):
        raise ValueError("Unable to locate assistant generation suffix for replaced prompt.")

    assistant_suffix_ids = full_ids[len(without_ids) :]
    assistant_suffix_replaced_ids = full_ids_replaced[len(without_ids_replaced) :]
    if assistant_suffix_ids != assistant_suffix_replaced_ids:
        raise ValueError("Assistant suffix differs between original and replaced prompts.")
    if not assistant_suffix_ids:
        raise ValueError("Assistant suffix is empty.")

    prefix_ids = full_ids[: len(without_ids)]
    prefix_ids_replaced = full_ids_replaced[: len(without_ids_replaced)]
    if len(prefix_ids) != len(prefix_ids_replaced):
        if not bool(replace_k):
            raise ValueError(
                "Replacement changed prompt token length before assistant marker. "
                "V-only mode requires equal prefix length; enable replace_k=true (KV mode), "
                "or use a replacement with the same tokenized length."
            )

    assistant_token_ids = tokenizer("assistant", add_special_tokens=False).get("input_ids") or []
    if len(assistant_token_ids) != 1:
        raise ValueError("Unable to resolve unique assistant marker token id.")
    assistant_token_id = int(assistant_token_ids[0])

    device = next(model.parameters()).device
    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    replaced_prefix_tensor = torch.tensor([prefix_ids_replaced], dtype=torch.long, device=device)

    with torch.inference_mode():
        original_prefill = model(
            input_ids=prefix_tensor,
            use_cache=True,
            return_dict=True,
        )
        replaced_prefill = model(
            input_ids=replaced_prefix_tensor,
            use_cache=True,
            return_dict=True,
        )
        original_past = original_prefill.past_key_values
        replaced_past = replaced_prefill.past_key_values

        assistant_mixed = False
        last_logits: torch.Tensor | None = None
        for token_id in assistant_suffix_ids:
            one_token = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
            if int(token_id) == assistant_token_id and not assistant_mixed:
                mixed_past = _merge_past_key_values(
                    original=original_past,
                    replaced=replaced_past,
                    replace_layer_indices_zero_based=replace_layer_indices_zero_based,
                    replace_k=bool(replace_k),
                )
                out = model(
                    input_ids=one_token,
                    past_key_values=mixed_past,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=bool(layer_jump_enabled),
                )
                original_past = out.past_key_values
                last_logits = out.logits[:, -1, :]
                if layer_jump_enabled:
                    last_logits = _maybe_apply_layer_shortcut_logits(
                        model=model,
                        tokenizer=tokenizer,
                        config=config,
                        input_ids=one_token,
                        step_hidden_states=getattr(out, "hidden_states", None),
                        shortcut_start_layer_zero_based=int(shortcut_start_layer_zero_based or 0),
                        shortcut_target_layer_zero_based=int(shortcut_target_layer_zero_based or 0),
                    )
                assistant_mixed = True
            else:
                out_original = model(
                    input_ids=one_token,
                    past_key_values=original_past,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=False,
                )
                original_past = out_original.past_key_values
                last_logits = out_original.logits[:, -1, :]
                if not assistant_mixed:
                    out_replaced = model(
                        input_ids=one_token,
                        past_key_values=replaced_past,
                        use_cache=True,
                        return_dict=True,
                    )
                    replaced_past = out_replaced.past_key_values

        if not assistant_mixed:
            raise ValueError("Assistant marker token not found in assistant suffix.")
        if last_logits is None:
            raise ValueError("Unable to compute next-token logits from assistant suffix.")

        generated: list[int] = []
        max_steps = int(max_new_tokens)
        if max_steps <= 0:
            top_logits = _top_logits_rows_from_tensor(tokenizer=tokenizer, logits=last_logits, top_k=100)
            return {
                "status": "ok",
                "assistant_message": "",
                "generated_token_count": 0,
                "top_logits": top_logits,
                "attention_word_replacement": {
                    "enabled": True,
                    "target_word": target,
                    "replacement_word": replacement,
                    "replace_layers": str(replace_layers or ""),
                    "replace_layers_zero_based": list(replace_layers_zero_based),
                    "replace_k": bool(replace_k),
                    "replace_mode": "kv" if bool(replace_k) else "v_only",
                    "replaced_occurrences": int(replaced_count),
                    "assistant_token_id": int(assistant_token_id),
                    "assistant_suffix_length": int(len(assistant_suffix_ids)),
                    "assistant_mixed": bool(assistant_mixed),
                    "layer_jump_enabled": bool(layer_jump_enabled),
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
    top_logits = _top_logits_rows_from_tensor(tokenizer=tokenizer, logits=last_logits, top_k=100)
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
        "heatmaps": assistant_target_heatmaps,
        "attention_word_replacement": {
            "enabled": True,
            "target_word": target,
            "replacement_word": replacement,
            "replace_layers": str(replace_layers or ""),
            "replace_layers_zero_based": list(replace_layers_zero_based),
            "replace_k": bool(replace_k),
            "replace_mode": "kv" if bool(replace_k) else "v_only",
            "replaced_occurrences": int(replaced_count),
            "assistant_token_id": int(assistant_token_id),
            "assistant_suffix_length": int(len(assistant_suffix_ids)),
            "assistant_mixed": bool(assistant_mixed),
            "layer_jump_enabled": bool(layer_jump_enabled),
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
        default="Alice gave Bob a book.Who has the book?Please directly give the name",
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
        "--replace-layers",
        type=str,
        default="0-",
        help="Layers to replace (0-based). Examples: 0- , 0-3 , 0,3",
    )
    parser.add_argument(
        "--replace-k",
        type=bool_parser,
        default=True,
        help="Whether to replace K together with V (true/false). false means V-only replacement.",
    )
    parser.add_argument(
        "--enable-layer-jump",
        type=bool_parser,
        default=False,
        help="Whether to enable layer jump (shortcut) during assistant suffix and generation.",
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
        help="Whether to include assistant generation marker (must be true for this study).",
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
        replace_layers=str(args.replace_layers or ""),
        replace_k=bool(args.replace_k),
        enable_layer_jump=bool(args.enable_layer_jump),
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
        f"Replace layers: {str(args.replace_layers or '')}\n"
        f"Replace K: {bool(args.replace_k)} ({'KV' if bool(args.replace_k) else 'V-only'})\n"
        f"Layer jump: {bool(args.enable_layer_jump)} "
        f"(start={int(args.shortcut_start_layer)}, target={int(args.shortcut_target_layer)}, 0-based)\n"
        f"Generated tokens: {int(result.get('generated_token_count') or 0)}\n"
        f"Assistant: {assistant_message}"
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
    return {
        "hidden_state_heatmap": {
            "ok": True,
            "study": "chat_attention_word_replacement",
            "prompt_text": str(args.prompt_text or ""),
            "assistant_message": assistant_message,
            "generated_token_count": int(result.get("generated_token_count") or 0),
            "attention_word_replacement": meta,
            "top_logits": top_logits,
            "heatmaps": valid_heatmaps,
            "summary_text": summary_text,
            "ui_tasks": ui_tasks,
        }
    }
