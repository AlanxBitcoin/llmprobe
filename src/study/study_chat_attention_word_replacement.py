from __future__ import annotations

from typing import Any

import torch


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
    top_k: int = 15,
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


def run_study(
    *,
    model,
    tokenizer,
    messages: list[dict[str, str]],
    target_word: str,
    replacement_word: str,
    replace_layers: str,
    replace_k: bool = False,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    eos_token_id: int | None,
    include_assistant_marker: bool = True,
) -> dict[str, Any]:
    if not include_assistant_marker:
        raise ValueError("Attention word replacement requires include_assistant_marker=true.")

    target = str(target_word or "").strip()
    replacement = str(replacement_word or "").strip()
    if not target:
        raise ValueError("target_word is required.")
    if not replacement:
        raise ValueError("replacement_word is required.")

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

    replaced_messages, replaced_count = _replace_target_in_user_messages(
        messages,
        target=target,
        replacement=replacement,
    )
    if replaced_count <= 0:
        raise ValueError("Target word was not found in user messages.")

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
        raise ValueError(
            "Replacement changed prompt token length before assistant marker; "
            "this mode requires equal prefix length."
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
                )
                original_past = out.past_key_values
                last_logits = out.logits[:, -1, :]
                assistant_mixed = True
            else:
                out_original = model(
                    input_ids=one_token,
                    past_key_values=original_past,
                    use_cache=True,
                    return_dict=True,
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
            top_logits = _top_logits_rows_from_tensor(tokenizer=tokenizer, logits=last_logits, top_k=15)
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
    top_logits = _top_logits_rows_from_tensor(tokenizer=tokenizer, logits=last_logits, top_k=15)
    return {
        "status": "ok",
        "assistant_message": text,
        "generated_token_count": int(len(generated)),
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
        },
    }
