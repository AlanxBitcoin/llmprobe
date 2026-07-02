from pathlib import Path
p = Path(r"C:\AI_Model\probe\src\study\study_chat_attention_word_replacement.py")
raw = p.read_text(encoding='utf-8')

anchor = "def _safe_decode_token(tokenizer, token_id: int) -> str:\n"
if anchor not in raw:
    raise SystemExit('anchor_not_found')
helper = '''def _truncate_cache_to_prefix_len(cache_obj: Any, seq_len: int) -> Any:
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


'''
raw = raw.replace(anchor, helper + anchor, 1)

old_block = '''        # Unified post-flow: assistant ids (if enabled) are already appended to input queue.
        # So the continuation starts from the prefill tail logits directly.
        assistant_mixed = bool(include_assistant_marker)
        prefill_logits = getattr(original_prefill, "logits", None)
        if not (torch.is_tensor(prefill_logits) and prefill_logits.ndim == 3 and int(prefill_logits.shape[1]) > 0):
            raise ValueError("Unable to compute next-token logits from prefill.")
        last_logits: torch.Tensor | None = prefill_logits[:, -1, :]
'''
new_block = '''        # Unified post-flow: compute next-token logits from the same cache path used for generation.
        assistant_mixed = bool(include_assistant_marker)
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
            output_hidden_states=bool(layer_jump_enabled),
        )
        original_past = out_last.past_key_values
        last_logits: torch.Tensor | None = out_last.logits[:, -1, :]
        if layer_jump_enabled:
            last_logits = _maybe_apply_layer_shortcut_logits(
                model=model,
                tokenizer=tokenizer,
                config=config,
                input_ids=torch.tensor([[last_token_id]], dtype=torch.long, device=device),
                step_hidden_states=getattr(out_last, "hidden_states", None),
                shortcut_start_layer_zero_based=int(shortcut_start_layer_zero_based or 0),
                shortcut_target_layer_zero_based=int(shortcut_target_layer_zero_based or 0),
            )
'''
if old_block not in raw:
    raise SystemExit('old_post_block_not_found')
raw = raw.replace(old_block, new_block, 1)

p.write_text(raw, encoding='utf-8')
print('fixed')
