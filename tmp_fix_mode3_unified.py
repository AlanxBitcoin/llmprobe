from pathlib import Path
p=Path(r"C:\AI_Model\probe\src\study\study_chat_attention_word_replacement.py")
raw=p.read_text(encoding='utf-8')
old='''        # Unified post-flow: compute next-token logits from the same cache path used for generation.
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
new='''        # Unified post-flow: still honor mode-specific cache mix before generation.
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
if old not in raw:
    raise SystemExit('target_block_not_found')
raw=raw.replace(old,new,1)
p.write_text(raw,encoding='utf-8')
print('patched')
