from pathlib import Path
p = Path(r"C:\AI_Model\probe\src\study\study_chat_attention_word_replacement.py")
raw = p.read_text(encoding='utf-8')
old = '''        assistant_mixed = False
        last_logits: torch.Tensor | None = None
        for token_id in assistant_suffix_ids:
            one_token = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
            if int(token_id) == assistant_token_id and not assistant_mixed:
                if int(parsed_kv_replace_mode) != 2 and replaced_past is None:
                    raise ValueError("Internal error: missing replaced cache before assistant merge.")
                if int(parsed_kv_replace_mode) == 1:
                    # mode 1 only changes assistant->target attention and restores immediately after this step.
                    mixed_past, original_past_before_assistant = _mix_assistant_cache_mode_1(
                        original_past=original_past,
                        replaced_past=replaced_past,
                        replace_layer_indices_zero_based=replace_layer_indices_zero_based,
                        replace_k=bool(replace_k),
                        target_token_index=int(target_token_index),
                    )
                elif int(parsed_kv_replace_mode) == 2:
                    mixed_past, original_past_before_assistant = _mix_assistant_cache_mode_2(
                        original_past=original_past,
                        replaced_past=replaced_past,
                        replace_layer_indices_zero_based=replace_layer_indices_zero_based,
                        replace_k=bool(replace_k),
                        target_token_index=int(target_token_index),
                    )
                else:
                    mixed_past, original_past_before_assistant = _mix_assistant_cache_mode_3(
                        original_past=original_past,
                        replaced_past=replaced_past,
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
                if int(parsed_kv_replace_mode) == 1 and baseline_prompt_past is not None:
                    # mode1 restores target token KV after assistant mix step.
                    restored_past = _merge_past_key_values(
                        original=out.past_key_values,
                        replaced=baseline_prompt_past,
                        replace_layer_indices_zero_based=replace_layer_indices_zero_based,
                        replace_k=bool(replace_k),
                        kv_replace_mode=1,
                        target_token_index=target_token_index,
                    )
                    original_past = restored_past
                else:
                    # mode2/mode3: continue directly from current cache.
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
                    if parsed_kv_replace_mode == 3:
                        out_replaced = model(
                            input_ids=one_token,
                            past_key_values=replaced_past,
                            use_cache=True,
                            return_dict=True,
                        )
                        replaced_past = out_replaced.past_key_values

        if not assistant_mixed:
            raise ValueError("Unable to locate mix trigger token in suffix.")
        if last_logits is None:
            raise ValueError("Unable to compute next-token logits from assistant suffix.")
'''
new = '''        # Unified post-flow: assistant ids (if enabled) are already appended to input queue.
        # So the continuation starts from the prefill tail logits directly.
        assistant_mixed = bool(include_assistant_marker)
        prefill_logits = getattr(original_prefill, "logits", None)
        if not (torch.is_tensor(prefill_logits) and prefill_logits.ndim == 3 and int(prefill_logits.shape[1]) > 0):
            raise ValueError("Unable to compute next-token logits from prefill.")
        last_logits: torch.Tensor | None = prefill_logits[:, -1, :]
'''
if old not in raw:
    raise SystemExit('old_block_not_found')
raw = raw.replace(old, new, 1)
p.write_text(raw, encoding='utf-8')
print('updated')
