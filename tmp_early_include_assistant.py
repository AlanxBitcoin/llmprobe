from pathlib import Path
p=Path(r"C:\AI_Model\probe\src\study\study_chat_attention_word_replacement.py")
raw=p.read_text(encoding='utf-8')
old='''    if len(full_ids) < len(assistant_suffix_ids) + 1 or len(full_ids_replaced) < len(assistant_suffix_replaced_ids) + 1:
        raise ValueError("Prompt before assistant suffix must contain at least one token.")

    if len(full_ids) != len(full_ids_replaced):
        if parsed_kv_replace_mode in (1, 2):
            raise ValueError(
                "kv_replace_mode 1/2 requires equal tokenized prompt length when include_assistant_marker=false."
            )
        if not bool(replace_k):
            raise ValueError(
                "V-only mode requires equal prompt length when include_assistant_marker=false."
            )

    prefix_ids = full_ids[: len(full_ids) - len(assistant_suffix_ids)]
    prefix_ids_replaced = full_ids_replaced[: len(full_ids_replaced) - len(assistant_suffix_replaced_ids)]
'''
new='''    if len(full_ids) != len(full_ids_replaced):
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
'''
if old not in raw:
    raise SystemExit('target_not_found')
raw=raw.replace(old,new,1)
p.write_text(raw,encoding='utf-8')
print('patched')
