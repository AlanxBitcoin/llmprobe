from __future__ import annotations

from typing import Any

import torch


def normalize_chat_template_ids(payload: Any) -> list[int]:
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


def get_assistant_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
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
    full_ids = normalize_chat_template_ids(chat_with_prompt)
    without_ids = normalize_chat_template_ids(chat_without_prompt)
    if not full_ids:
        raise ValueError("Chat template returned empty ids.")
    if len(without_ids) >= len(full_ids):
        raise ValueError("Unable to locate assistant generation suffix.")
    suffix_ids = full_ids[len(without_ids) :]
    if not suffix_ids:
        raise ValueError("Assistant suffix is empty.")
    return [int(x) for x in suffix_ids]
