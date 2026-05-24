from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Provide reusable vector ranking helpers for logits/cosine methods.
# - Keep ranking logic out of study/probe orchestration code.

from typing import Any

import torch
import torch.nn.functional as F


def rank_vector_by_logits(
    *,
    model,
    vector: torch.Tensor | list[float],
    tokenizer=None,
    top_k: int = 15,
    apply_final_norm: bool = True,
) -> list[dict[str, Any]]:
    """Top-k ranking by LM head logits."""
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return []

    device = next(model.parameters()).device
    hidden = torch.as_tensor(vector, dtype=torch.float32, device=device).flatten()
    if hidden.ndim != 1:
        raise ValueError(f"vector must be 1D, got shape={tuple(hidden.shape)}")

    hidden_for_logits = hidden.unsqueeze(0)
    lm_head_dtype = lm_head.weight.dtype
    hidden_for_logits = hidden_for_logits.to(dtype=lm_head_dtype)
    if apply_final_norm:
        base_model = getattr(model, "model", None)
        final_norm = getattr(base_model, "norm", None)
        if final_norm is not None:
            hidden_for_logits = final_norm(hidden_for_logits)

    with torch.no_grad():
        cfg = getattr(model, "config", None)
        pretraining_tp = int(getattr(cfg, "pretraining_tp", 1) or 1)
        if pretraining_tp > 1:
            vocab_size = int(getattr(cfg, "vocab_size", lm_head.weight.shape[0]))
            lm_head_slices = lm_head.weight.split(vocab_size // pretraining_tp, dim=0)
            logits_chunks = [F.linear(hidden_for_logits, lm_head_slice) for lm_head_slice in lm_head_slices]
            logits = torch.cat(logits_chunks, dim=-1).squeeze(0).to(dtype=torch.float32)
        else:
            logits = lm_head(hidden_for_logits).squeeze(0).to(dtype=torch.float32)
        k = min(int(top_k), int(logits.shape[0]))
        top_vals, top_ids = torch.topk(logits, k=k)

    top_ids_list = top_ids.detach().cpu().tolist()
    top_vals_list = top_vals.detach().cpu().tolist()
    token_texts = tokenizer.convert_ids_to_tokens(top_ids_list) if tokenizer is not None else ["" for _ in top_ids_list]
    decoded_texts = (
        [tokenizer.decode([int(tok_id)], clean_up_tokenization_spaces=False) for tok_id in top_ids_list]
        if tokenizer is not None
        else ["" for _ in top_ids_list]
    )

    rows: list[dict[str, Any]] = []
    for i, tok_id in enumerate(top_ids_list, start=1):
        rows.append(
            {
                "rank": i,
                "token_id": int(tok_id),
                "token": str(token_texts[i - 1]),
                "text": str(decoded_texts[i - 1]),
                "logit": float(top_vals_list[i - 1]),
            }
        )
    return rows


def rank_vector_by_cosine(
    *,
    model,
    vector: torch.Tensor | list[float],
    tokenizer=None,
    compare_matrix: torch.Tensor | None = None,
    top_k: int = 15,
) -> list[dict[str, Any]]:
    """Top-k ranking by cosine similarity."""
    embeddings = model.get_input_embeddings()
    if embeddings is None or getattr(embeddings, "weight", None) is None:
        return []

    device = next(model.parameters()).device
    hidden = torch.as_tensor(vector, dtype=torch.float32, device=device).flatten()
    if hidden.ndim != 1:
        raise ValueError(f"vector must be 1D, got shape={tuple(hidden.shape)}")

    target_matrix = compare_matrix if compare_matrix is not None else embeddings.weight
    target_matrix = target_matrix.to(device=device, dtype=torch.float32)
    if target_matrix.ndim != 2 or target_matrix.shape[1] != hidden.shape[0]:
        raise ValueError(
            f"compare_matrix shape mismatch: expected [N,{hidden.shape[0]}], got {tuple(target_matrix.shape)}"
        )

    with torch.no_grad():
        hidden_norm = F.normalize(hidden, dim=0)
        target_norm = F.normalize(target_matrix, dim=1)
        cosine = torch.sum(target_norm * hidden_norm.unsqueeze(0), dim=1)
        k = min(int(top_k), int(cosine.shape[0]))
        top_vals, top_ids = torch.topk(cosine, k=k)

    top_ids_list = top_ids.detach().cpu().tolist()
    top_vals_list = top_vals.detach().cpu().tolist()
    token_texts = tokenizer.convert_ids_to_tokens(top_ids_list) if tokenizer is not None else ["" for _ in top_ids_list]
    decoded_texts = (
        [tokenizer.decode([int(tok_id)], clean_up_tokenization_spaces=False) for tok_id in top_ids_list]
        if tokenizer is not None
        else ["" for _ in top_ids_list]
    )

    rows: list[dict[str, Any]] = []
    for i, tok_id in enumerate(top_ids_list, start=1):
        rows.append(
            {
                "rank": i,
                "token_id": int(tok_id),
                "token": str(token_texts[i - 1]),
                "text": str(decoded_texts[i - 1]),
                "cosine_similarity": float(top_vals_list[i - 1]),
            }
        )
    return rows


def rank_vector_logits_and_cosine(
    *,
    model,
    vector: torch.Tensor | list[float],
    tokenizer=None,
    compare_matrix: torch.Tensor | None = None,
    top_k: int = 15,
    apply_final_norm: bool = True,
) -> list[dict[str, Any]]:
    """Top-k by logits, with cosine similarity attached for the same ids.

    Cosine is computed in LM-head space by default (not input-embedding space).
    """
    logits_rows = rank_vector_by_logits(
        model=model,
        vector=vector,
        tokenizer=tokenizer,
        top_k=top_k,
        apply_final_norm=apply_final_norm,
    )
    if not logits_rows:
        return []

    device = next(model.parameters()).device
    hidden = torch.as_tensor(vector, dtype=torch.float32, device=device).flatten()
    lm_head = getattr(model, "lm_head", None)
    lm_head_weight = getattr(lm_head, "weight", None)
    if compare_matrix is None and lm_head_weight is None:
        return logits_rows

    target_matrix = compare_matrix if compare_matrix is not None else lm_head_weight
    target_matrix = target_matrix.to(device=device, dtype=torch.float32)
    ids = torch.tensor([int(row["token_id"]) for row in logits_rows], dtype=torch.long, device=device)
    target = target_matrix[ids]
    with torch.no_grad():
        hidden_norm = F.normalize(hidden, dim=0)
        target_norm = F.normalize(target, dim=1)
        cosine = torch.sum(target_norm * hidden_norm.unsqueeze(0), dim=1).detach().cpu().tolist()

    merged: list[dict[str, Any]] = []
    for i, row in enumerate(logits_rows):
        out = dict(row)
        out["cosine_similarity"] = float(cosine[i])
        merged.append(out)
    return merged
