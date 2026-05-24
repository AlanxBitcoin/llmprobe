from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .token_hidden_store import TokenHiddenStore, build_hidden_store_config


_STORE_CACHE: dict[str, TokenHiddenStore] = {}


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _last_token_hidden(hidden_state: torch.Tensor) -> torch.Tensor:
    return hidden_state[0, -1, :].detach().float().cpu()


def _store_key(protocol: str, data_file: str) -> str:
    return f"{protocol}:{data_file}"


def _get_hidden_store(bundle, config: dict[str, Any] | None) -> TokenHiddenStore | None:
    if not config:
        return None
    cfg = build_hidden_store_config(config)
    if cfg is None:
        return None
    key = _store_key(cfg.protocol, str(cfg.data_file))
    cached = _STORE_CACHE.get(key)
    if cached is None:
        cached = TokenHiddenStore(cfg, bundle.tokenizer)
        _STORE_CACHE[key] = cached
    return cached


def _single_token_id(tokenizer, word: str) -> int | None:
    ids = tokenizer(word, add_special_tokens=False).get("input_ids") or []
    if len(ids) != 1:
        return None
    return int(ids[0])


def extract_word_hidden_states(bundle, word: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    tokenizer = bundle.tokenizer
    store = _get_hidden_store(bundle, config)
    token_id = _single_token_id(tokenizer, word)

    if store is not None and token_id is not None:
        layers = store.get_or_compute_layers(bundle, token_id)
        per_layer = []
        for layer_idx, vector in enumerate(layers):
            tensor = torch.from_numpy(np.asarray(vector, dtype=np.float32))
            per_layer.append(
                {
                    "layer": layer_idx,
                    "vector": np.asarray(vector, dtype=np.float32),
                    "abs_mean": float(tensor.abs().mean().item()),
                    "abs_max": float(tensor.abs().max().item()),
                }
            )
        encoded_tokens = tokenizer(word, return_tensors="pt")["input_ids"][0]
        return {
            "word": word,
            "tokens": tokenizer.convert_ids_to_tokens(encoded_tokens),
            "layers": per_layer,
        }

    model = bundle.model
    encoded = tokenizer(word, return_tensors="pt")
    device = _model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)

    per_layer = []
    for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
        vector = _last_token_hidden(layer_hidden)
        per_layer.append(
            {
                "layer": layer_idx,
                "vector": vector.numpy(),
                "abs_mean": float(vector.abs().mean().item()),
                "abs_max": float(vector.abs().max().item()),
            }
        )

    return {
        "word": word,
        "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]),
        "layers": per_layer,
    }


def summarize_top_dims(vector: np.ndarray, top_k: int) -> list[dict[str, float]]:
    abs_values = np.abs(vector)
    indices = np.argsort(abs_values)[::-1][:top_k]
    return [
        {
            "dim": int(idx),
            "value": float(vector[idx]),
            "abs_value": float(abs_values[idx]),
        }
        for idx in indices
    ]


def extract_sequence_positional_states(
    bundle,
    words: list[str],
    target_layer: int,
) -> list[dict[str, Any]]:
    """Run ONE forward pass on the full word sequence and extract per-word hidden states.

    For each word its hidden-state vector is taken from its LAST sub-token position
    in the concatenated sequence at ``target_layer``.  This lets us study how each
    colour word is encoded when ALL other colour words are visible as left context —
    in contrast to the per-word mode (no context) and all-input mode (only last token).

    Returns a list of dicts, one per word:
        word             – original word string
        tokens           – list of token strings that make up this word
        token_positions  – list of absolute positions in the full token sequence
        last_token_pos   – position used for vector extraction (last sub-token)
        vector           – np.ndarray of shape [hidden_dim]
    """
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = _model_device(model)

    full_text = " ".join(words)
    full_encoded = tokenizer(full_text, return_tensors="pt")

    # Determine how many tokens each word contributes to the full sequence.
    # Words after the first are preceded by a space, which Llama BPE encodes as
    # a leading '▁' prefix merged into the following token.
    word_tok_ids: list[list[int]] = []
    for i, word in enumerate(words):
        prefix = "" if i == 0 else " "
        ids: list[int] = tokenizer(prefix + word, add_special_tokens=False)["input_ids"]
        word_tok_ids.append(ids)

    full_encoded_gpu = {k: v.to(device) for k, v in full_encoded.items()}
    with torch.no_grad():
        outputs = model(**full_encoded_gpu, output_hidden_states=True)

    layer_hidden = outputs.hidden_states[target_layer]  # [1, N, D]
    seq_len: int = layer_hidden.shape[1]

    results: list[dict[str, Any]] = []
    pos = 1  # index 0 is the BOS token
    for word, tok_ids in zip(words, word_tok_ids):
        n = len(tok_ids)
        last_pos = min(pos + n - 1, seq_len - 1)
        vector = layer_hidden[0, last_pos, :].detach().float().cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(tok_ids)
        results.append(
            {
                "word": word,
                "tokens": tokens,
                "token_positions": list(range(pos, min(pos + n, seq_len))),
                "last_token_pos": last_pos,
                "vector": vector,
            }
        )
        pos += n

    return results
