from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Hidden-state extraction APIs include three modes:
#   1) extract_single_word_states (per-word forward)
#   2) extract_all_words_state (single pass, last token)
#   3) extract_sequence_positional_states (single pass, per-word positions)
# - extract_word_hidden_states uses hidden_store read-through by default.

from typing import Any
import sys
import threading
import time

import numpy as np
import torch

from .token_hidden_store import TokenHiddenStore, build_hidden_store_config


class HiddenStoreRuntime:
    """Runtime owner for hidden-store instances (memmap handles)."""

    def __init__(self) -> None:
        self._stores: dict[str, TokenHiddenStore] = {}
        self._lock = threading.RLock()

    def _store_key(self, protocol: str, data_file: str) -> str:
        return f"{protocol}:{data_file}"

    def get_or_create(self, cfg, tokenizer) -> TokenHiddenStore:
        key = self._store_key(cfg.protocol, str(cfg.data_file))
        with self._lock:
            cached = self._stores.get(key)
            if cached is None:
                cached = TokenHiddenStore(cfg, tokenizer)
                self._stores[key] = cached
                print(
                    f"[single_word_hidden_state] opened store memmap key={key} "
                    f"data={cfg.data_file.name} done={cfg.progress_file.name}",
                    file=sys.stderr,
                )
            return cached


_HIDDEN_STORE_RUNTIME = HiddenStoreRuntime()


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _last_token_hidden(hidden_state: torch.Tensor) -> torch.Tensor:
    return hidden_state[0, -1, :].detach().float().cpu()


def _get_hidden_store(bundle, config: dict[str, Any] | None) -> TokenHiddenStore:
    cfg = build_hidden_store_config(config, bundle=bundle)
    return _HIDDEN_STORE_RUNTIME.get_or_create(cfg, bundle.tokenizer)


def preload_hidden_store(bundle, config: dict[str, Any] | None = None) -> None:
    """Explicit initialization hook for hidden-store memmap objects."""
    _ = _get_hidden_store(bundle, config)
    print("[single_word_hidden_state] hidden_store preloaded", file=sys.stderr)


def preload_hidden_store_from_disk(config: dict[str, Any]) -> None:
    """Main-startup preload: bind store object cache to concrete disk files."""
    from transformers import AutoConfig, AutoTokenizer

    model_cfg = dict((config or {}).get("model") or {})
    model_path = model_cfg.get("model_name_or_path")
    tokenizer_path = model_cfg.get("tokenizer_name_or_path") or model_path
    if not model_path:
        raise ValueError("Config requires model.model_name_or_path")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    auto_cfg = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    merged = dict(config or {})
    hs_cfg = dict((merged.get("hidden_store") or {}))
    hs_cfg.setdefault("n_layers", int(getattr(auto_cfg, "num_hidden_layers")) + 1)
    hs_cfg.setdefault("hidden_dim", int(getattr(auto_cfg, "hidden_size")))
    merged["hidden_store"] = hs_cfg
    store_cfg = build_hidden_store_config(merged, bundle=None)
    _ = _HIDDEN_STORE_RUNTIME.get_or_create(store_cfg, tokenizer)
    print(
        f"[startup] hidden_store initialized data={store_cfg.data_file} done={store_cfg.progress_file}",
        file=sys.stderr,
    )


def extract_word_hidden_states(bundle, word: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    tokenizer = bundle.tokenizer
    store = _get_hidden_store(bundle, config)

    layers = store.get_word_layers(bundle, word)
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


def extract_single_word_states(
    bundle,
    words: list[str],
    target_layer: int,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Per-word forward path: one model call per word.

    Returns a row per input word with the selected layer vector and summary stats.
    """
    rows: list[dict[str, Any]] = []
    for word in words:
        hidden = extract_word_hidden_states(bundle, word, config=config)
        layer_info = hidden["layers"][target_layer]
        rows.append(
            {
                "word": word,
                "tokens": hidden["tokens"],
                "target_layer": int(target_layer),
                "vector": np.asarray(layer_info["vector"], dtype=np.float32),
                "abs_mean": float(layer_info["abs_mean"]),
                "abs_max": float(layer_info["abs_max"]),
            }
        )
    return rows


def extract_all_words_state(
    bundle,
    words: list[str],
    target_layer: int,
) -> dict[str, Any]:
    """Single forward pass on concatenated words; returns last-token state."""
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = _model_device(model)

    text = " ".join(words)
    encoded = tokenizer(text, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)

    layer_hidden = outputs.hidden_states[target_layer]
    vector = _last_token_hidden(layer_hidden)
    return {
        "words": words,
        "text": text,
        "target_layer": int(target_layer),
        "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]),
        "vector": vector.numpy(),
        "abs_mean": float(vector.abs().mean().item()),
        "abs_max": float(vector.abs().max().item()),
    }


def extract_single_word_hidden_matrix(bundle, word: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one forward pass and return full [rows, hidden_dim] matrix for last token.

    Rows are embedding + each hidden layer output.
    """
    word = str(word).strip()
    tokenizer = bundle.tokenizer
    store = _get_hidden_store(bundle, config)
    token_ids = [int(x) for x in (tokenizer(word, add_special_tokens=False).get("input_ids") or [])]
    print(f"[single_word_hidden_state] word={word!r} token_ids={token_ids}", file=sys.stderr)
    if len(token_ids) != 1:
        print(
            f"[single_word_hidden_state] skip: single_token_required token_count={len(token_ids)} "
            f"tokens={tokenizer.convert_ids_to_tokens(token_ids)}",
            file=sys.stderr,
        )
        return {
            "ok": False,
            "word": word,
            "reason": "single_token_required",
            "token_count": int(len(token_ids)),
            "token_ids": token_ids,
            "matrix": [],
            "rows": 0,
            "cols": 0,
            "row_labels": [],
            "tokens": tokenizer.convert_ids_to_tokens(token_ids),
        }
    token_id = int(token_ids[0])
    hit = store.contains(token_id)
    print(
        f"[single_word_hidden_state] protocol={store.cfg.protocol} token_id={token_id} "
        f"store_contains={hit} data_file={store.cfg.data_file.name}",
        file=sys.stderr,
    )
    t0 = time.time()
    if hit:
        matrix = np.asarray(store.get_all_layers(token_id), dtype=np.float32)
        cache_source = "disk"
        print(f"[single_word_hidden_state] path=disk elapsed={time.time() - t0:.3f}s", file=sys.stderr)
    else:
        matrix = np.asarray(store.get_or_compute_layers(bundle, token_id), dtype=np.float32)
        # Ensure single-word path immediately persists done/data status on disk.
        store.flush()
        cache_source = "model"
        print(f"[single_word_hidden_state] path=model+writeback elapsed={time.time() - t0:.3f}s", file=sys.stderr)
    encoded = tokenizer(word, return_tensors="pt")
    row_labels = ["embedding"] + [f"layer_{idx}" for idx in range(1, matrix.shape[0])]
    return {
        "ok": True,
        "word": word,
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
        "row_labels": row_labels,
        "matrix": matrix.tolist(),
        "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]),
        "cache_source": cache_source,
    }


def extract_single_word_hidden_matrix_store_first(
    *,
    word: str,
    config: dict[str, Any],
    bundle_loader,
) -> dict[str, Any]:
    """Store-first path for UI hidden-state action.

    - If token is already cached in hidden_store: return from disk without model/GPU compute.
    - If token is missing: load model via bundle_loader and fallback to normal path.
    """
    from transformers import AutoConfig, AutoTokenizer

    normalized_word = str(word).strip()
    model_cfg = dict((config or {}).get("model") or {})
    model_path = model_cfg.get("model_name_or_path")
    tokenizer_path = model_cfg.get("tokenizer_name_or_path") or model_path
    if not model_path:
        raise ValueError("Config requires model.model_name_or_path")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    token_ids = [int(x) for x in (tokenizer(normalized_word, add_special_tokens=False).get("input_ids") or [])]
    print(f"[single_word_hidden_state] store_first word={normalized_word!r} token_ids={token_ids}", file=sys.stderr)
    if len(token_ids) != 1:
        return {
            "ok": False,
            "word": normalized_word,
            "reason": "single_token_required",
            "token_count": int(len(token_ids)),
            "token_ids": token_ids,
            "matrix": [],
            "rows": 0,
            "cols": 0,
            "row_labels": [],
            "tokens": tokenizer.convert_ids_to_tokens(token_ids),
        }

    # Build store config without requiring a loaded model: infer dims from model config file.
    merged = dict(config or {})
    hs_cfg = dict((merged.get("hidden_store") or {}))
    auto_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=bool(model_cfg.get("trust_remote_code", True)))
    if hs_cfg.get("n_layers") is None:
        hs_cfg["n_layers"] = int(getattr(auto_cfg, "num_hidden_layers")) + 1
    if hs_cfg.get("hidden_dim") is None:
        hs_cfg["hidden_dim"] = int(getattr(auto_cfg, "hidden_size"))
    merged["hidden_store"] = hs_cfg
    store_cfg = build_hidden_store_config(merged, bundle=None)
    store = _HIDDEN_STORE_RUNTIME.get_or_create(store_cfg, tokenizer)

    token_id = int(token_ids[0])
    if store.contains(token_id):
        matrix = np.asarray(store.get_all_layers(token_id), dtype=np.float32)
        return {
            "ok": True,
            "word": normalized_word,
            "rows": int(matrix.shape[0]),
            "cols": int(matrix.shape[1]),
            "row_labels": ["embedding"] + [f"layer_{idx}" for idx in range(1, matrix.shape[0])],
            "matrix": matrix.tolist(),
            "tokens": tokenizer.convert_ids_to_tokens(token_ids),
            "cache_source": "disk",
        }

    # Miss: only now allow model load + compute.
    bundle = bundle_loader()
    return extract_single_word_hidden_matrix(bundle, normalized_word, config=config)


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
