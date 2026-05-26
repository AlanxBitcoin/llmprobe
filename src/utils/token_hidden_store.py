from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Store token hidden states in fixed-size binary files under data/cache.
# - Keep protocol-separated files (bos0_assistant0 / bos1_assistant0 / bos1_assistant1).
# - Support read-through behavior: cache hit returns directly; miss computes then writes back.
# - Support resumable build via per-token done map.

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


_ALLOWED_PROTOCOLS = {"bos0_assistant0", "bos1_assistant0", "bos1_assistant1"}
_BOS_INPUT_SYMBOL = "<BOS>"


@dataclass(frozen=True)
class HiddenStoreConfig:
    protocol: str
    data_file: Path
    progress_file: Path
    n_layers: int
    hidden_dim: int
    dtype: str
    preallocate: bool


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _storage_protocol_name(protocol: str) -> str:
    # Cache namespace versioning:
    # bos1_assistant0 now uses user-header context (BOS+start_header+user+end_header+word).
    if protocol == "bos1_assistant0":
        return "bos1_assistant0_userhdr"
    # bos1_assistant1 now aligns with full chat generation boundary token.
    if protocol == "bos1_assistant1":
        return "bos1_assistant1_chatgen"
    return protocol


def build_hidden_store_config(config: dict[str, Any] | None, bundle: Any | None = None) -> HiddenStoreConfig:
    store_cfg = (config or {}).get("hidden_store") or {}

    protocol = str(store_cfg.get("protocol", "bos1_assistant0"))
    if protocol not in _ALLOWED_PROTOCOLS:
        raise ValueError(f"Unsupported hidden_store protocol: {protocol!r}")
    if protocol == "bos0_assistant1":
        raise ValueError("Invalid protocol combination: bos0_assistant1")

    project_root = Path(__file__).resolve().parents[2]
    data_tpl = str(store_cfg.get("data_file", "data/cache/hidden_states.{protocol}.f16.bin"))
    done_tpl = str(store_cfg.get("progress_file", "data/cache/hidden_states.{protocol}.done.bin"))
    storage_protocol = _storage_protocol_name(protocol)
    data_file = _resolve_path(project_root, data_tpl.format(protocol=storage_protocol))
    progress_file = _resolve_path(project_root, done_tpl.format(protocol=storage_protocol))

    model_cfg = getattr(getattr(bundle, "model", None), "config", None)
    model_layers = getattr(model_cfg, "num_hidden_layers", None)
    model_hidden = getattr(model_cfg, "hidden_size", None)
    inferred_layers = int(model_layers) + 1 if model_layers is not None else None
    inferred_hidden = int(model_hidden) if model_hidden is not None else None

    explicit_layers = store_cfg.get("n_layers")
    explicit_hidden = store_cfg.get("hidden_dim")
    n_layers = int(explicit_layers) if explicit_layers is not None else inferred_layers
    hidden_dim = int(explicit_hidden) if explicit_hidden is not None else inferred_hidden
    if n_layers is None or hidden_dim is None:
        raise ValueError(
            "hidden_store dimension inference failed: provide bundle model config "
            "(num_hidden_layers/hidden_size) or set hidden_store.n_layers/hidden_dim explicitly."
        )

    return HiddenStoreConfig(
        protocol=protocol,
        data_file=data_file,
        progress_file=progress_file,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        dtype=str(store_cfg.get("dtype", "float16")),
        preallocate=bool(store_cfg.get("preallocate", True)),
    )


def protocol_from_flags(bos: bool, assistant: bool) -> str:
    if assistant and not bos:
        raise ValueError("assistant=true requires bos=true")
    if not bos:
        return "bos0_assistant0"
    if assistant:
        return "bos1_assistant1"
    return "bos1_assistant0"


def parse_token_ids_with_bos_alias(tokenizer, text: str) -> list[int]:
    """Tokenize text with add_special_tokens=False and support explicit BOS symbol input."""
    normalized = str(text).strip()
    if normalized.upper() == _BOS_INPUT_SYMBOL:
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            raise ValueError("Tokenizer has no bos_token_id but '<BOS>' was requested")
        return [int(bos_id)]
    return [int(x) for x in (tokenizer(normalized, add_special_tokens=False).get("input_ids") or [])]


def resolve_assistant_token_id(tokenizer) -> int:
    """Resolve assistant marker token id for BOS->word->assistant protocol."""
    start_ids = tokenizer("<|start_header_id|>", add_special_tokens=False).get("input_ids") or []
    end_ids = tokenizer("<|end_header_id|>", add_special_tokens=False).get("input_ids") or []
    if len(start_ids) == 1 and len(end_ids) == 1:
        start_id = int(start_ids[0])
        end_id = int(end_ids[0])
        try:
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=True,
                add_generation_prompt=True,
            )
            ids = chat.get("input_ids") if isinstance(chat, dict) else chat
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            if isinstance(ids, list):
                ids = [int(x) for x in ids]
                for i in range(len(ids) - 1):
                    if ids[i] == start_id:
                        for j in range(i + 1, len(ids)):
                            if ids[j] == end_id:
                                between = ids[i + 1 : j]
                                if len(between) == 1:
                                    return int(between[0])
                                break
        except Exception:
            pass

    fallback = tokenizer("assistant", add_special_tokens=False).get("input_ids") or []
    if len(fallback) != 1:
        raise ValueError(
            "Unable to resolve assistant marker token id. "
            "Expected tokenizer('assistant', add_special_tokens=False) to return one token."
        )
    return int(fallback[0])


def resolve_user_token_id(tokenizer) -> int:
    """Resolve user role token id for BOS+header+user+end_header context."""
    start_ids = tokenizer("<|start_header_id|>", add_special_tokens=False).get("input_ids") or []
    end_ids = tokenizer("<|end_header_id|>", add_special_tokens=False).get("input_ids") or []
    if len(start_ids) == 1 and len(end_ids) == 1:
        start_id = int(start_ids[0])
        end_id = int(end_ids[0])
        try:
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=True,
                add_generation_prompt=True,
            )
            ids = _normalize_chat_template_ids(chat)
            for i in range(len(ids) - 1):
                if ids[i] == start_id:
                    for j in range(i + 1, len(ids)):
                        if ids[j] == end_id:
                            between = ids[i + 1 : j]
                            if len(between) == 1:
                                return int(between[0])
                            break
        except Exception:
            pass

    fallback = tokenizer("user", add_special_tokens=False).get("input_ids") or []
    if len(fallback) != 1:
        raise ValueError(
            "Unable to resolve user role token id. "
            "Expected tokenizer('user', add_special_tokens=False) to return one token."
        )
    return int(fallback[0])


def _normalize_chat_template_ids(payload: Any) -> list[int]:
    if hasattr(payload, "get"):
        ids = payload.get("input_ids")
    else:
        ids = payload
    if ids is None:
        return []
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, tuple):
        ids = list(ids)
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list):
        return []
    return [int(x) for x in ids]


def _build_assistant_chat_prefix(tokenizer, token_ids: list[int]) -> list[int]:
    """Build BOS+user+content+eot+assistant_header(+pre-generation separator) sequence."""
    empty_template = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=True,
        add_generation_prompt=True,
    )
    ids = _normalize_chat_template_ids(empty_template)
    if not ids:
        raise ValueError("Chat template returned empty ids for assistant protocol")

    eot_id = tokenizer.eos_token_id
    if eot_id is None:
        eot_ids = tokenizer("<|eot_id|>", add_special_tokens=False).get("input_ids") or []
        if len(eot_ids) != 1:
            raise ValueError("Unable to resolve <|eot_id|> for assistant protocol")
        eot_id = int(eot_ids[0])

    try:
        eot_pos = ids.index(int(eot_id))
    except ValueError as exc:
        raise ValueError("Assistant protocol template missing <|eot_id|>") from exc

    prefix_before_eot = ids[:eot_pos]  # BOS + user header + user-content separator
    tail_from_eot = ids[eot_pos:]  # eot + assistant header (+ possible separator before generation)
    return prefix_before_eot + [int(x) for x in token_ids] + tail_from_eot


def _build_user_header_prefix(tokenizer, token_ids: list[int]) -> list[int]:
    """Build BOS+user-header+user-content-separator + token_ids."""
    empty_template = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=True,
        add_generation_prompt=True,
    )
    ids = _normalize_chat_template_ids(empty_template)
    if not ids:
        raise ValueError("Chat template returned empty ids for user-prefix protocol")
    eot_id = tokenizer.eos_token_id
    if eot_id is None:
        eot_ids = tokenizer("<|eot_id|>", add_special_tokens=False).get("input_ids") or []
        if len(eot_ids) != 1:
            raise ValueError("Unable to resolve <|eot_id|> for user-prefix protocol")
        eot_id = int(eot_ids[0])
    try:
        eot_pos = ids.index(int(eot_id))
    except ValueError as exc:
        raise ValueError("User-prefix protocol template missing <|eot_id|>") from exc
    prefix_before_eot = ids[:eot_pos]
    return prefix_before_eot + [int(x) for x in token_ids]


def build_protocol_input_ids(tokenizer, protocol: str, token_ids: list[int]) -> list[int]:
    if not token_ids:
        raise ValueError("token_ids is empty")
    ids = [int(x) for x in token_ids]
    if protocol == "bos0_assistant0":
        return ids
    if protocol == "bos1_assistant0":
        return _build_user_header_prefix(tokenizer, ids)
    if protocol == "bos1_assistant1":
        return _build_assistant_chat_prefix(tokenizer, ids)
    raise ValueError(f"Unsupported hidden_store protocol: {protocol!r}")


def build_store_for_protocol(
    bundle,
    config: dict[str, Any],
    *,
    bos: bool,
    assistant: bool,
    limit: int = 0,
    start_token_id: int = 0,
) -> dict[str, Any]:
    protocol = protocol_from_flags(bos=bos, assistant=assistant)
    merged = dict(config)
    store_cfg = dict((config or {}).get("hidden_store") or {})
    store_cfg["protocol"] = protocol
    merged["hidden_store"] = store_cfg
    cfg = build_hidden_store_config(merged, bundle=bundle)

    store = TokenHiddenStore(cfg, bundle.tokenizer)
    start = max(0, int(start_token_id))
    end = store.token_count
    if start >= end:
        return {
            "protocol": protocol,
            "token_count": store.token_count,
            "processed": 0,
            "written": 0,
            "skipped_done": 0,
            "start_token_id": start,
            "end_token_id": end,
        }

    processed = 0
    written = 0
    skipped_done = 0
    for token_id in range(start, end):
        if limit > 0 and processed >= limit:
            break
        processed += 1
        if store.contains(token_id):
            skipped_done += 1
            continue
        _ = store.get_or_compute_layers(bundle, token_id)
        written += 1
        if written % 128 == 0:
            store.flush()
        if written % 1000 == 0:
            print(f"[hidden_store] protocol={protocol} written={written} token_id={token_id}")
    store.flush()

    return {
        "protocol": protocol,
        "token_count": store.token_count,
        "processed": processed,
        "written": written,
        "skipped_done": skipped_done,
        "start_token_id": start,
        "end_token_id": end,
        "data_file": str(cfg.data_file),
        "progress_file": str(cfg.progress_file),
    }


class TokenHiddenStore:
    def __init__(self, cfg: HiddenStoreConfig, tokenizer) -> None:
        if cfg.dtype != "float16":
            raise ValueError(f"Only float16 is currently supported, got {cfg.dtype!r}")
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.dtype = np.float16
        self.value_size = np.dtype(self.dtype).itemsize
        self.record_size = self.cfg.n_layers * self.cfg.hidden_dim * self.value_size
        self.token_count = self._infer_token_count()
        self.cfg.data_file.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self._sequence_cache_dir = self.cfg.data_file.parent / f"hidden_sequences.{self.cfg.protocol}"
        self._sequence_cache_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_files()
        self._data_mem = np.memmap(
            self.cfg.data_file,
            mode="r+",
            dtype=self.dtype,
            shape=(self.token_count, self.cfg.n_layers, self.cfg.hidden_dim),
            order="C",
        )
        self._done_mem = np.memmap(
            self.cfg.progress_file,
            mode="r+",
            dtype=np.uint8,
            shape=(self.token_count,),
            order="C",
        )
        self._token_cache: dict[int, np.ndarray] = {}
        self._dirty_writes = 0

    def _infer_token_count(self) -> int:
        vocab = self.tokenizer.get_vocab()
        max_token_id = max(vocab.values())
        return int(max_token_id) + 1

    def _ensure_files(self) -> None:
        total_data_bytes = self.record_size * self.token_count
        total_done_bytes = self.token_count

        data_exists = self.cfg.data_file.exists()
        done_exists = self.cfg.progress_file.exists()

        if not data_exists:
            self.cfg.data_file.touch()
        if not done_exists:
            self.cfg.progress_file.touch()

        data_size = self.cfg.data_file.stat().st_size
        if data_size == 0:
            with self.cfg.data_file.open("r+b") as fh:
                if total_data_bytes > 0:
                    # Fast file sizing: set logical length without bulk writes.
                    fh.seek(total_data_bytes - 1)
                    fh.write(b"\x00")
        elif data_size != total_data_bytes:
            raise ValueError(
                "Hidden-store data file size mismatch. "
                f"expected={total_data_bytes}, actual={data_size}. "
                "Please delete the file and rebuild."
            )

        done_size = self.cfg.progress_file.stat().st_size
        if done_size == 0:
            with self.cfg.progress_file.open("r+b") as fh:
                if total_done_bytes > 0:
                    fh.seek(total_done_bytes - 1)
                    fh.write(b"\x00")
        elif done_size != total_done_bytes:
            raise ValueError(
                "Hidden-store progress file size mismatch. "
                f"expected={total_done_bytes}, actual={done_size}. "
                "Please delete the file and rebuild."
            )

    def contains(self, token_id: int) -> bool:
        self._validate_token_id(token_id)
        return bool(self._done_mem[token_id])

    def get_layer_vector(self, token_id: int, layer_idx: int) -> np.ndarray | None:
        self._validate_token_id(token_id)
        self._validate_layer(layer_idx)
        if not self.contains(token_id):
            return None
        return np.asarray(self._data_mem[token_id, layer_idx], dtype=np.float32)

    def get_hidden_state(self, bundle, token_id: int, layer_idx: int) -> np.ndarray:
        """Read-through entrypoint used by probe/study code paths."""
        self._validate_token_id(token_id)
        self._validate_layer(layer_idx)
        cached = self.get_layer_vector(token_id, layer_idx)
        if cached is not None:
            return cached
        _ = self.get_or_compute_layers(bundle, token_id)
        return np.asarray(self._data_mem[token_id, layer_idx], dtype=np.float32)

    def get_all_layers(self, token_id: int) -> np.ndarray | None:
        self._validate_token_id(token_id)
        cached_mem = self._token_cache.get(int(token_id))
        if cached_mem is not None:
            return np.asarray(cached_mem, dtype=np.float32)
        if not self.contains(token_id):
            return None
        data = np.asarray(self._data_mem[token_id], dtype=np.float32)
        self._token_cache[int(token_id)] = data
        return data

    def get_or_compute_layers(self, bundle, token_id: int) -> np.ndarray:
        cached = self.get_all_layers(token_id)
        if cached is not None:
            print(f"[hidden_store] token_id={token_id} cache=memory_or_disk")
            return cached

        # For BOS-required protocol, make BOS itself cache-resident first.
        # This avoids repeatedly treating BOS as an uncached prerequisite.
        if self.cfg.protocol == "bos1_assistant0":
            bos_id = self.tokenizer.bos_token_id
            if bos_id is None:
                raise ValueError("Tokenizer has no bos_token_id but protocol requires BOS")
            bos_id = int(bos_id)
            if int(token_id) != bos_id and not self.contains(bos_id):
                print(f"[hidden_store] token_id={token_id} ensure_bos_cache=miss->compute bos_id={bos_id}")
                _ = self.get_or_compute_layers(bundle, bos_id)
            elif int(token_id) != bos_id:
                print(f"[hidden_store] token_id={token_id} ensure_bos_cache=hit bos_id={bos_id}")

        # Recovery path: if data exists but done flag is stale, mark done and reuse.
        if self._recover_done_from_data(token_id):
            print(f"[hidden_store] token_id={token_id} cache=recovered_from_data")
            recovered = self.get_all_layers(token_id)
            if recovered is not None:
                return recovered
        print(f"[hidden_store] token_id={token_id} cache=miss compute=1")
        computed = self._compute_layers(bundle, token_id)
        self._write_layers(token_id, computed)
        return np.asarray(computed, dtype=np.float32)

    def get_or_compute_layers_batch(self, bundle, token_ids: list[int], *, flush: bool = True) -> dict[int, np.ndarray]:
        """Batch read-through for many single-token requests.

        Returns a dict keyed by token_id with float32 matrices [n_layers, hidden_dim].
        Cache hits are served immediately; misses are grouped into one GPU forward.
        """
        if not token_ids:
            return {}
        ordered_ids: list[int] = []
        seen: set[int] = set()
        for raw in token_ids:
            tid = int(raw)
            self._validate_token_id(tid)
            if tid in seen:
                continue
            seen.add(tid)
            ordered_ids.append(tid)

        out: dict[int, np.ndarray] = {}
        misses: list[int] = []
        for tid in ordered_ids:
            cached = self.get_all_layers(tid)
            if cached is not None:
                out[tid] = cached
            else:
                misses.append(tid)

        # For BOS-required protocol, ensure BOS can be reused by others.
        if self.cfg.protocol == "bos1_assistant0":
            bos_id = self.tokenizer.bos_token_id
            if bos_id is not None:
                bos_id = int(bos_id)
                if bos_id not in out and bos_id not in misses and any(t != bos_id for t in misses):
                    misses.insert(0, bos_id)

        remaining: list[int] = []
        for tid in misses:
            if self._recover_done_from_data(tid):
                recovered = self.get_all_layers(tid)
                if recovered is not None:
                    out[tid] = recovered
                    continue
            remaining.append(tid)

        if remaining:
            batch_inputs = [self._build_protocol_input_ids(tid) for tid in remaining]
            batch_layers = self._compute_layers_for_input_ids_batch(bundle, batch_inputs)
            for tid, layers in zip(remaining, batch_layers):
                self._write_layers(int(tid), np.asarray(layers, dtype=np.float32))
                out[int(tid)] = np.asarray(layers, dtype=np.float32)
            if flush:
                self.flush()

        return {tid: out[tid] for tid in ordered_ids if tid in out}

    def get_word_layers(self, bundle, word: str) -> np.ndarray:
        """Unified store entrypoint for word-level callers.

        - Single-token word: token-id table read-through (cache hit or compute+writeback)
        - Multi-token word: sequence cache read-through (cache hit or compute+writeback)
        """
        ids = parse_token_ids_with_bos_alias(self.tokenizer, word)
        if not ids:
            raise ValueError(f"Word produced empty token sequence: {word!r}")
        if len(ids) == 1:
            return self.get_or_compute_layers(bundle, int(ids[0]))
        return self.get_or_compute_sequence_layers(bundle, [int(x) for x in ids])

    def _compute_layers(self, bundle, token_id: int) -> np.ndarray:
        input_ids = self._build_protocol_input_ids(token_id)
        return self._compute_layers_for_input_ids(bundle, input_ids)

    def get_or_compute_sequence_layers(self, bundle, token_ids: list[int]) -> np.ndarray:
        cached = self._get_cached_sequence_layers(token_ids)
        if cached is not None:
            return cached
        input_ids = self._build_protocol_input_ids_from_list(token_ids)
        computed = self._compute_layers_for_input_ids(bundle, input_ids)
        self._write_cached_sequence_layers(token_ids, computed)
        return np.asarray(computed, dtype=np.float32)

    def _compute_layers_for_input_ids(self, bundle, input_ids: list[int]) -> np.ndarray:
        model = bundle.model
        device = next(model.parameters()).device
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(input_ids=input_tensor, output_hidden_states=True)

        raw_layers = outputs.hidden_states
        layers = np.zeros((self.cfg.n_layers, self.cfg.hidden_dim), dtype=np.float32)
        copy_layers = min(len(raw_layers), self.cfg.n_layers)
        for idx in range(copy_layers):
            vector = raw_layers[idx][0, -1, :].detach().float().cpu().numpy()
            if vector.shape[0] != self.cfg.hidden_dim:
                raise ValueError(
                    f"Hidden dim mismatch at layer {idx}: expected {self.cfg.hidden_dim}, got {vector.shape[0]}"
                )
            layers[idx] = vector
        return layers

    def _compute_layers_for_input_ids_batch(self, bundle, batch_input_ids: list[list[int]]) -> np.ndarray:
        if not batch_input_ids:
            return np.zeros((0, self.cfg.n_layers, self.cfg.hidden_dim), dtype=np.float32)
        model = bundle.model
        device = next(model.parameters()).device
        batch_size = len(batch_input_ids)
        lengths = [len(x) for x in batch_input_ids]
        max_len = max(lengths)
        if max_len <= 0:
            raise ValueError("batch_input_ids contains an empty sequence")

        pad_id = int(getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", 0) or 0)
        input_ids = torch.full((batch_size, max_len), fill_value=pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        for i, seq in enumerate(batch_input_ids):
            seq_ids = [int(x) for x in seq]
            seq_len = len(seq_ids)
            input_ids[i, :seq_len] = torch.tensor(seq_ids, dtype=torch.long, device=device)
            attention_mask[i, :seq_len] = 1

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        raw_layers = outputs.hidden_states
        layers = np.zeros((batch_size, self.cfg.n_layers, self.cfg.hidden_dim), dtype=np.float32)
        copy_layers = min(len(raw_layers), self.cfg.n_layers)
        last_positions = torch.tensor([x - 1 for x in lengths], dtype=torch.long, device=device)
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
        for idx in range(copy_layers):
            h = raw_layers[idx]
            selected = h[batch_indices, last_positions, :].detach().float().cpu().numpy()
            if selected.shape[1] != self.cfg.hidden_dim:
                raise ValueError(
                    f"Hidden dim mismatch at layer {idx}: expected {self.cfg.hidden_dim}, got {selected.shape[1]}"
                )
            layers[:, idx, :] = selected
        return layers

    @staticmethod
    def _normalize_chat_prefix_ids(prefix: Any) -> list[int]:
        """Normalize apply_chat_template outputs into a flat list[int]."""
        data = prefix
        if isinstance(data, dict):
            data = data.get("input_ids")
        if data is None:
            return []
        if hasattr(data, "tolist"):
            data = data.tolist()
        if isinstance(data, tuple):
            data = list(data)
        if not isinstance(data, list):
            raise ValueError(f"Unsupported chat template output type: {type(prefix)!r}")
        if data and isinstance(data[0], list):
            data = data[0]
        return [int(x) for x in data]

    def _build_protocol_input_ids(self, token_id: int) -> list[int]:
        return build_protocol_input_ids(self.tokenizer, self.cfg.protocol, [int(token_id)])

    def _build_protocol_input_ids_from_list(self, token_ids: list[int]) -> list[int]:
        return build_protocol_input_ids(self.tokenizer, self.cfg.protocol, [int(x) for x in token_ids])

    def _sequence_cache_paths(self, token_ids: list[int]) -> tuple[Path, Path]:
        payload = ",".join(str(int(x)) for x in token_ids).encode("utf-8")
        digest = hashlib.sha1(payload).hexdigest()
        base = self._sequence_cache_dir / digest
        return base.with_suffix(".f16.npy"), base.with_suffix(".json")

    def _get_cached_sequence_layers(self, token_ids: list[int]) -> np.ndarray | None:
        data_path, meta_path = self._sequence_cache_paths(token_ids)
        if not data_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if [int(x) for x in meta.get("token_ids", [])] != [int(x) for x in token_ids]:
            return None
        try:
            arr = np.load(data_path, allow_pickle=False)
        except OSError:
            return None
        if arr.shape != (self.cfg.n_layers, self.cfg.hidden_dim):
            return None
        return np.asarray(arr, dtype=np.float32)

    def _write_cached_sequence_layers(self, token_ids: list[int], layers: np.ndarray) -> None:
        data_path, meta_path = self._sequence_cache_paths(token_ids)
        data = np.asarray(layers, dtype=self.dtype)
        np.save(data_path, data, allow_pickle=False)
        meta = {"token_ids": [int(x) for x in token_ids], "protocol": self.cfg.protocol}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def _write_layers(self, token_id: int, layers: np.ndarray) -> None:
        self._validate_token_id(token_id)
        if layers.shape != (self.cfg.n_layers, self.cfg.hidden_dim):
            raise ValueError(f"Invalid layer shape: {layers.shape}")
        self._data_mem[token_id, :, :] = layers.astype(self.dtype, copy=False)
        self._done_mem[token_id] = 1
        self._token_cache[int(token_id)] = np.asarray(layers, dtype=np.float32)
        self._dirty_writes += 1

    def flush(self) -> None:
        if self._dirty_writes <= 0:
            return
        self._data_mem.flush()
        self._done_mem.flush()
        self._dirty_writes = 0

    def _recover_done_from_data(self, token_id: int) -> bool:
        self._validate_token_id(token_id)
        row = np.asarray(self._data_mem[token_id], dtype=np.float32)
        # Hidden states are practically never all-zero; this rescues stale done maps.
        if not np.any(row):
            return False
        self._done_mem[token_id] = 1
        self._done_mem.flush()
        self._token_cache[int(token_id)] = row
        return True

    def _validate_token_id(self, token_id: int) -> None:
        if token_id < 0 or token_id >= self.token_count:
            raise IndexError(f"token_id out of range: {token_id}")

    def _validate_layer(self, layer_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= self.cfg.n_layers:
            raise IndexError(f"layer_idx out of range: {layer_idx}")
