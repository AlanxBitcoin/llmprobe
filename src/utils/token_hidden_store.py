from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Store token hidden states in fixed-size binary files under data/cache.
# - Keep protocol-separated files (bos0_assistant0 / bos1_assistant0 / bos1_assistant1).
# - Support read-through behavior: cache hit returns directly; miss computes then writes back.
# - Support resumable build via per-token done map.

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


_ALLOWED_PROTOCOLS = {"bos0_assistant0", "bos1_assistant0", "bos1_assistant1"}


@dataclass(frozen=True)
class HiddenStoreConfig:
    enabled: bool
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


def build_hidden_store_config(config: dict[str, Any]) -> HiddenStoreConfig | None:
    store_cfg = (config or {}).get("hidden_store") or {}
    if not bool(store_cfg.get("enabled", False)):
        return None

    protocol = str(store_cfg.get("protocol", "bos1_assistant0"))
    if protocol not in _ALLOWED_PROTOCOLS:
        raise ValueError(f"Unsupported hidden_store protocol: {protocol!r}")
    if protocol == "bos0_assistant1":
        raise ValueError("Invalid protocol combination: bos0_assistant1")

    project_root = Path(__file__).resolve().parents[2]
    data_tpl = str(store_cfg.get("data_file", "data/cache/hidden_states.{protocol}.f16.bin"))
    done_tpl = str(store_cfg.get("progress_file", "data/cache/hidden_states.{protocol}.done.bin"))
    data_file = _resolve_path(project_root, data_tpl.format(protocol=protocol))
    progress_file = _resolve_path(project_root, done_tpl.format(protocol=protocol))

    return HiddenStoreConfig(
        enabled=True,
        protocol=protocol,
        data_file=data_file,
        progress_file=progress_file,
        n_layers=int(store_cfg.get("n_layers", 36)),
        hidden_dim=int(store_cfg.get("hidden_dim", 4096)),
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
    store_cfg["enabled"] = True
    store_cfg["protocol"] = protocol
    merged["hidden_store"] = store_cfg
    cfg = build_hidden_store_config(merged)
    if cfg is None:
        raise ValueError("failed to build hidden_store config")

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
        if written % 1000 == 0:
            print(f"[hidden_store] protocol={protocol} written={written} token_id={token_id}")

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

    def _infer_token_count(self) -> int:
        vocab = self.tokenizer.get_vocab()
        max_token_id = max(vocab.values())
        return int(max_token_id) + 1

    def _ensure_files(self) -> None:
        total_data_bytes = self.record_size * self.token_count
        total_done_bytes = self.token_count

        if not self.cfg.data_file.exists():
            self.cfg.data_file.touch()
        if not self.cfg.progress_file.exists():
            self.cfg.progress_file.touch()

        if self.cfg.preallocate or self.cfg.data_file.stat().st_size != total_data_bytes:
            with self.cfg.data_file.open("r+b") as fh:
                fh.truncate(total_data_bytes)

        if self.cfg.progress_file.stat().st_size != total_done_bytes:
            with self.cfg.progress_file.open("r+b") as fh:
                fh.truncate(total_done_bytes)
                fh.seek(0)
                fh.write(b"\x00" * total_done_bytes)

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
        if not self.contains(token_id):
            return None
        return np.asarray(self._data_mem[token_id], dtype=np.float32)

    def get_or_compute_layers(self, bundle, token_id: int) -> np.ndarray:
        cached = self.get_all_layers(token_id)
        if cached is not None:
            return cached
        computed = self._compute_layers(bundle, token_id)
        self._write_layers(token_id, computed)
        return np.asarray(computed, dtype=np.float32)

    def _compute_layers(self, bundle, token_id: int) -> np.ndarray:
        model = bundle.model
        device = next(model.parameters()).device
        input_ids = self._build_protocol_input_ids(token_id)
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

    def _build_protocol_input_ids(self, token_id: int) -> list[int]:
        protocol = self.cfg.protocol
        if protocol == "bos0_assistant0":
            return [token_id]

        if protocol == "bos1_assistant0":
            bos_id = self.tokenizer.bos_token_id
            if bos_id is None:
                raise ValueError("Tokenizer has no bos_token_id but protocol requires BOS")
            return [int(bos_id), token_id]

        # bos1_assistant1
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support apply_chat_template for assistant protocol")
        prefix = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if not prefix:
            raise ValueError("Chat template returned empty prefix")
        return [int(x) for x in prefix] + [token_id]

    def _write_layers(self, token_id: int, layers: np.ndarray) -> None:
        self._validate_token_id(token_id)
        if layers.shape != (self.cfg.n_layers, self.cfg.hidden_dim):
            raise ValueError(f"Invalid layer shape: {layers.shape}")
        self._data_mem[token_id, :, :] = layers.astype(self.dtype, copy=False)
        self._data_mem.flush()
        self._done_mem[token_id] = 1
        self._done_mem.flush()

    def _validate_token_id(self, token_id: int) -> None:
        if token_id < 0 or token_id >= self.token_count:
            raise IndexError(f"token_id out of range: {token_id}")

    def _validate_layer(self, layer_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= self.cfg.n_layers:
            raise IndexError(f"layer_idx out of range: {layer_idx}")
