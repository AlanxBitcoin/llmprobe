from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Provide model loading + reuse APIs for long-lived runtime use.
# - Expose get_model_bundle(config, force_reload=False).
# - Reuse cached bundle when config is compatible; reload only when required.
# - Expose release_model_bundle() for explicit resource cleanup.

from pathlib import Path
import importlib.util
import site
import sys
import threading
from typing import Any


def _inject_user_site_packages() -> None:
    candidates: list[str] = []
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            candidates.append(user_site)
        else:
            candidates.extend(user_site)
    except Exception:
        pass

    version_tag = f"Python{sys.version_info.major}{sys.version_info.minor}"
    roaming = Path.home() / "AppData" / "Roaming" / "Python" / version_tag / "site-packages"
    candidates.append(str(roaming))

    for candidate in candidates:
        path = Path(candidate)
        try:
            should_add = path.exists()
        except PermissionError:
            should_add = True
        if should_add and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_inject_user_site_packages()

import torch


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_name, torch.float16)


def _accelerate_available() -> bool:
    return importlib.util.find_spec("accelerate") is not None


def _build_quantization_config(model_cfg: dict[str, Any]):
    load_in_4bit = bool(model_cfg.get("load_in_4bit"))
    load_in_8bit = bool(model_cfg.get("load_in_8bit"))
    if not load_in_4bit and not load_in_8bit:
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_compute_dtype=_resolve_dtype(model_cfg.get("torch_dtype", "float16")),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


class LocalModelBundle:
    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self._compat_signature: tuple[Any, ...] | None = None


_CACHED_BUNDLE: LocalModelBundle | None = None
_BUNDLE_LOCK = threading.RLock()


def _build_compat_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    model_cfg = dict((config or {}).get("model") or {})
    return (
        str(model_cfg.get("model_name_or_path", "")),
        str(model_cfg.get("tokenizer_name_or_path") or ""),
        str(model_cfg.get("device_map", "auto")),
        str(model_cfg.get("torch_dtype", "float16")),
        bool(model_cfg.get("load_in_4bit", False)),
        bool(model_cfg.get("load_in_8bit", False)),
        bool(model_cfg.get("trust_remote_code", True)),
    )


def load_local_model(config: dict[str, Any]) -> LocalModelBundle:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = config["model"]
    model_path = Path(model_cfg["model_name_or_path"])
    tokenizer_path = model_cfg.get("tokenizer_name_or_path") or str(model_path)
    has_cuda = torch.cuda.is_available()
    dtype_name = model_cfg.get("torch_dtype", "float16")
    resolved_dtype = _resolve_dtype(dtype_name)
    if not has_cuda and resolved_dtype in {torch.float16, torch.bfloat16}:
        resolved_dtype = torch.float32

    quantization_config = _build_quantization_config(model_cfg)
    if quantization_config is not None and not has_cuda:
        quantization_config = None

    requested_device_map = model_cfg.get("device_map", "auto")
    use_device_map = requested_device_map
    if not has_cuda or not _accelerate_available():
        use_device_map = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load tokenizer from the configured path. "
            "If you pointed at the default Ollama blob store, please replace "
            "`model_name_or_path` with a Hugging Face compatible local model directory."
        ) from exc
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model_kwargs = {
            "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
            "dtype": resolved_dtype,
            "quantization_config": quantization_config,
        }
        if use_device_map is not None:
            model_kwargs["device_map"] = use_device_map

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            **model_kwargs,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load model from the configured path. "
            "This project needs a Hugging Face compatible local model directory "
            "or model ID, not the raw Ollama blob cache layout."
        ) from exc
    if use_device_map is None:
        runtime_device = "cuda" if has_cuda else "cpu"
        model.to(runtime_device)
    model.eval()
    bundle = LocalModelBundle(tokenizer=tokenizer, model=model)
    bundle._compat_signature = _build_compat_signature(config)
    return bundle


def is_model_compatible(current: LocalModelBundle | None, config: dict[str, Any]) -> bool:
    if current is None:
        return False
    expected = _build_compat_signature(config)
    return getattr(current, "_compat_signature", None) == expected


def release_model_bundle() -> None:
    global _CACHED_BUNDLE
    with _BUNDLE_LOCK:
        bundle = _CACHED_BUNDLE
        _CACHED_BUNDLE = None
    if bundle is None:
        return
    try:
        del bundle.model
        del bundle.tokenizer
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model_bundle(config: dict[str, Any], force_reload: bool = False) -> LocalModelBundle:
    global _CACHED_BUNDLE
    with _BUNDLE_LOCK:
        if not force_reload and is_model_compatible(_CACHED_BUNDLE, config):
            return _CACHED_BUNDLE  # type: ignore[return-value]
        if _CACHED_BUNDLE is not None:
            release_model_bundle()
        _CACHED_BUNDLE = load_local_model(config)
        return _CACHED_BUNDLE
