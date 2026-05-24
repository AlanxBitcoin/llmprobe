from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Runtime service entry for model lifecycle and dispatch.
# - Must expose start_llama_api / get_runtime_api / execute_model_call / shutdown_llama_api.
# - All upper layers should access model bundle through this runtime boundary.

from dataclasses import dataclass
import threading
from typing import Any

from .model_loader import get_model_bundle, release_model_bundle


@dataclass(frozen=True)
class RuntimeRequest:
    config: dict[str, Any]
    force_reload: bool = False


@dataclass(frozen=True)
class RuntimeResult:
    bundle: Any


class LlamaRuntimeAPI:
    def __init__(self, config: dict[str, Any]) -> None:
        self._last_config = config
        self._bundle = get_model_bundle(config, force_reload=False)

    def execute_model_call(self, request: RuntimeRequest) -> RuntimeResult:
        self._last_config = request.config
        self._bundle = get_model_bundle(request.config, force_reload=request.force_reload)
        return RuntimeResult(bundle=self._bundle)

    def get_bundle(self):
        return self._bundle


_RUNTIME_API: LlamaRuntimeAPI | None = None
_RUNTIME_LOCK = threading.RLock()


def start_llama_api(config: dict[str, Any]) -> LlamaRuntimeAPI:
    global _RUNTIME_API
    with _RUNTIME_LOCK:
        _RUNTIME_API = LlamaRuntimeAPI(config)
        return _RUNTIME_API


def get_runtime_api() -> LlamaRuntimeAPI:
    with _RUNTIME_LOCK:
        if _RUNTIME_API is None:
            raise RuntimeError("Llama runtime API is not started. Call start_llama_api(config) first.")
        return _RUNTIME_API


def execute_model_call(request: RuntimeRequest) -> RuntimeResult:
    api = get_runtime_api()
    return api.execute_model_call(request)


def shutdown_llama_api() -> None:
    global _RUNTIME_API
    with _RUNTIME_LOCK:
        _RUNTIME_API = None
    release_model_bundle()
