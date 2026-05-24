from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Verify model loader cache reuse and force-reload behavior.
# - Keep tests lightweight via monkeypatch/mocks.

from dataclasses import dataclass

from src import model_loader


@dataclass
class _DummyBundle:
    tokenizer: object
    model: object
    _compat_signature: tuple | None = None


def test_get_model_bundle_reuses_cached_model(monkeypatch):
    calls = {"count": 0}
    model_loader._CACHED_BUNDLE = None

    def _fake_load(config):
        calls["count"] += 1
        return _DummyBundle(
            tokenizer=object(),
            model=object(),
            _compat_signature=model_loader._build_compat_signature(config),
        )

    monkeypatch.setattr(model_loader, "load_local_model", _fake_load)
    config = {"model": {"model_name_or_path": "m", "torch_dtype": "float16", "device_map": "auto"}}

    first = model_loader.get_model_bundle(config)
    second = model_loader.get_model_bundle(config)

    assert first is second
    assert calls["count"] == 1


def test_get_model_bundle_force_reload(monkeypatch):
    calls = {"count": 0}
    model_loader._CACHED_BUNDLE = None

    def _fake_load(config):
        calls["count"] += 1
        return _DummyBundle(
            tokenizer=object(),
            model=object(),
            _compat_signature=model_loader._build_compat_signature(config),
        )

    monkeypatch.setattr(model_loader, "load_local_model", _fake_load)
    monkeypatch.setattr(model_loader.torch.cuda, "is_available", lambda: False)
    config = {"model": {"model_name_or_path": "m", "torch_dtype": "float16", "device_map": "auto"}}

    _ = model_loader.get_model_bundle(config)
    _ = model_loader.get_model_bundle(config, force_reload=True)

    assert calls["count"] == 2
