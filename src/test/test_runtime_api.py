from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Verify runtime API start/dispatch/shutdown flow.
# - Use mocks to avoid real model initialization in tests.

from src import runtime_api


def test_runtime_api_start_execute_and_shutdown(monkeypatch):
    events: list[tuple] = []

    def _fake_get_model_bundle(config, force_reload=False):
        events.append(("get", config.get("name"), force_reload))
        return {"bundle": config.get("name"), "force_reload": force_reload}

    def _fake_release():
        events.append(("release",))

    monkeypatch.setattr(runtime_api, "get_model_bundle", _fake_get_model_bundle)
    monkeypatch.setattr(runtime_api, "release_model_bundle", _fake_release)
    runtime_api._RUNTIME_API = None

    api = runtime_api.start_llama_api({"name": "boot"})
    assert runtime_api.get_runtime_api() is api

    result = runtime_api.execute_model_call(runtime_api.RuntimeRequest(config={"name": "call"}, force_reload=True))
    assert result.bundle["bundle"] == "call"
    assert result.bundle["force_reload"] is True

    runtime_api.shutdown_llama_api()
    assert ("release",) in events
