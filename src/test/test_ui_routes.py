from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Verify UI route dispatch payload shape and command mapping behavior.
# - Keep tests lightweight with monkeypatched execution helpers.

from pathlib import Path

from src.ui.routes import actions_payload, execute_ui_action, parse_json_body


def test_actions_payload_returns_action_list():
    payload = actions_payload()
    assert "actions" in payload
    assert isinstance(payload["actions"], list)
    assert payload["actions"]


def test_execute_ui_action_dispatch(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def _fake_run_cli_command(config_path, command_args):
        captured["config_path"] = str(config_path)
        captured["command_args"] = list(command_args)
        return 0

    monkeypatch.setattr("src.ui.routes.run_cli_command", _fake_run_cli_command)
    monkeypatch.setattr("src.ui.routes.collect_recent_artifacts", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("src.ui.routes.newest_csv_preview", lambda *_args, **_kwargs: None)

    result = execute_ui_action(
        action_id="study_single_word",
        params={"word": "apple"},
        project_root=tmp_path,
        config_path="configs/custom.yaml",
    )

    assert result["status"] == "ok"
    assert captured["config_path"] == "configs/custom.yaml"
    assert captured["command_args"] == ["run-single-word", "apple"]


def test_parse_json_body():
    payload = parse_json_body(b'{"a": 1, "b": "x"}')
    assert payload == {"a": 1, "b": "x"}
