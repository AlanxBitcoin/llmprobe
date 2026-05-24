from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Map UI action requests to study/probe execution entrypoints.
# - Return execution status, logs, and artifact previews for UI rendering.
# - Keep execution orchestration in UI layer, not model internals.

import contextlib
import io
import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from src.main import run_cli_command

from .registry import build_command_args, get_ui_action, list_ui_actions
from .result_render import collect_recent_artifacts, newest_csv_preview

_RUN_LOCK = threading.Lock()
_RUN_STATE: dict[str, Any] = {"action_id": None, "started_at": 0.0, "eta_seconds": 0.0}
_ACTION_DURATION_SECONDS: dict[str, float] = {}


def actions_payload() -> dict[str, Any]:
    return {"actions": list_ui_actions()}


def execute_ui_action(
    action_id: str,
    params: dict[str, Any] | None = None,
    *,
    project_root: str | Path,
    config_path: str | Path,
    timeout_seconds: int = 0,
) -> dict[str, Any]:
    params = params or {}
    root = Path(project_root).resolve()
    action = get_ui_action(action_id)
    command_args = build_command_args(action, params)
    cmd = ["inprocess", "src.main", "--config", str(config_path), *command_args]
    started_at = time.time()
    eta_seconds = _estimate_duration_seconds(action.id)

    if not _RUN_LOCK.acquire(blocking=False):
        running_for = max(0.0, time.time() - float(_RUN_STATE.get("started_at") or time.time()))
        current_eta = float(_RUN_STATE.get("eta_seconds") or 0.0)
        remaining = max(0.0, current_eta - running_for)
        return {
            "status": "busy",
            "action": action.to_dict(),
            "command": cmd,
            "return_code": None,
            "stdout": "",
            "stderr": "Another task is running. Please wait for it to finish.",
            "artifacts": [],
            "csv_preview": None,
            "hidden_state_heatmap": None,
            "busy": True,
            "running_action_id": _RUN_STATE.get("action_id"),
            "running_for_seconds": round(running_for, 1),
            "estimated_remaining_seconds": round(remaining, 1),
            "started_at": started_at,
            "finished_at": time.time(),
        }

    _RUN_STATE["action_id"] = action.id
    _RUN_STATE["started_at"] = started_at
    _RUN_STATE["eta_seconds"] = eta_seconds
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with _working_directory(root), contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            if timeout_seconds > 0:
                print(
                    f"[ui] timeout_seconds={timeout_seconds} is ignored in in-process mode.",
                    file=stderr_buffer,
                )
            command_result = run_cli_command(config_path=config_path, command_args=command_args)
        return_code = 0
        # Hidden-state heatmap action returns payload directly; scanning large outputs
        # directories here can block UI response for a long time.
        if action.command in {"run-single-word-hidden-state", "run-single-word-top-100-neurons"}:
            artifacts = []
            csv_preview = None
        else:
            output_root = root / "data" / "outputs"
            artifacts = collect_recent_artifacts(output_root, since_timestamp=started_at)
            csv_preview = newest_csv_preview(artifacts)
        return {
            "status": "ok" if return_code == 0 else "error",
            "action": action.to_dict(),
            "command": cmd,
            "return_code": return_code,
            "stdout": stdout_buffer.getvalue()[-12000:],
            "stderr": stderr_buffer.getvalue()[-12000:],
            "artifacts": artifacts,
            "csv_preview": csv_preview,
            "hidden_state_heatmap": (command_result or {}).get("hidden_state_heatmap") if isinstance(command_result, dict) else None,
            "started_at": started_at,
            "finished_at": time.time(),
        }
    except Exception as exc:  # noqa: BLE001 - UI should return diagnostics.
        return {
            "status": "error",
            "action": action.to_dict(),
            "command": cmd,
            "return_code": None,
            "stdout": stdout_buffer.getvalue()[-12000:],
            "stderr": (stderr_buffer.getvalue() + "".join(traceback.format_exception(exc)))[-12000:],
            "artifacts": [],
            "csv_preview": None,
            "started_at": started_at,
            "finished_at": time.time(),
        }
    finally:
        elapsed = max(0.0, time.time() - started_at)
        _remember_duration_seconds(action.id, elapsed)
        _RUN_STATE["action_id"] = None
        _RUN_STATE["started_at"] = 0.0
        _RUN_STATE["eta_seconds"] = 0.0
        _RUN_LOCK.release()


def parse_json_body(raw_body: bytes) -> dict[str, Any]:
    if not raw_body:
        return {}
    return json.loads(raw_body.decode("utf-8"))


def _estimate_duration_seconds(action_id: str) -> float:
    remembered = _ACTION_DURATION_SECONDS.get(action_id)
    if remembered is not None:
        return max(3.0, float(remembered))
    # Conservative fallback for first run.
    if action_id == "study_single_word_hidden_state":
        return 20.0
    return 30.0


def _remember_duration_seconds(action_id: str, elapsed: float) -> None:
    prev = _ACTION_DURATION_SECONDS.get(action_id)
    if prev is None:
        _ACTION_DURATION_SECONDS[action_id] = float(elapsed)
        return
    # EWMA to smooth countdown estimate.
    _ACTION_DURATION_SECONDS[action_id] = prev * 0.7 + float(elapsed) * 0.3


@contextlib.contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)
