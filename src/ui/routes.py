from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Map UI action requests to study/probe execution entrypoints.
# - Return execution status, logs, and artifact previews for UI rendering.
# - Keep execution orchestration in UI layer, not model internals.

import contextlib
import io
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from src.main import run_cli_command

from .registry import build_command_args, get_ui_action, list_ui_actions
from .result_render import collect_recent_artifacts, newest_csv_preview


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
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with _working_directory(root), contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            if timeout_seconds > 0:
                print(
                    f"[ui] timeout_seconds={timeout_seconds} is ignored in in-process mode.",
                    file=stderr_buffer,
                )
            run_cli_command(config_path=config_path, command_args=command_args)
        return_code = 0
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


def parse_json_body(raw_body: bytes) -> dict[str, Any]:
    if not raw_body:
        return {}
    return json.loads(raw_body.decode("utf-8"))


@contextlib.contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(previous)
