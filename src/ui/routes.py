from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

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
    cmd = [sys.executable, str(root / "main.py"), "--config", str(config_path), *command_args]
    started_at = time.time()
    try:
        completed = subprocess.run(
            cmd,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
        )
        output_root = root / "data" / "outputs"
        artifacts = collect_recent_artifacts(output_root, since_timestamp=started_at)
        csv_preview = newest_csv_preview(artifacts)
        return {
            "status": "ok" if completed.returncode == 0 else "error",
            "action": action.to_dict(),
            "command": cmd,
            "return_code": completed.returncode,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-12000:],
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
            "stdout": "",
            "stderr": "".join(traceback.format_exception(exc)),
            "artifacts": [],
            "csv_preview": None,
            "started_at": started_at,
            "finished_at": time.time(),
        }


def parse_json_body(raw_body: bytes) -> dict[str, Any]:
    if not raw_body:
        return {}
    return json.loads(raw_body.decode("utf-8"))
