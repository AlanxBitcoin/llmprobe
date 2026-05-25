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
import uuid
from pathlib import Path
from typing import Any

from src.main import run_cli_command
from src.config import load_config
from src.runtime_api import RuntimeRequest, get_runtime_api, start_llama_api
import torch

from .registry import build_command_args, get_ui_action, list_ui_actions
from .result_render import collect_recent_artifacts, newest_csv_preview

_RUN_LOCK = threading.Lock()
_RUN_STATE: dict[str, Any] = {"action_id": None, "started_at": 0.0, "eta_seconds": 0.0}
_ACTION_DURATION_SECONDS: dict[str, float] = {}
_TASKS_LOCK = threading.Lock()
_TASKS: dict[str, dict[str, Any]] = {}


def _batch_cache_path(project_root: Path) -> Path:
    return (project_root / "data" / "cache" / "batch_words.json").resolve()


def load_batch_cache(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).resolve()
    path = _batch_cache_path(root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in data.items():
        name = str(k).strip()
        words_csv = str(v).strip()
        if name and words_csv:
            out[name] = words_csv
    return out


def save_batch_cache(project_root: str | Path, mapping: dict[str, str]) -> None:
    root = Path(project_root).resolve()
    path = _batch_cache_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(k): str(v) for k, v in mapping.items() if str(k).strip() and str(v).strip()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_batch_mapping(project_root: str | Path, *, batch_name: str, words_csv: str) -> None:
    name = str(batch_name).strip()
    words = str(words_csv).strip()
    if not name or not words:
        return
    data = load_batch_cache(project_root)
    data[name] = words
    save_batch_cache(project_root, data)


def batch_options_payload(project_root: str | Path) -> dict[str, Any]:
    data = load_batch_cache(project_root)
    items = [{"name": k, "words_csv": data[k]} for k in sorted(data.keys(), key=lambda x: x.lower())]
    return {"batches": items}


def actions_payload() -> dict[str, Any]:
    return {"actions": list_ui_actions()}


def execute_chat_completion(
    messages: list[dict[str, Any]] | None,
    *,
    config_path: str | Path,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> dict[str, Any]:
    """Run one chat turn against the currently loaded model bundle."""
    started_at = time.time()
    if not _RUN_LOCK.acquire(blocking=False):
        running_for = max(0.0, time.time() - float(_RUN_STATE.get("started_at") or time.time()))
        return {
            "status": "busy",
            "error": "Another task is running. Please wait for it to finish.",
            "running_action_id": _RUN_STATE.get("action_id"),
            "running_for_seconds": round(running_for, 1),
            "started_at": started_at,
            "finished_at": time.time(),
        }

    _RUN_STATE["action_id"] = "chat_completion"
    _RUN_STATE["started_at"] = started_at
    _RUN_STATE["eta_seconds"] = 30.0
    try:
        config = load_config(config_path)
        try:
            api = get_runtime_api()
        except RuntimeError:
            api = start_llama_api(config)

        bundle = api.execute_model_call(RuntimeRequest(config=config, force_reload=False)).bundle
        tokenizer = bundle.tokenizer
        model = bundle.model

        normalized: list[dict[str, str]] = []
        for item in messages or []:
            role = str((item or {}).get("role") or "").strip().lower()
            content = str((item or {}).get("content") or "").strip()
            if role not in {"system", "user", "assistant"} or not content:
                continue
            normalized.append({"role": role, "content": content})
        if not normalized:
            return {
                "status": "error",
                "error": "No valid chat messages were provided.",
                "started_at": started_at,
                "finished_at": time.time(),
            }

        chat_payload = tokenizer.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(chat_payload, "get"):
            input_ids = chat_payload.get("input_ids")
            attention_mask = chat_payload.get("attention_mask")
        else:
            input_ids = chat_payload
            attention_mask = None
        if input_ids is None:
            raise ValueError("Tokenizer did not return input_ids for chat template")
        if not torch.is_tensor(input_ids):
            input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and not torch.is_tensor(attention_mask):
            attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        device = next(model.parameters()).device
        input_ids = input_ids.to(device=device, dtype=torch.long)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device, dtype=torch.long)

        safe_max_new_tokens = int(max(1, min(int(max_new_tokens), 1024)))
        safe_temperature = float(temperature)
        safe_top_p = float(top_p)
        do_sample = safe_temperature > 0.0
        if do_sample:
            safe_temperature = max(0.05, min(safe_temperature, 5.0))
            safe_top_p = max(0.05, min(safe_top_p, 1.0))

        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": safe_max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
            "use_cache": True,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if do_sample:
            gen_kwargs["temperature"] = safe_temperature
            gen_kwargs["top_p"] = safe_top_p

        with torch.no_grad():
            output_ids = model.generate(**gen_kwargs)
        prompt_len = int(input_ids.shape[1])
        new_ids = output_ids[0, prompt_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        if not text:
            text = tokenizer.decode(new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()

        return {
            "status": "ok",
            "assistant_message": text,
            "generated_token_count": int(new_ids.shape[0]),
            "generation": {
                "do_sample": bool(do_sample),
                "temperature": safe_temperature if do_sample else 0.0,
                "top_p": safe_top_p if do_sample else 1.0,
                "max_new_tokens": safe_max_new_tokens,
            },
            "started_at": started_at,
            "finished_at": time.time(),
        }
    except Exception as exc:  # noqa: BLE001 - chat endpoint should return diagnostics.
        return {
            "status": "error",
            "error": "".join(traceback.format_exception(exc))[-12000:],
            "started_at": started_at,
            "finished_at": time.time(),
        }
    finally:
        _RUN_STATE["action_id"] = None
        _RUN_STATE["started_at"] = 0.0
        _RUN_STATE["eta_seconds"] = 0.0
        _RUN_LOCK.release()


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
    if action.command == "run-single-word-hidden-state-batch-average":
        upsert_batch_mapping(
            root,
            batch_name=str(params.get("batch_name") or ""),
            words_csv=str(params.get("words_csv") or ""),
        )
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
        if action.command in {
            "run-single-word-hidden-state",
            "run-single-word-hidden-state-batch-average",
            "run-single-word-top-100-neurons",
        }:
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


def start_ui_action_task(
    action_id: str,
    params: dict[str, Any] | None = None,
    *,
    project_root: str | Path,
    config_path: str | Path,
    timeout_seconds: int = 0,
) -> dict[str, Any]:
    """Start a background UI action task and return task id immediately."""
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
            "task_id": None,
            "running_action_id": _RUN_STATE.get("action_id"),
            "running_for_seconds": round(running_for, 1),
            "estimated_remaining_seconds": round(remaining, 1),
            "started_at": started_at,
            "finished_at": time.time(),
        }

    task_id = uuid.uuid4().hex
    with _TASKS_LOCK:
        _TASKS[task_id] = {
            "status": "running",
            "task_id": task_id,
            "action_id": action.id,
            "started_at": started_at,
            "updated_at": started_at,
            "eta_seconds": eta_seconds,
            "result": None,
            "error": None,
        }
    _RUN_STATE["action_id"] = action.id
    _RUN_STATE["started_at"] = started_at
    _RUN_STATE["eta_seconds"] = eta_seconds

    def _worker() -> None:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            if action.command == "run-single-word-hidden-state-batch-average":
                upsert_batch_mapping(
                    root,
                    batch_name=str(params.get("batch_name") or ""),
                    words_csv=str(params.get("words_csv") or ""),
                )
            with _working_directory(root), contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                if timeout_seconds > 0:
                    print(
                        f"[ui] timeout_seconds={timeout_seconds} is ignored in in-process mode.",
                        file=stderr_buffer,
                    )
                command_result = run_cli_command(config_path=config_path, command_args=command_args)

            if action.command in {
                "run-single-word-hidden-state",
                "run-single-word-hidden-state-batch-average",
                "run-single-word-top-100-neurons",
                "run-layer-neuron-logits-table",
            }:
                artifacts = []
                csv_preview = None
            else:
                output_root = root / "data" / "outputs"
                artifacts = collect_recent_artifacts(output_root, since_timestamp=started_at)
                csv_preview = newest_csv_preview(artifacts)

            result = {
                "status": "ok",
                "action": action.to_dict(),
                "command": cmd,
                "return_code": 0,
                "stdout": stdout_buffer.getvalue()[-12000:],
                "stderr": stderr_buffer.getvalue()[-12000:],
                "artifacts": artifacts,
                "csv_preview": csv_preview,
                "hidden_state_heatmap": (command_result or {}).get("hidden_state_heatmap") if isinstance(command_result, dict) else None,
                "started_at": started_at,
                "finished_at": time.time(),
                "task_id": task_id,
            }
            with _TASKS_LOCK:
                if task_id in _TASKS:
                    _TASKS[task_id]["status"] = "ok"
                    _TASKS[task_id]["updated_at"] = time.time()
                    _TASKS[task_id]["result"] = result
        except Exception as exc:  # noqa: BLE001
            result = {
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
                "task_id": task_id,
            }
            with _TASKS_LOCK:
                if task_id in _TASKS:
                    _TASKS[task_id]["status"] = "error"
                    _TASKS[task_id]["updated_at"] = time.time()
                    _TASKS[task_id]["result"] = result
                    _TASKS[task_id]["error"] = str(exc)
        finally:
            elapsed = max(0.0, time.time() - started_at)
            _remember_duration_seconds(action.id, elapsed)
            _RUN_STATE["action_id"] = None
            _RUN_STATE["started_at"] = 0.0
            _RUN_STATE["eta_seconds"] = 0.0
            _RUN_LOCK.release()

    thread = threading.Thread(target=_worker, name=f"ui-task-{task_id[:8]}", daemon=True)
    thread.start()
    return {
        "status": "accepted",
        "task_id": task_id,
        "action": action.to_dict(),
        "command": cmd,
        "started_at": started_at,
        "estimated_remaining_seconds": round(eta_seconds, 1),
    }


def get_ui_action_task(task_id: str) -> dict[str, Any]:
    with _TASKS_LOCK:
        task = _TASKS.get(str(task_id))
    if not task:
        return {"status": "not_found", "task_id": str(task_id)}

    status = str(task.get("status") or "running")
    if status in {"ok", "error"} and isinstance(task.get("result"), dict):
        return dict(task["result"])

    started_at = float(task.get("started_at") or time.time())
    eta_seconds = float(task.get("eta_seconds") or 0.0)
    running_for = max(0.0, time.time() - started_at)
    remaining = max(0.0, eta_seconds - running_for)
    return {
        "status": "running",
        "task_id": str(task_id),
        "action_id": task.get("action_id"),
        "started_at": started_at,
        "running_for_seconds": round(running_for, 1),
        "estimated_remaining_seconds": round(remaining, 1),
        "updated_at": float(task.get("updated_at") or started_at),
    }


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
