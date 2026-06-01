from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Host local web UI server and static/template/output routes.
# - UI layer handles interaction only; no probe training logic here.
# - All file serving remains under project-root-safe paths.

import json
import mimetypes
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.parse import parse_qs

from .routes import (
    actions_payload,
    attribute_groups_payload,
    batch_options_payload,
    delete_batch_mapping,
    upsert_batch_mapping,
    execute_chat_completion,
    execute_ui_action,
    get_ui_action_task,
    layer_neurons_list_payload,
    list_ffn_neuron_history,
    load_latest_ffn_neuron_history_result,
    load_ffn_neuron_history_result,
    parse_json_body,
    start_ui_action_task,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def run_ui_server(
    config: dict[str, Any] | None = None,
    *,
    project_root: str | Path | None = None,
    config_path: str | Path = "configs/custom.yaml",
    host: str | None = None,
    port: int | None = None,
) -> None:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    ui_cfg = (config or {}).get("ui", {})
    host = host or ui_cfg.get("host") or DEFAULT_HOST
    port = int(port or ui_cfg.get("port") or DEFAULT_PORT)
    # Default to built-in threaded server for responsiveness under heavy local jobs.
    prefer_builtin = bool(ui_cfg.get("prefer_builtin_threading", True))
    if prefer_builtin:
        _run_builtin_server(root.resolve(), Path(config_path), host=str(host), port=int(port))
        return
    try:
        _run_fastapi_server(root.resolve(), Path(config_path), host=str(host), port=int(port))
        return
    except Exception as exc:  # noqa: BLE001 - fall back to built-in server for local robustness.
        print(f"[ui] FastAPI server unavailable, fallback to built-in server: {exc}")
    _run_builtin_server(root.resolve(), Path(config_path), host=str(host), port=int(port))


def _run_fastapi_server(project_root: Path, config_path: Path, *, host: str, port: int) -> None:
    from fastapi import Body, FastAPI
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    import asyncio
    import uvicorn

    template_dir = project_root / "src" / "ui" / "templates"
    static_dir = project_root / "src" / "ui" / "static"
    outputs_dir = project_root / "data" / "outputs"

    app = FastAPI(title="LLM Probe UI", docs_url=None, redoc_url=None)

    @app.get("/")
    async def index():
        path = template_dir / "index.html"
        if not path.exists():
            return JSONResponse({"error": "File not found"}, status_code=404)
        return FileResponse(path, media_type="text/html; charset=utf-8")

    @app.get("/popup")
    async def popup_page():
        path = template_dir / "popup.html"
        if not path.exists():
            return JSONResponse({"error": "File not found"}, status_code=404)
        return FileResponse(path, media_type="text/html; charset=utf-8")

    @app.get("/api/actions")
    async def api_actions():
        return JSONResponse(actions_payload())

    @app.get("/api/batches")
    async def api_batches():
        return JSONResponse(batch_options_payload(project_root))

    @app.post("/api/batches/upsert")
    async def api_batches_upsert(payload: dict[str, Any] | None = Body(default=None)):
        payload = payload or {}
        upsert_batch_mapping(
            project_root,
            batch_name=str(payload.get("batch_name") or ""),
            words_csv=str(payload.get("words_csv") or ""),
        )
        return JSONResponse({"status": "ok", **batch_options_payload(project_root)}, status_code=200)

    @app.post("/api/batches/delete")
    async def api_batches_delete(payload: dict[str, Any] | None = Body(default=None)):
        payload = payload or {}
        delete_batch_mapping(
            project_root,
            batch_name=str(payload.get("batch_name") or ""),
        )
        return JSONResponse({"status": "ok", **batch_options_payload(project_root)}, status_code=200)

    @app.get("/api/layer-neurons/list-json")
    async def api_layer_neurons_list_json():
        payload = layer_neurons_list_payload(project_root)
        status = 200 if payload.get("status") == "ok" else 500
        return JSONResponse(payload, status_code=status)

    @app.get("/api/attribute-groups/json")
    async def api_attribute_groups_json():
        payload = attribute_groups_payload(project_root)
        status = 200 if payload.get("status") == "ok" else 500
        return JSONResponse(payload, status_code=status)

    @app.get("/api/history/layer-ffn-neuron/latest")
    async def api_history_ffn_latest():
        result = load_latest_ffn_neuron_history_result(project_root)
        status = 200 if result.get("status") == "ok" else 404
        return JSONResponse(result, status_code=status)

    @app.get("/api/history/layer-ffn-neuron/list")
    async def api_history_ffn_list():
        return JSONResponse(list_ffn_neuron_history(project_root), status_code=200)

    @app.get("/api/history/layer-ffn-neuron/item")
    async def api_history_ffn_item(name: str = ""):
        result = load_ffn_neuron_history_result(project_root, name=str(name or "").strip() or None)
        status = 200 if result.get("status") == "ok" else 404
        return JSONResponse(result, status_code=status)

    @app.post("/api/execute")
    async def api_execute(payload: dict[str, Any] | None = Body(default=None)):
        payload = payload or {}
        action_id = str(payload.get("action_id") or "")
        params = payload.get("params") or {}
        if action_id in {"study_layer_neuron_logits_table", "study_layer_ffn_neuron_logits_table", "study_layer_neurons", "study_attribute_group_neurons"}:
            result = start_ui_action_task(
                action_id=action_id,
                params=params,
                project_root=project_root,
                config_path=config_path,
                timeout_seconds=int(payload.get("timeout_seconds") or 0),
            )
        else:
            result = execute_ui_action(
                action_id=action_id,
                params=params,
                project_root=project_root,
                config_path=config_path,
                timeout_seconds=int(payload.get("timeout_seconds") or 0),
            )
        if result.get("status") == "ok":
            status = 200
        elif result.get("status") == "accepted":
            status = 202
        elif result.get("status") == "busy":
            status = 409
        else:
            status = 500
        return JSONResponse(result, status_code=status)

    @app.get("/api/tasks/{task_id}")
    async def api_task(task_id: str):
        result = get_ui_action_task(task_id)
        if result.get("status") in {"ok", "running"}:
            status = 200
        elif result.get("status") == "not_found":
            status = 404
        elif result.get("status") == "error":
            status = 500
        else:
            status = 200
        return JSONResponse(result, status_code=status)

    @app.get("/api/tasks/{task_id}/events")
    async def api_task_events(task_id: str):
        async def event_gen():
            # Push snapshot every second until task reaches terminal state.
            while True:
                payload = get_ui_action_task(task_id)
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                status = str(payload.get("status") or "")
                if status in {"ok", "error", "not_found"}:
                    break
                await asyncio.sleep(1.0)

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    @app.post("/api/chat")
    async def api_chat(payload: dict[str, Any] | None = Body(default=None)):
        payload = payload or {}
        result = execute_chat_completion(
            messages=payload.get("messages") or [],
            config_path=config_path,
            max_new_tokens=int(payload.get("max_new_tokens") or 128),
            temperature=float(payload.get("temperature") or 0.7),
            top_p=float(payload.get("top_p") or 0.9),
            include_assistant_marker=bool(payload.get("include_assistant_marker", True)),
            layer_neuron_change=payload.get("layer_neuron_change"),
            ffn_neuron_change=payload.get("ffn_neuron_change"),
        )
        if result.get("status") == "ok":
            status = 200
        elif result.get("status") == "busy":
            status = 409
        else:
            status = 500
        return JSONResponse(result, status_code=status)

    @app.get("/static/{rel_path:path}")
    async def static_file(rel_path: str):
        path = _safe_join(static_dir, unquote(rel_path))
        if not path.exists() or not path.is_file():
            return JSONResponse({"error": "File not found"}, status_code=404)
        guessed_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return FileResponse(path, media_type=guessed_type)

    @app.get("/outputs/{rel_path:path}")
    async def output_file(rel_path: str):
        path = _safe_join(outputs_dir, unquote(rel_path))
        if not path.exists() or not path.is_file():
            return JSONResponse({"error": "File not found"}, status_code=404)
        guessed_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return FileResponse(path, media_type=guessed_type)

    print(f"UI server running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    uvicorn.run(app, host=host, port=port, log_level="info")


def _run_builtin_server(project_root: Path, config_path: Path, *, host: str, port: int) -> None:
    handler = _make_handler(project_root.resolve(), Path(config_path))
    class _ThreadedServer(ThreadingHTTPServer):
        daemon_threads = True
    server = _ThreadedServer((host, port), handler)
    print(f"UI server running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping UI server...")
    finally:
        server.server_close()


def _make_handler(project_root: Path, config_path: Path) -> type[BaseHTTPRequestHandler]:
    template_dir = project_root / "src" / "ui" / "templates"
    static_dir = project_root / "src" / "ui" / "static"
    outputs_dir = project_root / "data" / "outputs"

    class UIRequestHandler(BaseHTTPRequestHandler):
        server_version = "LLMProbeUI/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/":
                self._send_file(template_dir / "index.html", "text/html; charset=utf-8")
                return
            if path == "/popup":
                self._send_file(template_dir / "popup.html", "text/html; charset=utf-8")
                return
            if path == "/api/actions":
                self._send_json(actions_payload())
                return
            if path == "/api/batches":
                self._send_json(batch_options_payload(project_root))
                return
            if path == "/api/layer-neurons/list-json":
                payload = layer_neurons_list_payload(project_root)
                self._send_json(payload, status=200 if payload.get("status") == "ok" else 500)
                return
            if path == "/api/attribute-groups/json":
                payload = attribute_groups_payload(project_root)
                self._send_json(payload, status=200 if payload.get("status") == "ok" else 500)
                return
            if path == "/api/history/layer-ffn-neuron/latest":
                result = load_latest_ffn_neuron_history_result(project_root)
                self._send_json(result, status=200 if result.get("status") == "ok" else 404)
                return
            if path == "/api/history/layer-ffn-neuron/list":
                self._send_json(list_ffn_neuron_history(project_root), status=200)
                return
            if path == "/api/history/layer-ffn-neuron/item":
                query = parse_qs(parsed.query or "")
                name = str((query.get("name") or [""])[0] or "").strip()
                result = load_ffn_neuron_history_result(project_root, name=name or None)
                self._send_json(result, status=200 if result.get("status") == "ok" else 404)
                return
            if path.startswith("/api/tasks/") and path.endswith("/events"):
                # Built-in server fallback: emulate SSE stream.
                task_id = unquote(path.removeprefix("/api/tasks/").removesuffix("/events")).strip("/")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                while True:
                    payload = get_ui_action_task(task_id)
                    chunk = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    status = str(payload.get("status") or "")
                    if status in {"ok", "error", "not_found"}:
                        break
                    time.sleep(1.0)
                return
            if path.startswith("/api/tasks/"):
                task_id = unquote(path.removeprefix("/api/tasks/"))
                result = get_ui_action_task(task_id)
                if result.get("status") in {"ok", "running"}:
                    self._send_json(result, status=200)
                elif result.get("status") == "not_found":
                    self._send_json(result, status=404)
                elif result.get("status") == "error":
                    self._send_json(result, status=500)
                else:
                    self._send_json(result, status=200)
                return
            if path.startswith("/static/"):
                rel = unquote(path.removeprefix("/static/"))
                self._send_file(_safe_join(static_dir, rel))
                return
            if path.startswith("/outputs/"):
                rel = unquote(path.removeprefix("/outputs/"))
                self._send_file(_safe_join(outputs_dir, rel))
                return
            self._send_json({"error": "Not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path not in {"/api/execute", "/api/chat", "/api/batches/upsert", "/api/batches/delete"}:
                self._send_json({"error": "Not found"}, status=404)
                return
            content_length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(content_length)
            try:
                payload = parse_json_body(body)
                if parsed.path == "/api/chat":
                    result = execute_chat_completion(
                        messages=payload.get("messages") or [],
                        config_path=config_path,
                        max_new_tokens=int(payload.get("max_new_tokens") or 128),
                        temperature=float(payload.get("temperature") or 0.7),
                        top_p=float(payload.get("top_p") or 0.9),
                        include_assistant_marker=bool(payload.get("include_assistant_marker", True)),
                        layer_neuron_change=payload.get("layer_neuron_change"),
                        ffn_neuron_change=payload.get("ffn_neuron_change"),
                    )
                elif parsed.path == "/api/batches/upsert":
                    upsert_batch_mapping(
                        project_root,
                        batch_name=str(payload.get("batch_name") or ""),
                        words_csv=str(payload.get("words_csv") or ""),
                    )
                    result = {"status": "ok", **batch_options_payload(project_root)}
                elif parsed.path == "/api/batches/delete":
                    delete_batch_mapping(
                        project_root,
                        batch_name=str(payload.get("batch_name") or ""),
                    )
                    result = {"status": "ok", **batch_options_payload(project_root)}
                else:
                    action_id = str(payload.get("action_id") or "")
                    params = payload.get("params") or {}
                    if action_id in {"study_layer_neuron_logits_table", "study_layer_ffn_neuron_logits_table", "study_layer_neurons", "study_attribute_group_neurons"}:
                        result = start_ui_action_task(
                            action_id=action_id,
                            params=params,
                            project_root=project_root,
                            config_path=config_path,
                            timeout_seconds=int(payload.get("timeout_seconds") or 0),
                        )
                    else:
                        result = execute_ui_action(
                            action_id=action_id,
                            params=params,
                            project_root=project_root,
                            config_path=config_path,
                            timeout_seconds=int(payload.get("timeout_seconds") or 0),
                        )
                if result.get("status") == "ok":
                    self._send_json(result, status=200)
                elif result.get("status") == "accepted":
                    self._send_json(result, status=202)
                elif result.get("status") == "busy":
                    self._send_json(result, status=409)
                else:
                    self._send_json(result, status=500)
            except Exception as exc:  # noqa: BLE001 - this is a local debugging UI.
                self._send_json({"status": "error", "error": str(exc)}, status=500)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[ui] {self.address_string()} - {format % args}")

        def _send_file(self, path: Path, content_type: str | None = None) -> None:
            if not path.exists() or not path.is_file():
                self._send_json({"error": "File not found"}, status=404)
                return
            guessed_type = content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", guessed_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return UIRequestHandler


def _safe_join(root: Path, rel_path: str) -> Path:
    candidate = (root / rel_path).resolve()
    root_resolved = root.resolve()
    if os.path.commonpath([str(root_resolved), str(candidate)]) != str(root_resolved):
        return root_resolved / "__missing__"
    return candidate
