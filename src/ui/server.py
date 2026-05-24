from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Host local web UI server and static/template/output routes.
# - UI layer handles interaction only; no probe training logic here.
# - All file serving remains under project-root-safe paths.

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .routes import actions_payload, execute_ui_action, parse_json_body


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
    try:
        _run_fastapi_server(root.resolve(), Path(config_path), host=str(host), port=int(port))
        return
    except Exception as exc:  # noqa: BLE001 - fall back to built-in server for local robustness.
        print(f"[ui] FastAPI server unavailable, fallback to built-in server: {exc}")
    _run_builtin_server(root.resolve(), Path(config_path), host=str(host), port=int(port))


def _run_fastapi_server(project_root: Path, config_path: Path, *, host: str, port: int) -> None:
    from fastapi import Body, FastAPI
    from fastapi.responses import FileResponse, JSONResponse
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

    @app.get("/api/actions")
    async def api_actions():
        return JSONResponse(actions_payload())

    @app.post("/api/execute")
    async def api_execute(payload: dict[str, Any] | None = Body(default=None)):
        payload = payload or {}
        action_id = str(payload.get("action_id") or "")
        params = payload.get("params") or {}
        result = execute_ui_action(
            action_id=action_id,
            params=params,
            project_root=project_root,
            config_path=config_path,
            timeout_seconds=int(payload.get("timeout_seconds") or 0),
        )
        status = 200 if result.get("status") == "ok" else 500
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
    server = ThreadingHTTPServer((host, port), handler)
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
            if path == "/api/actions":
                self._send_json(actions_payload())
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
            if parsed.path != "/api/execute":
                self._send_json({"error": "Not found"}, status=404)
                return
            content_length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(content_length)
            try:
                payload = parse_json_body(body)
                action_id = str(payload.get("action_id") or "")
                params = payload.get("params") or {}
                result = execute_ui_action(
                    action_id=action_id,
                    params=params,
                    project_root=project_root,
                    config_path=config_path,
                    timeout_seconds=int(payload.get("timeout_seconds") or 0),
                )
                self._send_json(result, status=200 if result.get("status") == "ok" else 500)
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
