from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Expose UI server entrypoint for main/startup usage.

from .server import run_ui_server

__all__ = ["run_ui_server"]
