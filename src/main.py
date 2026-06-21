from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Single application entry under src/main.py.
# - Default config path is configs/custom.yaml.
# - No-command launch should start runtime API then local UI server.
# - CLI subcommands execute study/probe/cache build workflows.
#
# Responsibility split:
# - main.py owns app bootstrap, shared CLI skeleton, and non-study command orchestration.
# - src/study/cli.py owns study-related CLI argument registration and study command dispatch.
# - src/non_study_cli.py owns non-study CLI argument registration and non-study command dispatch.
# - Keep argparse/run wiring out of main.py to keep this entry file concise.

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import Any
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# Always use non-GUI matplotlib backend to avoid Tk main-loop/thread teardown issues.
os.environ.setdefault("MPLBACKEND", "Agg")

# 1. 关日志、关梯度（必开）
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
torch.set_grad_enabled(False)

# 2. 强制TF32（40系Tensor Core靠这个）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# 3. 锁定GPU，杜绝CPU fallback
device = torch.device("cuda:0")

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = PROJECT_ROOT / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir.resolve())

from src.config import load_config
from src.non_study_cli import (
    add_non_study_subparsers_after_study,
    add_non_study_subparsers_before_study,
    try_execute_non_study_command,
)
from src.runtime_api import start_llama_api
from src.study.cli import add_study_subparsers, try_execute_study_command
from src.utils.extract_hidden import preload_hidden_store_from_disk


def _parse_bool_flag(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean flag: {value!r}. Use true/false.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM hidden-state probe MVP")
    parser.add_argument("--config", default="configs/custom.yaml", help="Path to YAML config")
    parser.add_argument(
        "--start-llama-api",
        type=_parse_bool_flag,
        default=True,
        help="Whether to start Llama runtime API on app boot (true/false)",
    )
    subparsers = parser.add_subparsers(dest="command")

    add_non_study_subparsers_before_study(subparsers)

    add_study_subparsers(subparsers, _parse_bool_flag)

    add_non_study_subparsers_after_study(subparsers, _parse_bool_flag)

    return parser


def start_app(
    config_path: str | Path = "configs/custom.yaml",
    *,
    config: dict[str, Any] | None = None,
    start_llama_runtime: bool | None = None,
) -> None:
    effective_config = config or load_config(config_path)
    runtime_cfg = (effective_config or {}).get("runtime", {})
    ui_cfg = (effective_config or {}).get("ui", {})
    hidden_store_cfg = dict((effective_config or {}).get("hidden_store") or {})
    should_init_hidden_store_disk = bool(hidden_store_cfg.get("init_on_main_boot", True))

    should_start_llama = bool(runtime_cfg.get("start_llama_api_on_boot", True)) if start_llama_runtime is None else bool(start_llama_runtime)

    if bool(ui_cfg.get("enabled", True)) and bool(ui_cfg.get("start_server_on_boot", True)):
        from src.ui import run_ui_server

        # Boot order: start UI first (independent thread), then warm up model in background.
        # This keeps the UI responsive immediately even when model warmup is heavy.
        ui_runtime_config = dict(effective_config)
        ui_runtime_cfg = dict((ui_runtime_config.get("ui") or {}))
        ui_runtime_cfg.setdefault("prefer_builtin_threading", True)
        ui_runtime_config["ui"] = ui_runtime_cfg

        ui_thread = threading.Thread(
            target=run_ui_server,
            kwargs={"config": ui_runtime_config, "config_path": config_path},
            name="ui-server",
            daemon=True,
        )
        ui_thread.start()

        if should_start_llama:
            # Model/tokenizer warmup should stay enabled, but hidden-store memmap preload
            # is intentionally skipped on boot to avoid heavy disk scan/read at startup.
            _start_llama_background(effective_config)

        if should_init_hidden_store_disk:
            _start_hidden_store_disk_init_background(effective_config)

        # Keep process alive while still allowing Ctrl+C to stop the app cleanly.
        try:
            while ui_thread.is_alive():
                ui_thread.join(timeout=0.5)
        except KeyboardInterrupt:
            print("\nStopping app...")
            return
        return

    if should_init_hidden_store_disk:
        try:
            preload_hidden_store_from_disk(effective_config)
        except Exception as exc:  # noqa: BLE001 - startup continues even if preload fails.
            print(f"[startup] hidden_store disk init skipped: {exc}")

    if should_start_llama:
        _ = start_llama_api(effective_config)
    print("No server started because ui.enabled/start_server_on_boot is false.")


def _start_hidden_store_disk_init_background(config: dict[str, Any]) -> None:
    def _runner() -> None:
        try:
            preload_hidden_store_from_disk(config)
            print("[startup] hidden_store disk init finished.")
        except Exception as exc:  # noqa: BLE001
            print(f"[startup] hidden_store disk init skipped: {exc}")

    thread = threading.Thread(target=_runner, name="hidden-store-disk-init", daemon=True)
    thread.start()


def _start_llama_background(config: dict[str, Any]) -> None:
    def _runner() -> None:
        try:
            print("[startup] background model warmup started...")
            _ = start_llama_api(config)
            print("[startup] background model warmup finished.")
        except Exception as exc:  # noqa: BLE001
            print(f"[startup] background model warmup failed: {exc}")

    thread = threading.Thread(target=_runner, name="llama-warmup", daemon=True)
    thread.start()


def _execute_parsed_args(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any] | None:
    if args.command is None:
        start_app(args.config, config=config, start_llama_runtime=bool(args.start_llama_api))
        return None

    study_payload = try_execute_study_command(args, config)
    if study_payload is not None:
        return study_payload

    return try_execute_non_study_command(args, config)


def run_cli_command(config_path: str | Path, command_args: list[str]) -> dict[str, Any] | None:
    parser = build_parser()
    args = parser.parse_args(["--config", str(config_path), *command_args])
    config = load_config(args.config)
    return _execute_parsed_args(args, config)


def main() -> None:
    os.chdir(PROJECT_ROOT)
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    _execute_parsed_args(args, config)


if __name__ == "__main__":
    main()
