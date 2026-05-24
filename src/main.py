from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Single application entry under src/main.py.
# - Default config path is configs/custom.yaml.
# - No-command launch should start runtime API then local UI server.
# - CLI subcommands execute study/probe/cache build workflows.

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

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
from src.model_loader import get_model_bundle
from src.pipeline import ProbePipeline
from src.probes.attribute_probe import (
    build_feature_bank,
    fit_full_attribute_probes,
    load_attribute_rows,
    predict_word_attributes,
)
from src.runtime_api import start_llama_api
from src.study import run_attribute_probe_study, run_linear_probe_study, run_single_word_hidden_state_study
from src.utils.extract_hidden import preload_hidden_store, preload_hidden_store_from_disk
from src.utils.token_hidden_store import build_store_for_protocol
from src.utils.utils import write_json


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

    subparsers.add_parser("run-single-batch", help="Analyze the configured word file one word at a time")

    multi = subparsers.add_parser("run-multi-batch", help="Analyze the configured word file in fixed-size groups")
    multi.add_argument("--batch-size", type=int, default=2, help="Words per group, must be >= 2")

    global_analysis = subparsers.add_parser("run-global-analysis", help="Run single-batch, multi-batch, and dimension report for a specific word list")
    global_analysis.add_argument("word_file", help="Path to the target word list")
    global_analysis.add_argument("run_name", help="Output run prefix")
    global_analysis.add_argument("--batch-size", type=int, default=2, help="Words per group for multi-batch")

    global_analysis_all = subparsers.add_parser(
        "run-global-analysis-all",
        help="Run global analysis for every discovered word list or an explicit set of word lists",
    )
    global_analysis_all.add_argument(
        "word_files",
        nargs="*",
        help="Optional explicit word list paths. If omitted, all matching .txt files under --word-dir are used.",
    )
    global_analysis_all.add_argument("--word-dir", default="data", help="Directory used for word list discovery")
    global_analysis_all.add_argument("--glob", default="*.txt", help="Glob pattern used for discovery inside --word-dir")
    global_analysis_all.add_argument("--run-name", default="global_analysis_all", help="Output folder name for the aggregate run")
    global_analysis_all.add_argument("--batch-size", type=int, default=2, help="Words per group for multi-batch")

    color_experiment = subparsers.add_parser(
        "run-color-words-experiment",
        help="Analyze a color-word list one word at a time and as one combined input",
    )
    color_experiment.add_argument("--word-file", default="data/color_words.txt", help="Path to the color word list")
    color_experiment.add_argument("--run-name", default="color_words", help="Output folder name")

    single = subparsers.add_parser("run-single-word", help="Analyze one word")
    single.add_argument("word", help="Bare English word")

    hidden_map = subparsers.add_parser(
        "run-single-word-hidden-state",
        help="Compute and return full hidden-state matrix (embedding + layers) for one word",
    )
    hidden_map.add_argument("word", help="Bare English word")

    combo = subparsers.add_parser("run-word-sum", help="Analyze the layer-8 summed representation of two or more words")
    combo.add_argument("words", nargs="+", help="Two or more bare English words")

    diff = subparsers.add_parser("run-word-diff", help="Analyze the layer-8 difference representation of two words")
    diff.add_argument("minuend", help="Left word, e.g. asphalt")
    diff.add_argument("subtrahend", help="Right word, e.g. cloud")

    many = subparsers.add_parser("run-multi-word", help="Analyze multiple words")
    many.add_argument("words", nargs="+", help="Two or more bare English words")

    dim_report = subparsers.add_parser("run-dim-report", help="Build a stability report for selected hidden dimensions")
    dim_report.add_argument("dims", nargs="*", type=int, help="Dimension ids, e.g. 4055 1800 1856 912")
    dim_report.add_argument("--word-file", default=None, help="Optional custom word list for the dimension report")
    dim_report.add_argument("--run-name", default="dimension_report", help="Output folder name for the dimension report")

    family_report = subparsers.add_parser("run-word-family-report", help="Build a dimension-word-attribute report for all salient dims from related word analyses")
    family_report.add_argument("words", nargs="+", help="Words to compare, e.g. asphalt cloud")

    contrast_report = subparsers.add_parser("run-word-contrast-report", help="Build a shared-vs-difference dimension report for two words")
    contrast_report.add_argument("left_word", help="Left word")
    contrast_report.add_argument("right_word", help="Right word")

    probe = subparsers.add_parser("run-probe", help="Train a simple linear probe on the labeled 100-word dataset")
    probe.add_argument("--label-file", default="data/word_labels.csv", help="CSV file with word labels")

    attr_probe = subparsers.add_parser("run-attribute-probe", help="Train attribute-family probes on structured word attributes")
    attr_probe.add_argument("--attribute-file", default="data/word_attributes.csv", help="CSV file with structured word attributes")

    attr_predict = subparsers.add_parser("predict-attributes", help="Predict attribute families for a single word using full-dataset probes")
    attr_predict.add_argument("word", help="Bare English word")
    attr_predict.add_argument("--attribute-file", default="data/word_attributes.csv", help="CSV file with structured word attributes")

    build_store = subparsers.add_parser("build-token-hidden-store", help="Build token hidden-state cache for one protocol")
    build_store.add_argument("--bos", type=_parse_bool_flag, default=True, help="Whether to prepend BOS (true/false)")
    build_store.add_argument("--assistant", type=_parse_bool_flag, default=False, help="Whether to include assistant prompt prefix (true/false)")
    build_store.add_argument("--limit", type=int, default=0, help="Optional max token count to process in this run (0 = all)")
    build_store.add_argument("--start-token-id", type=int, default=0, help="Token id to start from (default: 0)")

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

    if bool(hidden_store_cfg.get("init_on_main_boot", True)):
        try:
            preload_hidden_store_from_disk(effective_config)
        except Exception as exc:  # noqa: BLE001 - startup continues even if preload fails.
            print(f"[startup] hidden_store disk init skipped: {exc}")

    should_start_llama = bool(runtime_cfg.get("start_llama_api_on_boot", True)) if start_llama_runtime is None else bool(start_llama_runtime)

    if bool(ui_cfg.get("enabled", True)) and bool(ui_cfg.get("start_server_on_boot", True)):
        from src.ui import run_ui_server

        if should_start_llama:
            _start_llama_after_ui_boot(effective_config, preload_hidden=bool(hidden_store_cfg.get("preload_on_boot", True)))
        run_ui_server(effective_config, config_path=config_path)
        return

    if should_start_llama:
        api = start_llama_api(effective_config)
        if bool(hidden_store_cfg.get("preload_on_boot", True)):
            try:
                preload_hidden_store(api.get_bundle(), effective_config)
            except Exception as exc:  # noqa: BLE001 - preload failure should not block startup.
                print(f"[startup] hidden_store preload skipped: {exc}")
    print("No server started because ui.enabled/start_server_on_boot is false.")


def _start_llama_after_ui_boot(config: dict[str, Any], *, preload_hidden: bool) -> None:
    def _runner() -> None:
        time.sleep(1.0)
        try:
            print("[startup] background model warmup started...")
            api = start_llama_api(config)
            if preload_hidden:
                try:
                    preload_hidden_store(api.get_bundle(), config)
                except Exception as exc:  # noqa: BLE001 - preload failure should not break warmup thread.
                    print(f"[startup] hidden_store preload skipped: {exc}")
            print("[startup] background model warmup finished.")
        except Exception as exc:  # noqa: BLE001
            print(f"[startup] background model warmup failed: {exc}")

    thread = threading.Thread(target=_runner, name="llama-warmup", daemon=True)
    thread.start()


def _execute_parsed_args(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any] | None:
    if args.command is None:
        start_app(args.config, config=config, start_llama_runtime=bool(args.start_llama_api))
        return None

    if args.command == "run-single-word-hidden-state":
        heatmap = run_single_word_hidden_state_study(
            word=args.word,
            config=config,
            config_path=args.config,
        )
        return {"hidden_state_heatmap": heatmap}

    if args.command == "build-token-hidden-store":
        bundle = get_model_bundle(config)
        result = build_store_for_protocol(
            bundle,
            config,
            bos=bool(args.bos),
            assistant=bool(args.assistant),
            limit=int(args.limit),
            start_token_id=int(args.start_token_id),
        )
        print("[hidden_store] build finished")
        for key in ("protocol", "token_count", "processed", "written", "skipped_done", "start_token_id", "end_token_id", "data_file", "progress_file"):
            if key in result:
                print(f"[hidden_store] {key}={result[key]}")
        return {"hidden_store": result}

    bundle = get_model_bundle(config)
    pipeline = ProbePipeline(config, bundle)

    if args.command == "run-single-batch":
        pipeline.run_single_batch()
    elif args.command == "run-multi-batch":
        if args.batch_size < 2:
            raise ValueError("--batch-size must be >= 2")
        pipeline.run_multi_batch(args.batch_size)
    elif args.command == "run-global-analysis":
        if args.batch_size < 2:
            raise ValueError("--batch-size must be >= 2")
        pipeline.run_global_analysis(args.word_file, args.run_name, batch_size=args.batch_size)
    elif args.command == "run-global-analysis-all":
        if args.batch_size < 2:
            raise ValueError("--batch-size must be >= 2")
        pipeline.run_global_analysis_all(
            word_files=args.word_files,
            run_name=args.run_name,
            batch_size=args.batch_size,
            word_dir=args.word_dir,
            pattern=args.glob,
        )
    elif args.command == "run-color-words-experiment":
        pipeline.run_color_words_experiment(word_file=args.word_file, run_name=args.run_name)
    elif args.command == "run-single-word":
        result = pipeline.run_single_word(args.word)
        pipeline.save_single_word_outputs(result, "single_word")
    elif args.command == "run-word-sum":
        if len(args.words) < 2:
            raise ValueError("Please provide at least two words.")
        result = pipeline.run_combined_word_sum(args.words)
        pipeline.save_single_word_outputs(result, "word_sum")
    elif args.command == "run-word-diff":
        result = pipeline.run_combined_word_diff(args.minuend, args.subtrahend)
        pipeline.save_single_word_outputs(result, "word_diff")
    elif args.command == "run-multi-word":
        if len(args.words) < 2:
            raise ValueError("Please provide at least two words.")
        result = pipeline.run_multi_word(args.words)
        pipeline.save_multi_word_outputs(result, "multi_word")
    elif args.command == "run-dim-report":
        report = pipeline.run_dimension_report(args.dims, word_file=args.word_file)
        pipeline.save_dimension_report(report, run_name=args.run_name)
    elif args.command == "run-word-family-report":
        if len(args.words) < 2:
            raise ValueError("Please provide at least two words.")
        report = pipeline.run_word_family_dim_report(args.words)
        pipeline.save_dimension_report(report, run_name="word_family_report")
    elif args.command == "run-word-contrast-report":
        report = pipeline.run_word_contrast_report(args.left_word, args.right_word)
        pipeline.save_word_contrast_report(report)
    elif args.command == "run-probe":
        run_linear_probe_study(
            config=config,
            config_path=args.config,
            label_file=args.label_file,
            output_dir="data/outputs/probe",
        )
    elif args.command == "run-attribute-probe":
        run_attribute_probe_study(
            config=config,
            config_path=args.config,
            attribute_file=args.attribute_file,
            output_dir="data/outputs/attribute_probe",
        )
    elif args.command == "predict-attributes":
        target_layer = int(config["analysis"]["target_layer"])
        rows = load_attribute_rows(args.attribute_file)
        feature_bank = build_feature_bank(bundle, rows, target_layer, config=config)
        fitted = fit_full_attribute_probes(feature_bank, rows, config=config)
        result = predict_word_attributes(bundle, fitted, args.word, target_layer, config=config)
        write_json(f"data/outputs/predict_{args.word}.json", result)
    return None


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
