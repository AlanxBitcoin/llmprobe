from __future__ import annotations

import argparse
import os
from pathlib import Path
import torch

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
    mpl_dir = Path(".mplconfig")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir.resolve())

from src.config import load_config
from src.model_loader import load_local_model
from src.pipeline import ProbePipeline
from src.probes.attribute_probe import (
    build_feature_bank,
    export_attribute_probe_results,
    fit_full_attribute_probes,
    load_attribute_rows,
    predict_word_attributes,
    train_attribute_probes,
)
from src.probes.linear_probe import build_probe_dataset, export_probe_results, load_labeled_words, train_linear_probe
from src.utils import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM hidden-state probe MVP")
    parser.add_argument("--config", default="configs/custom.yaml", help="Path to YAML config")
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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command is None:
        from src.ui import run_ui_server

        run_ui_server(config, config_path=args.config)
        return

    bundle = load_local_model(config)
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
        label_rows = load_labeled_words(args.label_file)
        target_layer = int(config["analysis"]["target_layer"])
        dataset = build_probe_dataset(bundle, label_rows, target_layer)
        results = train_linear_probe(dataset, config=config)
        export_probe_results("data/outputs/probe", results)
    elif args.command == "run-attribute-probe":
        target_layer = int(config["analysis"]["target_layer"])
        rows = load_attribute_rows(args.attribute_file)
        feature_bank = build_feature_bank(bundle, rows, target_layer)
        results = train_attribute_probes(feature_bank, rows, config=config)
        export_attribute_probe_results("data/outputs/attribute_probe", results)
    elif args.command == "predict-attributes":
        target_layer = int(config["analysis"]["target_layer"])
        rows = load_attribute_rows(args.attribute_file)
        feature_bank = build_feature_bank(bundle, rows, target_layer)
        fitted = fit_full_attribute_probes(feature_bank, rows, config=config)
        result = predict_word_attributes(bundle, fitted, args.word, target_layer, config=config)
        write_json(f"data/outputs/predict_{args.word}.json", result)


if __name__ == "__main__":
    main()
