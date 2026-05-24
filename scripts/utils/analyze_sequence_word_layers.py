from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(".mplconfig")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir.resolve())

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.model_loader import load_local_model
from src.utils import ensure_dir, read_lines, write_csv, write_json, write_text
from tools.analyze_command_word_layers import (
    diff_top_rows,
    full_stats_rows,
    generate_token_payload,
    render_diff_heatmap,
    render_diff_top_chart,
    render_dim_frequency_chart,
    render_layer_top10_lines,
)


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def extract_sequence_mean_vectors(bundle, text: str, include_bos: bool = False) -> tuple[np.ndarray, list[int]]:
    """Return [32, hidden_dim] vectors by averaging token states in the sequence."""
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = _model_device(model)
    encoded = tokenizer(text, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    if len(hidden_states) < 33:
        raise ValueError(f"Expected embedding + 32 hidden layers, got {len(hidden_states)} hidden states.")

    token_ids = encoded["input_ids"][0].detach().cpu().tolist()
    start_index = 0 if include_bos else 1
    if start_index >= len(token_ids):
        start_index = 0

    layer_vectors = []
    for layer_hidden in hidden_states[1:33]:
        token_matrix = layer_hidden[0, start_index:, :].detach().float().cpu().numpy()
        layer_vectors.append(token_matrix.mean(axis=0))
    return np.stack(layer_vectors, axis=0).astype(np.float32), token_ids


def summarize_sequence_file(bundle, word_file: Path, include_bos: bool = False) -> tuple[np.ndarray, list[str], list[list[int]]]:
    inputs = read_lines(word_file)
    if not inputs:
        raise ValueError(f"No non-empty inputs found in {word_file}")

    vectors: list[np.ndarray] = []
    token_id_batches: list[list[int]] = []
    for text in inputs:
        layer_vectors, token_ids = extract_sequence_mean_vectors(bundle, text, include_bos=include_bos)
        vectors.append(layer_vectors)
        token_id_batches.append(token_ids)
    return np.stack(vectors, axis=0), inputs, token_id_batches


def build_html(left_name: str, right_name: str, include_bos: bool) -> str:
    token_note = "including BOS token" if include_bos else "excluding BOS token"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Sequence Word Layer Difference</title>
  <style>
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 24px; background: #f7f5ef; color: #1f2937; }}
    .card {{ background: white; border: 1px solid #d6d3d1; border-radius: 14px; padding: 18px; margin-bottom: 18px; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 10px; background: white; }}
    code {{ background: #f3f4f6; padding: 2px 5px; border-radius: 5px; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>Sequence Word: 32层全4096维顺序差异分析</h1>
  <section class="card">
    <p>输入A：<code>{left_name}</code></p>
    <p>输入B：<code>{right_name}</code></p>
    <p>分析范围：Transformer layer 1-32，每层4096维。</p>
    <p>序列处理方式：对每个输入的全部内容 token hidden states 求均值作为该层4096维表示，{token_note}。这不同于只取最后一个 token 的处理方式，更适合观察词序变化造成的整体表征差异。</p>
  </section>
  <section class="card">
    <h2>每层最大差值概览</h2>
    <img src="layer_top1_abs_diff.png" alt="layer top1 abs diff" />
  </section>
  <section class="card">
    <h2>每层 Top10 差值热力图</h2>
    <img src="layer_top10_abs_diff_heatmap.png" alt="layer top10 abs diff heatmap" />
  </section>
  <section class="card">
    <h2>Top10 差值随层变化曲线</h2>
    <img src="layer_top10_abs_diff_lines.png" alt="layer top10 abs diff lines" />
  </section>
  <section class="card">
    <h2>高差异维度出现频次</h2>
    <img src="top_diff_dim_frequency.png" alt="top diff dim frequency" />
  </section>
  <section class="card">
    <h2>CSV / JSON 输出</h2>
    <p><a href="sequence1_full_32_layers_4096.csv">sequence1 全32层4096维统计 CSV</a></p>
    <p><a href="sequence2_full_32_layers_4096.csv">sequence2 全32层4096维统计 CSV</a></p>
    <p><a href="layer_dim_abs_diff_top10.csv">对应层/维度相减取绝对值 Top10 CSV</a></p>
    <p><a href="layer_dim_abs_diff_top10.json">Top10 JSON</a></p>
    <p><a href="sequence1_prompt_tokens.csv">sequence1 prompt token CSV</a></p>
    <p><a href="sequence2_prompt_tokens.csv">sequence2 prompt token CSV</a></p>
    <p><a href="sequence1_generated_tokens.csv">sequence1 generated token CSV</a></p>
    <p><a href="sequence2_generated_tokens.csv">sequence2 generated token CSV</a></p>
    <p><a href="generated_tokens.json">两组 prompt/generated token JSON</a></p>
  </section>
</body>
</html>
"""


def relabel_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        if row.get("larger_group") == "command":
            row["larger_group"] = "sequence1"
        elif row.get("larger_group") == "no_command":
            row["larger_group"] = "sequence2"
        if "command_mean" in row:
            row["sequence1_mean"] = row.pop("command_mean")
        if "no_command_mean" in row:
            row["sequence2_mean"] = row.pop("no_command_mean")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two sequence-word files across 32 layers and 4096 dims.")
    parser.add_argument("--config", default="configs/quantized-4bit.yaml")
    parser.add_argument("--left", default="data/outputs/sequence word/sequence1.txt")
    parser.add_argument("--right", default="data/outputs/sequence word/sequence2.txt")
    parser.add_argument("--out-dir", default="data/outputs/sequence word/layer32_4096_sequence_mean_diff")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--include-bos", action="store_true")
    args = parser.parse_args()

    left_path = Path(args.left)
    right_path = Path(args.right)
    out_dir = ensure_dir(args.out_dir)

    config = load_config(args.config)
    bundle = load_local_model(config)

    left_stack, left_inputs, left_token_ids = summarize_sequence_file(bundle, left_path, include_bos=args.include_bos)
    right_stack, right_inputs, right_token_ids = summarize_sequence_file(bundle, right_path, include_bos=args.include_bos)
    left_payload = generate_token_payload(bundle, left_inputs[0], max_new_tokens=args.max_new_tokens)
    right_payload = generate_token_payload(bundle, right_inputs[0], max_new_tokens=args.max_new_tokens)

    left_mean = left_stack.mean(axis=0)
    right_mean = right_stack.mean(axis=0)
    top_rows = relabel_rows(diff_top_rows(left_mean, right_mean, top_k=args.top_k))

    write_csv(out_dir / "sequence1_full_32_layers_4096.csv", full_stats_rows(left_stack))
    write_csv(out_dir / "sequence2_full_32_layers_4096.csv", full_stats_rows(right_stack))
    write_csv(out_dir / "layer_dim_abs_diff_top10.csv", top_rows)
    write_csv(out_dir / "sequence1_prompt_tokens.csv", left_payload["prompt_tokens"])
    write_csv(out_dir / "sequence1_generated_tokens.csv", left_payload["generated_tokens"])
    write_csv(out_dir / "sequence2_prompt_tokens.csv", right_payload["prompt_tokens"])
    write_csv(out_dir / "sequence2_generated_tokens.csv", right_payload["generated_tokens"])
    write_json(
        out_dir / "generated_tokens.json",
        {
            "left_file": str(left_path),
            "right_file": str(right_path),
            "max_new_tokens": args.max_new_tokens,
            "sequence1": left_payload,
            "sequence2": right_payload,
        },
    )
    write_json(
        out_dir / "layer_dim_abs_diff_top10.json",
        {
            "left_file": str(left_path),
            "right_file": str(right_path),
            "left_inputs": left_inputs,
            "right_inputs": right_inputs,
            "aggregation": "mean over sequence tokens",
            "include_bos": bool(args.include_bos),
            "layer_count": int(left_mean.shape[0]),
            "hidden_dim": int(left_mean.shape[1]),
            "top_k": args.top_k,
            "rows": top_rows,
        },
    )
    write_text(
        out_dir / "metadata.json",
        json.dumps(
            {
                "left_file": str(left_path),
                "right_file": str(right_path),
                "left_inputs": left_inputs,
                "right_inputs": right_inputs,
                "left_token_ids": left_token_ids,
                "right_token_ids": right_token_ids,
                "left_sample_count": int(left_stack.shape[0]),
                "right_sample_count": int(right_stack.shape[0]),
                "layer_count": int(left_stack.shape[1]),
                "hidden_dim": int(left_stack.shape[2]),
                "aggregation": "mean over all sequence token hidden states",
                "include_bos": bool(args.include_bos),
                "note": "Hidden state 0 (embedding) is skipped; layers are transformer layers 1-32.",
            },
            ensure_ascii=False,
            indent=2,
        ),
    )

    render_diff_top_chart(top_rows, out_dir / "layer_top1_abs_diff.png")
    render_diff_heatmap(top_rows, out_dir / "layer_top10_abs_diff_heatmap.png")
    render_layer_top10_lines(top_rows, out_dir / "layer_top10_abs_diff_lines.png")
    render_dim_frequency_chart(top_rows, out_dir / "top_diff_dim_frequency.png")
    write_text(out_dir / "index.html", build_html(left_path.name, right_path.name, include_bos=args.include_bos))
    print(out_dir / "index.html")


if __name__ == "__main__":
    main()
