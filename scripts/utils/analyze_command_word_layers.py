from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(".mplconfig")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir.resolve())

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.model_loader import load_local_model
from src.utils.utils import ensure_dir, read_lines, write_csv, write_json, write_text


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def extract_32_layer_vectors(bundle, text: str) -> np.ndarray:
    """Return [32, hidden_dim] vectors from transformer layers, excluding embedding state."""
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

    layer_vectors = []
    for layer_hidden in hidden_states[1:33]:
        layer_vectors.append(layer_hidden[0, -1, :].detach().float().cpu().numpy())
    return np.stack(layer_vectors, axis=0).astype(np.float32)


def summarize_word_file(bundle, word_file: Path) -> tuple[np.ndarray, list[str]]:
    inputs = read_lines(word_file)
    if not inputs:
        raise ValueError(f"No non-empty inputs found in {word_file}")

    vectors = [extract_32_layer_vectors(bundle, text) for text in inputs]
    stacked = np.stack(vectors, axis=0)  # [sample_count, 32, 4096]
    return stacked, inputs


def token_rows(bundle, text: str, token_ids: list[int], token_type: str) -> list[dict[str, Any]]:
    tokens = bundle.tokenizer.convert_ids_to_tokens(token_ids)
    rows: list[dict[str, Any]] = []
    for index, (token_id, token) in enumerate(zip(token_ids, tokens), start=1):
        rows.append(
            {
                "token_type": token_type,
                "index": index,
                "token_id": int(token_id),
                "token": token,
                "token_escaped": token.encode("unicode_escape").decode("ascii"),
                "decoded": bundle.tokenizer.decode([token_id], skip_special_tokens=False),
                "decoded_escaped": bundle.tokenizer.decode([token_id], skip_special_tokens=False)
                .encode("unicode_escape")
                .decode("ascii"),
            }
        )
    return rows


def prompt_token_rows(bundle, text: str) -> list[dict[str, Any]]:
    encoded = bundle.tokenizer(text, return_tensors="pt")
    token_ids = encoded["input_ids"][0].tolist()
    return token_rows(bundle, text, token_ids, token_type="prompt")


def generate_token_payload(bundle, text: str, max_new_tokens: int) -> dict[str, Any]:
    tokenizer = bundle.tokenizer
    model = bundle.model
    device = _model_device(model)
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_ids = generated[0].tolist()
    prompt_ids = encoded["input_ids"][0].tolist()
    generated_ids = full_ids[len(prompt_ids) :]
    return {
        "text": text,
        "max_new_tokens": max_new_tokens,
        "prompt_token_count": len(prompt_ids),
        "generated_token_count": len(generated_ids),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "prompt_tokens": token_rows(bundle, text, prompt_ids, token_type="prompt"),
        "generated_tokens": token_rows(bundle, text, generated_ids, token_type="generated"),
    }


def full_stats_rows(stacked: np.ndarray) -> list[dict[str, Any]]:
    sample_count, layer_count, hidden_dim = stacked.shape
    mean_values = stacked.mean(axis=0)
    mean_abs_values = np.abs(stacked).mean(axis=0)
    max_values = stacked.max(axis=0)
    min_values = stacked.min(axis=0)
    std_values = stacked.std(axis=0)
    positive_counts = (stacked > 0).sum(axis=0)
    negative_counts = (stacked < 0).sum(axis=0)
    zero_counts = (stacked == 0).sum(axis=0)

    rows: list[dict[str, Any]] = []
    for layer_idx in range(layer_count):
        for dim in range(hidden_dim):
            mean_value = float(mean_values[layer_idx, dim])
            rows.append(
                {
                    "layer": layer_idx + 1,
                    "dim": dim,
                    "sample_count": sample_count,
                    "mean_value": mean_value,
                    "mean_abs_value": float(mean_abs_values[layer_idx, dim]),
                    "max_value": float(max_values[layer_idx, dim]),
                    "min_value": float(min_values[layer_idx, dim]),
                    "std_value": float(std_values[layer_idx, dim]),
                    "positive_count": int(positive_counts[layer_idx, dim]),
                    "negative_count": int(negative_counts[layer_idx, dim]),
                    "zero_count": int(zero_counts[layer_idx, dim]),
                    "mean_direction": "positive" if mean_value >= 0 else "negative",
                }
            )
    return rows


def diff_top_rows(left_mean: np.ndarray, right_mean: np.ndarray, top_k: int = 10) -> list[dict[str, Any]]:
    diff = left_mean - right_mean
    abs_diff = np.abs(diff)
    layer_count, hidden_dim = diff.shape
    rows: list[dict[str, Any]] = []
    for layer_idx in range(layer_count):
        top_indices = np.argsort(abs_diff[layer_idx])[::-1][:top_k]
        for rank, dim in enumerate(top_indices, start=1):
            rows.append(
                {
                    "layer": layer_idx + 1,
                    "rank": rank,
                    "dim": int(dim),
                    "abs_diff": float(abs_diff[layer_idx, dim]),
                    "signed_diff": float(diff[layer_idx, dim]),
                    "command_mean": float(left_mean[layer_idx, dim]),
                    "no_command_mean": float(right_mean[layer_idx, dim]),
                    "larger_group": "command" if abs(left_mean[layer_idx, dim]) >= abs(right_mean[layer_idx, dim]) else "no_command",
                }
            )
    return rows


def render_diff_top_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    best_by_layer = [row for row in rows if int(row["rank"]) == 1]
    layers = [int(row["layer"]) for row in best_by_layer]
    values = [float(row["abs_diff"]) for row in best_by_layer]
    labels = [f"d{row['dim']}" for row in best_by_layer]

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(18, 7), dpi=160)
    bars = ax.bar(layers, values, color="#c55a11", edgecolor="white", linewidth=0.5)
    ax.set_title("Input A vs Input B: Largest |Difference| per Transformer Layer", fontsize=13)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Top-1 absolute difference across 4096 dims")
    ax.set_xticks(layers)
    ax.grid(axis="y", alpha=0.2)
    for bar, label, value in zip(bars, labels, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{label}\n{value:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_diff_heatmap(rows: list[dict[str, Any]], output_path: Path) -> None:
    matrix = np.zeros((32, 10), dtype=np.float32)
    labels = [["" for _ in range(10)] for _ in range(32)]
    for row in rows:
        layer_idx = int(row["layer"]) - 1
        rank_idx = int(row["rank"]) - 1
        if 0 <= layer_idx < 32 and 0 <= rank_idx < 10:
            matrix[layer_idx, rank_idx] = float(row["abs_diff"])
            labels[layer_idx][rank_idx] = f"d{row['dim']}\n{float(row['abs_diff']):.2f}"

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(16, 14), dpi=160)
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_title("Layer × Rank Heatmap: Top10 |Input A - Input B| Dimensions", fontsize=13)
    ax.set_xlabel("Rank within each layer")
    ax.set_ylabel("Transformer layer")
    ax.set_xticks(np.arange(10), labels=[f"Top {i}" for i in range(1, 11)])
    ax.set_yticks(np.arange(32), labels=[str(i) for i in range(1, 33)])
    for layer_idx in range(32):
        for rank_idx in range(10):
            ax.text(rank_idx, layer_idx, labels[layer_idx][rank_idx], ha="center", va="center", fontsize=5.2, color="#111827")
    fig.colorbar(im, ax=ax, label="Absolute difference")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_dim_frequency_chart(rows: list[dict[str, Any]], output_path: Path, top_n: int = 20) -> None:
    counts: dict[int, int] = {}
    max_diff: dict[int, float] = {}
    for row in rows:
        dim = int(row["dim"])
        diff = float(row["abs_diff"])
        counts[dim] = counts.get(dim, 0) + 1
        max_diff[dim] = max(max_diff.get(dim, 0.0), diff)

    selected = sorted(counts, key=lambda dim: (counts[dim], max_diff[dim]), reverse=True)[:top_n]
    labels = [f"d{dim}" for dim in selected]
    values = [counts[dim] for dim in selected]
    max_values = [max_diff[dim] for dim in selected]

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 8), dpi=160)
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color="#3d405b", edgecolor="white", linewidth=0.5)
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_title("Most Frequent Dimensions in Layer Top10 Difference Lists", fontsize=13)
    ax.set_xlabel("Appearance count across 32×Top10 slots")
    ax.grid(axis="x", alpha=0.2)
    for bar, count, max_val in zip(bars, values, max_values):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2, f"n={count}, max={max_val:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_layer_top10_lines(rows: list[dict[str, Any]], output_path: Path) -> None:
    by_rank: dict[int, list[tuple[int, float]]] = {rank: [] for rank in range(1, 11)}
    for row in rows:
        by_rank[int(row["rank"])].append((int(row["layer"]), float(row["abs_diff"])))

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(16, 8), dpi=160)
    for rank in range(1, 11):
        points = sorted(by_rank[rank])
        layers = [layer for layer, _ in points]
        values = [value for _, value in points]
        linewidth = 2.4 if rank == 1 else 1.0
        alpha = 0.95 if rank <= 3 else 0.45
        ax.plot(layers, values, marker="o", markersize=3, linewidth=linewidth, alpha=alpha, label=f"Top {rank}")
    ax.set_title("Top10 Absolute Differences Across Layers", fontsize=13)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Absolute difference")
    ax.set_xticks(range(1, 33))
    ax.grid(alpha=0.2)
    ax.legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_html(out_dir: Path, left_name: str, right_name: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Command Word Layer Difference</title>
  <style>
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 24px; background: #faf7f2; color: #1f2937; }}
    .card {{ background: white; border: 1px solid #d6d3d1; border-radius: 14px; padding: 18px; margin-bottom: 18px; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 10px; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>两输入文本：32层全4096维差值分析</h1>
  <section class="card">
    <p>左侧词库：{left_name}</p>
    <p>右侧词库：{right_name}</p>
    <p>记录范围：Transformer layer 1-32，每层 4096 维，取最后 token hidden state。</p>
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
    <p><a href="command_full_32_layers_4096.csv">command 全32层4096维统计 CSV</a></p>
    <p><a href="no_command_full_32_layers_4096.csv">no-command 全32层4096维统计 CSV</a></p>
    <p><a href="layer_dim_abs_diff_top10.csv">对应层对应维度相减取绝对值 Top10 CSV</a></p>
    <p><a href="layer_dim_abs_diff_top10.json">Top10 JSON</a></p>
    <p><a href="command_generated_tokens.csv">command 生成 token CSV</a></p>
    <p><a href="no_command_generated_tokens.csv">no-command 生成 token CSV</a></p>
    <p><a href="generated_tokens.json">两组 prompt/generated token JSON</a></p>
  </section>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two command-word files across all 32 layers and 4096 dims.")
    parser.add_argument("--config", default="configs/quantized-4bit.yaml")
    parser.add_argument("--left", default="data/outputs/command word/command word.txt")
    parser.add_argument("--right", default="data/outputs/command word/no command word.txt")
    parser.add_argument("--out-dir", default="data/outputs/command word/layer32_4096_diff")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    left_path = Path(args.left)
    right_path = Path(args.right)
    out_dir = ensure_dir(args.out_dir)

    config = load_config(args.config)
    bundle = load_local_model(config)

    left_stack, left_inputs = summarize_word_file(bundle, left_path)
    right_stack, right_inputs = summarize_word_file(bundle, right_path)
    left_generated_payload = generate_token_payload(bundle, left_inputs[0], max_new_tokens=args.max_new_tokens)
    right_generated_payload = generate_token_payload(bundle, right_inputs[0], max_new_tokens=args.max_new_tokens)

    left_mean = left_stack.mean(axis=0)
    right_mean = right_stack.mean(axis=0)
    top_rows = diff_top_rows(left_mean, right_mean, top_k=args.top_k)

    write_csv(out_dir / "command_full_32_layers_4096.csv", full_stats_rows(left_stack))
    write_csv(out_dir / "no_command_full_32_layers_4096.csv", full_stats_rows(right_stack))
    write_csv(out_dir / "layer_dim_abs_diff_top10.csv", top_rows)
    write_csv(out_dir / "command_prompt_tokens.csv", left_generated_payload["prompt_tokens"])
    write_csv(out_dir / "command_generated_tokens.csv", left_generated_payload["generated_tokens"])
    write_csv(out_dir / "no_command_prompt_tokens.csv", right_generated_payload["prompt_tokens"])
    write_csv(out_dir / "no_command_generated_tokens.csv", right_generated_payload["generated_tokens"])
    write_json(
        out_dir / "generated_tokens.json",
        {
            "left_file": str(left_path),
            "right_file": str(right_path),
            "max_new_tokens": args.max_new_tokens,
            "command": left_generated_payload,
            "no_command": right_generated_payload,
        },
    )
    write_json(
        out_dir / "layer_dim_abs_diff_top10.json",
        {
            "left_file": str(left_path),
            "right_file": str(right_path),
            "left_inputs": left_inputs,
            "right_inputs": right_inputs,
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
                "left_sample_count": int(left_stack.shape[0]),
                "right_sample_count": int(right_stack.shape[0]),
                "layer_count": int(left_stack.shape[1]),
                "hidden_dim": int(left_stack.shape[2]),
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
    write_text(out_dir / "index.html", build_html(out_dir, left_path.name, right_path.name))
    print(out_dir / "index.html")


if __name__ == "__main__":
    main()
