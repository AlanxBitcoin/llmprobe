"""Visualization helpers for the color-words experiment.

Charts produced here use actual signed activation values (not relative / normalized values).
Positive and negative dimensions are drawn as separate signed bars so the sign is always
visible from the bar direction on the x-axis.
"""

from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Generate color-experiment charts used by pipeline outputs.
# - Keep chart generation isolated from probe/model logic.

import os
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(".mplconfig")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir.resolve())

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


_POS_COLOR = "#e07a5f"
_NEG_COLOR = "#3d405b"
_COUNT_COLOR = "#81b29a"
_MEAN_COLOR = "#f2cc8f"
_MAX_COLOR = "#e07a5f"


def _configure_font_fallback() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def render_per_word_dim_stats(
    average_rows: list[dict[str, Any]],
    output_path: str | Path,
    top_k: int = 20,
    dpi: int = 160,
) -> None:
    """Bar charts summarising per-dim statistics across all single-word inputs.

    Each row in *average_rows* has keys:
        group ('max' or 'min'), dim (int),
        appearance_count, mean_abs_value, max_abs_value, min_abs_value.

    Three subplots are drawn:
    1. Occurrence count per dim (how many words triggered this dim in their top-k)
    2. Mean absolute activation value per dim
    3. Max absolute activation value per dim

    Positive-group dims are drawn in a warm colour; negative-group dims in a cool colour.
    All values are raw activation magnitudes — no relative / normalised scaling.
    """
    _configure_font_fallback()

    max_rows = sorted(
        [r for r in average_rows if r.get("group") == "max"],
        key=lambda r: int(r.get("appearance_count", 0)),
        reverse=True,
    )[:top_k]
    min_rows = sorted(
        [r for r in average_rows if r.get("group") == "min"],
        key=lambda r: int(r.get("appearance_count", 0)),
        reverse=True,
    )[:top_k]

    fig, axes = plt.subplots(2, 3, figsize=(24, 12), dpi=dpi)
    fig.suptitle("Per-Word Input Mode — Dimension Statistics (Layer 8, Signed Values)", fontsize=14)

    _draw_dim_stats_group(axes[0], max_rows, group_label="Positive", color=_POS_COLOR)
    _draw_dim_stats_group(axes[1], min_rows, group_label="Negative", color=_NEG_COLOR)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _draw_dim_stats_group(
    axes_row: np.ndarray,
    rows: list[dict[str, Any]],
    group_label: str,
    color: str,
) -> None:
    """Fill one row of axes (3 subplots) with count / mean-abs / max-abs charts for a dim group."""
    ax_count, ax_mean, ax_max = axes_row[0], axes_row[1], axes_row[2]

    if not rows:
        for ax in axes_row:
            ax.axis("off")
            ax.set_title(f"{group_label} — No data")
        return

    labels = [str(r["dim"]) for r in rows]
    counts = [int(r.get("appearance_count", 0)) for r in rows]
    means = [float(r.get("mean_abs_value", 0.0)) for r in rows]
    maxes = [float(r.get("max_abs_value", 0.0)) for r in rows]
    y = np.arange(len(labels))

    def _barh(ax: plt.Axes, values: list[float], title: str, xlabel: str, bar_color: str) -> None:
        ax.barh(y, values, color=bar_color, edgecolor="white", linewidth=0.4)
        ax.set_yticks(y, labels=labels, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(f"{group_label} Dims — {title}", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        x_max = max(values) if values else 1.0
        for idx, val in enumerate(values):
            ax.text(
                val + x_max * 0.01,
                idx,
                f"{val:.3f}" if isinstance(val, float) else str(val),
                va="center",
                fontsize=7,
            )

    _barh(ax_count, [float(c) for c in counts], "Occurrence Count", "Number of words that activated this dim", _COUNT_COLOR)
    _barh(ax_mean, means, "Mean |Activation|", "Mean absolute activation value (raw, not normalized)", _MEAN_COLOR)
    _barh(ax_max, maxes, "Max |Activation|", "Maximum absolute activation value (raw, not normalized)", bar_color=color)


def render_all_input_dim_stats(
    top_dims: list[dict[str, Any]],
    output_path: str | Path,
    word: str = "all color words",
    layer_index: int = 8,
    dpi: int = 160,
) -> None:
    """Signed-value bar chart for the all-words-in-one-input mode.

    *top_dims* is the list returned by ``summarize_top_dims`` — each entry has:
        dim (int), value (float, signed), abs_value (float).

    Positive dims are shown with bars to the RIGHT (positive x); negative dims
    with bars to the LEFT (negative x).  No normalization or relative scaling is applied.
    """
    _configure_font_fallback()

    positive_dims = sorted([d for d in top_dims if d["value"] >= 0], key=lambda d: d["value"], reverse=True)
    negative_dims = sorted([d for d in top_dims if d["value"] < 0], key=lambda d: d["value"])

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=dpi)
    fig.suptitle(
        f"All-Input Mode — Dimension Statistics  |  input: \"{word}\"  |  Layer {layer_index}\n"
        "(Actual signed activation values — no normalization)",
        fontsize=12,
    )

    ax_pos, ax_neg, ax_combined = axes[0], axes[1], axes[2]

    # ── Positive dims (bars go right) ──────────────────────────────────────
    if positive_dims:
        pos_labels = [str(d["dim"]) for d in positive_dims]
        pos_vals = [float(d["value"]) for d in positive_dims]
        pos_y = np.arange(len(pos_labels))
        ax_pos.barh(pos_y, pos_vals, color=_POS_COLOR, edgecolor="white", linewidth=0.4)
        ax_pos.set_yticks(pos_y, labels=pos_labels, fontsize=7)
        ax_pos.invert_yaxis()
        ax_pos.axvline(0, color="black", linewidth=0.8)
        ax_pos.set_title("Top Positive Dims (Signed Values)", fontsize=10)
        ax_pos.set_xlabel("Activation Value (positive → right)", fontsize=9)
        x_max = max(pos_vals)
        for idx, val in enumerate(pos_vals):
            ax_pos.text(val + x_max * 0.01, idx, f"+{val:.3f}", va="center", fontsize=7)
    else:
        ax_pos.axis("off")
        ax_pos.set_title("Top Positive Dims — No data")

    # ── Negative dims (bars go left) ────────────────────────────────────────
    if negative_dims:
        neg_labels = [str(d["dim"]) for d in negative_dims]
        neg_vals = [float(d["value"]) for d in negative_dims]
        neg_y = np.arange(len(neg_labels))
        ax_neg.barh(neg_y, neg_vals, color=_NEG_COLOR, edgecolor="white", linewidth=0.4)
        ax_neg.set_yticks(neg_y, labels=neg_labels, fontsize=7)
        ax_neg.invert_yaxis()
        ax_neg.axvline(0, color="black", linewidth=0.8)
        ax_neg.set_title("Top Negative Dims (Signed Values)", fontsize=10)
        ax_neg.set_xlabel("Activation Value (negative → left)", fontsize=9)
        x_min = min(neg_vals)
        for idx, val in enumerate(neg_vals):
            ax_neg.text(max(val + x_min * 0.01, x_min * 1.05), idx, f"{val:.3f}", va="center", ha="right", fontsize=7)
    else:
        ax_neg.axis("off")
        ax_neg.set_title("Top Negative Dims — No data")

    # ── Combined signed chart (all top dims sorted by abs value) ────────────
    all_dims = sorted(top_dims, key=lambda d: abs(d["value"]), reverse=True)
    if all_dims:
        comb_labels = [str(d["dim"]) for d in all_dims]
        comb_vals = [float(d["value"]) for d in all_dims]
        comb_colors = [_POS_COLOR if v >= 0 else _NEG_COLOR for v in comb_vals]
        comb_y = np.arange(len(comb_labels))
        ax_combined.barh(comb_y, comb_vals, color=comb_colors, edgecolor="white", linewidth=0.4)
        ax_combined.set_yticks(comb_y, labels=comb_labels, fontsize=7)
        ax_combined.invert_yaxis()
        ax_combined.axvline(0, color="black", linewidth=0.8)
        ax_combined.set_title("All Top Dims — Signed Values (sorted by |value|)", fontsize=10)
        ax_combined.set_xlabel("Signed Activation Value", fontsize=9)
        x_range = max(abs(v) for v in comb_vals) if comb_vals else 1.0
        for idx, val in enumerate(comb_vals):
            offset = x_range * 0.01
            ha = "left" if val >= 0 else "right"
            x_pos = val + (offset if val >= 0 else -offset)
            ax_combined.text(x_pos, idx, f"{val:+.3f}", va="center", ha=ha, fontsize=7)
        pos_patch = mpatches.Patch(color=_POS_COLOR, label="Positive")
        neg_patch = mpatches.Patch(color=_NEG_COLOR, label="Negative")
        ax_combined.legend(handles=[pos_patch, neg_patch], fontsize=8, loc="lower right")
    else:
        ax_combined.axis("off")
        ax_combined.set_title("All Top Dims — No data")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_mode_comparison(
    per_word_average_rows: list[dict[str, Any]],
    all_input_top_dims: list[dict[str, Any]],
    output_path: str | Path,
    top_k: int = 20,
    layer_index: int = 8,
    dpi: int = 160,
) -> None:
    """Side-by-side comparison of dimension signatures for both input modes.

    Left column: top dims from per-word mode (by occurrence count, signed direction indicated by color).
    Right column: top dims from all-input mode (signed values).
    """
    _configure_font_fallback()

    # Per-word: pick top-k by occurrence count (any direction)
    sorted_per_word = sorted(
        per_word_average_rows,
        key=lambda r: int(r.get("appearance_count", 0)),
        reverse=True,
    )[:top_k]

    # All-input: top-k by abs value
    sorted_all_input = sorted(all_input_top_dims, key=lambda d: abs(d["value"]), reverse=True)[:top_k]

    fig, axes = plt.subplots(1, 2, figsize=(20, max(8, top_k * 0.55)), dpi=dpi)
    fig.suptitle(
        f"Input Mode Comparison — Layer {layer_index} | Left: per-word  Right: all-input\n"
        "(Raw signed values — no normalization)",
        fontsize=12,
    )

    ax_pw, ax_ai = axes[0], axes[1]

    # ── Per-word: bar = mean_abs_value, text = count + mean + max ───────────
    if sorted_per_word:
        pw_labels = [f"d{r['dim']} ({'pos' if r['group'] == 'max' else 'neg'})" for r in sorted_per_word]
        pw_means = [float(r.get("mean_abs_value", 0.0)) for r in sorted_per_word]
        pw_counts = [int(r.get("appearance_count", 0)) for r in sorted_per_word]
        pw_maxes = [float(r.get("max_abs_value", 0.0)) for r in sorted_per_word]
        pw_colors = [_POS_COLOR if r["group"] == "max" else _NEG_COLOR for r in sorted_per_word]
        y = np.arange(len(pw_labels))
        ax_pw.barh(y, pw_means, color=pw_colors, edgecolor="white", linewidth=0.4)
        ax_pw.set_yticks(y, labels=pw_labels, fontsize=7)
        ax_pw.invert_yaxis()
        ax_pw.axvline(0, color="black", linewidth=0.6)
        ax_pw.set_title("Per-Word Mode: Top Dims by Occurrence\n(bar = mean |activation|)", fontsize=10)
        ax_pw.set_xlabel("Mean |Activation Value| (raw)", fontsize=9)
        x_max = max(pw_means) if pw_means else 1.0
        for idx, (mean_val, count, max_val) in enumerate(zip(pw_means, pw_counts, pw_maxes)):
            ax_pw.text(
                mean_val + x_max * 0.01,
                idx,
                f"n={count}  mean={mean_val:.2f}  max={max_val:.2f}",
                va="center",
                fontsize=6.5,
            )
    else:
        ax_pw.axis("off")
        ax_pw.set_title("Per-Word Mode — No data")

    # ── All-input: bar = signed value ───────────────────────────────────────
    if sorted_all_input:
        ai_labels = [str(d["dim"]) for d in sorted_all_input]
        ai_vals = [float(d["value"]) for d in sorted_all_input]
        ai_colors = [_POS_COLOR if v >= 0 else _NEG_COLOR for v in ai_vals]
        y = np.arange(len(ai_labels))
        ax_ai.barh(y, ai_vals, color=ai_colors, edgecolor="white", linewidth=0.4)
        ax_ai.set_yticks(y, labels=ai_labels, fontsize=7)
        ax_ai.invert_yaxis()
        ax_ai.axvline(0, color="black", linewidth=0.8)
        ax_ai.set_title("All-Input Mode: Top Dims (Signed Values)\n(sorted by |activation|)", fontsize=10)
        ax_ai.set_xlabel("Signed Activation Value (raw)", fontsize=9)
        x_range = max(abs(v) for v in ai_vals) if ai_vals else 1.0
        for idx, val in enumerate(ai_vals):
            offset = x_range * 0.01
            ha = "left" if val >= 0 else "right"
            ax_ai.text(val + (offset if val >= 0 else -offset), idx, f"{val:+.3f}", va="center", ha=ha, fontsize=7)
        pos_patch = mpatches.Patch(color=_POS_COLOR, label="Positive")
        neg_patch = mpatches.Patch(color=_NEG_COLOR, label="Negative")
        ax_ai.legend(handles=[pos_patch, neg_patch], fontsize=8, loc="lower right")
    else:
        ax_ai.axis("off")
        ax_ai.set_title("All-Input Mode — No data")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_three_mode_comparison(
    per_word_average_rows: list[dict[str, Any]],
    all_input_top_dims: list[dict[str, Any]],
    positional_average_rows: list[dict[str, Any]],
    output_path: str | Path,
    top_k: int = 20,
    layer_index: int = 8,
    dpi: int = 160,
) -> None:
    """Three-panel comparison of all three input modes.

    Left  : per-word mode — top dims by occurrence count (bar = mean |activation|).
    Centre: all-input mode — top dims by |signed value| (actual signed values).
    Right : positional mode — top dims by occurrence count (bar = mean |activation|).

    All values are raw, not normalised.
    """
    _configure_font_fallback()

    sorted_per_word = sorted(
        per_word_average_rows,
        key=lambda r: int(r.get("appearance_count", 0)),
        reverse=True,
    )[:top_k]

    sorted_all_input = sorted(all_input_top_dims, key=lambda d: abs(d["value"]), reverse=True)[:top_k]

    sorted_positional = sorted(
        positional_average_rows,
        key=lambda r: int(r.get("appearance_count", 0)),
        reverse=True,
    )[:top_k]

    fig, axes = plt.subplots(1, 3, figsize=(30, max(8, top_k * 0.55)), dpi=dpi)
    fig.suptitle(
        f"Three-Mode Comparison — Layer {layer_index}\n"
        "Left: per-word (isolated)  |  Centre: all-input (last token)  |  Right: positional (in-sequence)\n"
        "(Raw signed / absolute activation values — no normalization)",
        fontsize=12,
    )

    ax_pw, ax_ai, ax_pos = axes[0], axes[1], axes[2]

    def _draw_per_word_panel(ax: plt.Axes, rows: list[dict[str, Any]], title: str) -> None:
        if not rows:
            ax.axis("off")
            ax.set_title(title + " — No data")
            return
        labels = [f"d{r['dim']} ({'pos' if r['group'] == 'max' else 'neg'})" for r in rows]
        means = [float(r.get("mean_abs_value", 0.0)) for r in rows]
        counts = [int(r.get("appearance_count", 0)) for r in rows]
        maxes = [float(r.get("max_abs_value", 0.0)) for r in rows]
        colors = [_POS_COLOR if r["group"] == "max" else _NEG_COLOR for r in rows]
        y = np.arange(len(labels))
        ax.barh(y, means, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_yticks(y, labels=labels, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Mean |Activation| (raw)", fontsize=9)
        x_max = max(means) if means else 1.0
        for idx, (mean_val, count, max_val) in enumerate(zip(means, counts, maxes)):
            ax.text(
                mean_val + x_max * 0.01,
                idx,
                f"n={count}  mu={mean_val:.2f}  max={max_val:.2f}",
                va="center",
                fontsize=6.5,
            )

    _draw_per_word_panel(
        ax_pw,
        sorted_per_word,
        "Mode 1: Per-Word\nTop dims by occurrence\n(each word isolated)",
    )

    # Centre: all-input signed values
    if sorted_all_input:
        ai_labels = [str(d["dim"]) for d in sorted_all_input]
        ai_vals = [float(d["value"]) for d in sorted_all_input]
        ai_colors = [_POS_COLOR if v >= 0 else _NEG_COLOR for v in ai_vals]
        y = np.arange(len(ai_labels))
        ax_ai.barh(y, ai_vals, color=ai_colors, edgecolor="white", linewidth=0.4)
        ax_ai.set_yticks(y, labels=ai_labels, fontsize=7)
        ax_ai.invert_yaxis()
        ax_ai.axvline(0, color="black", linewidth=0.8)
        ax_ai.set_title(
            "Mode 2: All-Input\nTop dims by |value|\n(last token, all words as context)",
            fontsize=10,
        )
        ax_ai.set_xlabel("Signed Activation Value (raw)", fontsize=9)
        x_range = max(abs(v) for v in ai_vals) if ai_vals else 1.0
        for idx, val in enumerate(ai_vals):
            offset = x_range * 0.01
            ha = "left" if val >= 0 else "right"
            ax_ai.text(val + (offset if val >= 0 else -offset), idx, f"{val:+.3f}", va="center", ha=ha, fontsize=7)
        pos_patch = mpatches.Patch(color=_POS_COLOR, label="Positive")
        neg_patch = mpatches.Patch(color=_NEG_COLOR, label="Negative")
        ax_ai.legend(handles=[pos_patch, neg_patch], fontsize=8, loc="lower right")
    else:
        ax_ai.axis("off")
        ax_ai.set_title("Mode 2: All-Input — No data")

    _draw_per_word_panel(
        ax_pos,
        sorted_positional,
        "Mode 3: Positional\nTop dims by occurrence\n(each word in shared sequence context)",
    )

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_full_4096_mean_landscape(
    per_word_rows: list[dict[str, Any]],
    all_input_rows: list[dict[str, Any]],
    positional_rows: list[dict[str, Any]],
    output_path: str | Path,
    layer_index: int = 8,
    dpi: int = 160,
) -> None:
    """Visualize the signed mean value for every hidden dimension.

    The chart keeps all dimensions (4096 for Llama-3-8B) instead of only top-k
    dimensions. Values are raw signed means, not normalized.
    """
    _configure_font_fallback()

    series = [
        ("Mode 1: per-word mean", per_word_rows, "#e07a5f"),
        ("Mode 2: all-input last-token", all_input_rows, "#81b29a"),
        ("Mode 3: positional mean", positional_rows, "#3d405b"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(26, 13), dpi=dpi, sharex=True)
    fig.suptitle(
        f"Full Hidden-State Landscape — Layer {layer_index} (all 4096 dimensions)\n"
        "Y axis = raw signed mean activation value; zero line separates positive and negative directions",
        fontsize=13,
    )

    for ax, (title, rows, color) in zip(axes, series):
        dims, values = _rows_to_dim_values(rows, "mean_value")
        ax.plot(dims, values, color=color, linewidth=0.8, alpha=0.95)
        ax.fill_between(dims, 0.0, values, where=values >= 0, color="#e07a5f", alpha=0.22, interpolate=True)
        ax.fill_between(dims, 0.0, values, where=values < 0, color="#3d405b", alpha=0.22, interpolate=True)
        ax.axhline(0.0, color="#111827", linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("Mean value", fontsize=9)
        ax.grid(axis="y", alpha=0.18)
        _annotate_extreme_dims(ax, dims, values, top_k=5)

    axes[-1].set_xlabel("Hidden dimension index", fontsize=10)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_full_4096_top_signed_bars(
    per_word_rows: list[dict[str, Any]],
    all_input_rows: list[dict[str, Any]],
    positional_rows: list[dict[str, Any]],
    output_path: str | Path,
    top_k: int = 20,
    layer_index: int = 8,
    dpi: int = 160,
) -> None:
    """Bar charts for strongest full-4096 signed mean dimensions in each mode."""
    _configure_font_fallback()

    panels = [
        ("Mode 1: Per-word average across words", per_word_rows),
        ("Mode 2: All-input last token", all_input_rows),
        ("Mode 3: Positional average across sequence slots", positional_rows),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(30, max(9, top_k * 0.55)), dpi=dpi)
    fig.suptitle(
        f"Full 4096-Dim Top Signed Mean Dimensions — Layer {layer_index}\n"
        "Bars use raw signed mean_value; selected by largest |mean_value|",
        fontsize=13,
    )

    for ax, (title, rows) in zip(axes, panels):
        selected = sorted(rows, key=lambda r: abs(float(r.get("mean_value", 0.0))), reverse=True)[:top_k]
        if not selected:
            ax.axis("off")
            ax.set_title(title + " — no data")
            continue

        labels = [f"d{row['dim']}" for row in selected]
        values = [float(row.get("mean_value", 0.0)) for row in selected]
        mean_abs = [float(row.get("mean_abs_value", abs(v))) for row, v in zip(selected, values)]
        counts = [int(row.get("sample_count", 1)) for row in selected]
        colors = [_POS_COLOR if value >= 0 else _NEG_COLOR for value in values]
        y = np.arange(len(labels))
        ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_yticks(y, labels=labels, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0.0, color="#111827", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Signed mean_value (raw)", fontsize=9)
        x_range = max(abs(value) for value in values) if values else 1.0
        for idx, (value, abs_value, count) in enumerate(zip(values, mean_abs, counts)):
            ha = "left" if value >= 0 else "right"
            offset = x_range * 0.012
            ax.text(
                value + (offset if value >= 0 else -offset),
                idx,
                f"{value:+.3f} | mean|x|={abs_value:.3f} | n={count}",
                va="center",
                ha=ha,
                fontsize=6.5,
            )
        ax.grid(axis="x", alpha=0.18)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _rows_to_dim_values(rows: list[dict[str, Any]], value_key: str) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(rows, key=lambda row: int(row.get("dim", 0)))
    dims = np.asarray([int(row.get("dim", 0)) for row in ordered], dtype=np.int32)
    values = np.asarray([float(row.get(value_key, 0.0)) for row in ordered], dtype=np.float32)
    return dims, values


def _annotate_extreme_dims(ax: plt.Axes, dims: np.ndarray, values: np.ndarray, top_k: int = 5) -> None:
    if dims.size == 0:
        return
    selected_indices = np.argsort(np.abs(values))[::-1][:top_k]
    y_span = float(values.max() - values.min()) if values.size else 1.0
    offset = max(y_span * 0.035, 0.025)
    for idx in selected_indices:
        dim = int(dims[idx])
        value = float(values[idx])
        va = "bottom" if value >= 0 else "top"
        ax.scatter([dim], [value], s=20, color="#111827", zorder=4)
        ax.text(
            dim,
            value + (offset if value >= 0 else -offset),
            f"d{dim}\n{value:+.2f}",
            ha="center",
            va=va,
            fontsize=6.5,
            color="#111827",
        )
