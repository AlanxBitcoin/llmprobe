from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Render single-word analysis dashboard images from prepared result payloads.
# - Visualization module only; no model execution logic here.

import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(".mplconfig")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir.resolve())

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _configure_font_fallback() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def render_single_word_dashboard(
    word: str,
    layer_index: int,
    top_dims: list[dict],
    matches: dict,
    attribute_predictions: dict | None,
    dimension_semantics: list[dict] | None,
    token_profile: dict | None,
    output_path: str | Path,
    dpi: int = 160,
    cmap: str = "magma",
) -> None:
    _configure_font_fallback()
    fig, axes = plt.subplots(2, 3, figsize=(22, 10), dpi=dpi)
    fig.suptitle(f"Single Token Analysis: {word} | Layer {layer_index}", fontsize=14)
    match_domain = (token_profile or {}).get("match_domain", "lexical")
    type_title = "Concept Type Similarity" if match_domain == "lexical" else "Symbolic Attribute Types"
    value_title = "Concept Value Similarity" if match_domain == "lexical" else "Symbolic Attribute Values"
    attr_title = "Attribute Probe Card" if match_domain == "lexical" else "Non-Lexical Attribute Card"
    ax_pos = axes[0, 0]
    ax_neg = axes[0, 1]
    ax_types = axes[0, 2]
    ax_values = axes[1, 0]
    ax_attr = axes[1, 1]
    ax_sem = axes[1, 2]

    positive_dims = [item for item in top_dims if item["value"] >= 0]
    negative_dims = [item for item in top_dims if item["value"] < 0]

    pos_labels = [str(item["dim"]) for item in positive_dims]
    pos_values = [float(item["value"]) for item in positive_dims]
    if pos_values:
        pos_y = np.arange(len(pos_labels))
        ax_pos.barh(pos_y, pos_values, color="#e07a5f")
        ax_pos.set_yticks(pos_y, labels=pos_labels)
        ax_pos.invert_yaxis()
        ax_pos.axvline(0, color="black", linewidth=0.8)
        ax_pos.set_title("Top Positive Dims (Signed Values)")
        ax_pos.set_xlabel("Activation Value (Positive)")
        x_max = max(pos_values) if pos_values else 1.0
        for idx, value in enumerate(pos_values):
            ax_pos.text(min(value + x_max * 0.01, x_max * 1.05), idx, f"+{value:.3f}", va="center", fontsize=8)
    else:
        ax_pos.axis("off")
        ax_pos.set_title("Top Positive Dims")

    neg_labels = [str(item["dim"]) for item in negative_dims]
    neg_values = [float(item["value"]) for item in negative_dims]
    if neg_values:
        neg_y = np.arange(len(neg_labels))
        ax_neg.barh(neg_y, neg_values, color="#3d405b")
        ax_neg.set_yticks(neg_y, labels=neg_labels)
        ax_neg.invert_yaxis()
        ax_neg.axvline(0, color="black", linewidth=0.8)
        ax_neg.set_title("Top Negative Dims (Signed Values)")
        ax_neg.set_xlabel("Activation Value (Negative)")
        x_min = min(neg_values) if neg_values else -1.0
        for idx, value in enumerate(neg_values):
            ax_neg.text(max(value + x_min * 0.01, x_min * 1.05), idx, f"{value:.3f}", va="center", ha="right", fontsize=8)
    else:
        ax_neg.axis("off")
        ax_neg.set_title("Top Negative Dims")

    type_terms = [item["term"] for item in matches["types"]]
    if type_terms:
        type_scores = np.array([[abs(float(item["score"])) for item in matches["types"]]])
        sns.heatmap(type_scores, annot=True, fmt=".3f", cmap=cmap, xticklabels=type_terms, yticklabels=["types"], ax=ax_types)
        ax_types.set_title(type_title)
        ax_types.tick_params(axis="x", rotation=45)
    else:
        ax_types.axis("off")
        ax_types.set_title(type_title)
        ax_types.text(0.5, 0.5, "No attribute-type matching", ha="center", va="center", fontsize=10)

    value_terms = [item["term"] for item in matches["values"]]
    if value_terms:
        value_scores = np.array([[abs(float(item["score"])) for item in matches["values"]]])
        sns.heatmap(value_scores, annot=True, fmt=".3f", cmap=cmap, xticklabels=value_terms, yticklabels=["values"], ax=ax_values)
        ax_values.set_title(value_title)
        ax_values.tick_params(axis="x", rotation=45)
    else:
        ax_values.axis("off")
        ax_values.set_title(value_title)
        ax_values.text(0.5, 0.5, "No attribute-value matching", ha="center", va="center", fontsize=10)

    ax_attr.set_title(attr_title)
    labels = []
    scores = []
    colors = []
    core_attribute_order = ["category", "color", "shape", "taste"]
    core_lines: list[str] = []
    if attribute_predictions:
        for attr_name in core_attribute_order:
            payload = attribute_predictions.get(attr_name)
            if not payload:
                continue
            top = payload.get("top_prediction") or {}
            label = top.get("label", "-")
            score = top.get("score", 0.0)
            core_lines.append(f"{attr_name:<8} {label:<12} {score:>5.3f}")
        for attr_name, payload in attribute_predictions.items():
            top = payload.get("top_prediction") or {}
            label = top.get("label", "-")
            score = top.get("score", 0.0)
            labels.append(f"{attr_name}: {label}")
            scores.append(score)
            colors.append("#3d405b" if attr_name in core_attribute_order else "#81b29a")
    else:
        labels.append("No attribute predictions")
        scores.append(0.0)
        colors.append("#c9b79c")
    y_pos = np.arange(len(labels))
    ax_attr.barh(y_pos, scores, color=colors)
    ax_attr.set_yticks(y_pos, labels=labels)
    ax_attr.set_xlim(0, 1.0)
    ax_attr.invert_yaxis()
    ax_attr.set_xlabel("Top-1 Probability")
    for idx, score in enumerate(scores):
        ax_attr.text(min(score + 0.02, 0.98), idx, f"{score:.3f}", va="center", fontsize=9)
    if core_lines:
        ax_attr.text(
            0.02,
            0.98,
            "Core Attributes\n" + "\n".join(core_lines),
            transform=ax_attr.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
            color="#111827",
            bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f2cc8f", "edgecolor": "#9c6644", "alpha": 0.95},
        )
    elif token_profile and not token_profile.get("attribute_probe_enabled", True):
        ax_attr.text(
            0.02,
            0.98,
            "Attribute Probe Skipped\n"
            f"class: {token_profile.get('token_class', '-')}\n"
            f"{token_profile.get('reason', '')}",
            transform=ax_attr.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
            color="#111827",
            bbox={"boxstyle": "round,pad=0.55", "facecolor": "#dbeafe", "edgecolor": "#2563eb", "alpha": 0.95},
        )

    ax_sem.set_title("Dimension -> Concept Network")
    _draw_dimension_semantic_network(ax_sem, dimension_semantics)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _draw_dimension_semantic_network(ax, dimension_semantics: list[dict] | None) -> None:
    ax.axis("off")
    if not dimension_semantics:
        ax.text(
            0.02,
            0.98,
            "No dimension semantics available.",
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.55", "facecolor": "#eef2ff", "edgecolor": "#6366f1", "alpha": 0.95},
        )
        return

    dims = dimension_semantics[:6]
    concept_names: list[str] = []
    type_names: list[str] = []
    for row in dims:
        for concept in row.get("aligned_concepts", [])[:2]:
            if concept["term"] not in concept_names:
                concept_names.append(concept["term"])
        for concept_type in row.get("aligned_types", [])[:2]:
            if concept_type["term"] not in type_names:
                type_names.append(concept_type["term"])

    concept_names = concept_names[:6]
    type_names = type_names[:4]

    x_dim, x_concept, x_type = 0.12, 0.50, 0.86
    dim_y = np.linspace(0.88, 0.16, max(len(dims), 1))
    concept_y = np.linspace(0.88, 0.16, max(len(concept_names), 1))
    type_y = np.linspace(0.82, 0.22, max(len(type_names), 1))

    concept_pos = {name: concept_y[idx] for idx, name in enumerate(concept_names)}
    type_pos = {name: type_y[idx] for idx, name in enumerate(type_names)}

    for idx, row in enumerate(dims):
        dim_name = f"d{row['dim']} {'+' if row['direction'] == 'positive' else '-'}"
        dim_color = "#e07a5f" if row["direction"] == "positive" else "#3d405b"
        y = dim_y[idx]
        ax.scatter([x_dim], [y], s=560, color=dim_color, edgecolors="white", linewidths=1.5, zorder=3)
        ax.text(x_dim, y, dim_name, ha="center", va="center", color="white", fontsize=8, weight="bold")

        for concept in row.get("aligned_concepts", [])[:2]:
            if concept["term"] not in concept_pos:
                continue
            cy = concept_pos[concept["term"]]
            width = 1.2 + 4.0 * min(max(abs(float(concept.get("alignment", 0.0))), 0.0), 1.0)
            ax.plot([x_dim + 0.035, x_concept - 0.035], [y, cy], color=dim_color, alpha=0.35, linewidth=width, zorder=1)

        for concept_type in row.get("aligned_types", [])[:2]:
            if concept_type["term"] not in type_pos:
                continue
            ty = type_pos[concept_type["term"]]
            width = 1.0 + 3.0 * min(max(abs(float(concept_type.get("score", 0.0))), 0.0), 1.0)
            ax.plot([x_dim + 0.035, x_type - 0.035], [y, ty], color="#9c6644", alpha=0.15, linewidth=width, zorder=1)

    for name, y in concept_pos.items():
        ax.scatter([x_concept], [y], s=500, color="#81b29a", edgecolors="white", linewidths=1.5, zorder=3)
        ax.text(x_concept, y, name, ha="center", va="center", color="white", fontsize=8, weight="bold")

    for name, y in type_pos.items():
        ax.scatter([x_type], [y], s=600, color="#f2cc8f", edgecolors="#9c6644", linewidths=1.2, zorder=3)
        ax.text(x_type, y, name, ha="center", va="center", color="#111827", fontsize=8, weight="bold")

    for row in dims:
        for concept in row.get("aligned_concepts", [])[:2]:
            parent = concept.get("parent_type")
            if concept["term"] not in concept_pos or parent not in type_pos:
                continue
            cy = concept_pos[concept["term"]]
            ty = type_pos[parent]
            width = 1.0 + 3.0 * min(max(abs(float(concept.get("alignment", 0.0))), 0.0), 1.0)
            ax.plot([x_concept + 0.035, x_type - 0.035], [cy, ty], color="#2a9d8f", alpha=0.28, linewidth=width, zorder=2)

    ax.text(x_dim, 0.98, "dims", ha="center", va="top", fontsize=10, weight="bold")
    ax.text(x_concept, 0.98, "concept values", ha="center", va="top", fontsize=10, weight="bold")
    ax.text(x_type, 0.98, "concept types", ha="center", va="top", fontsize=10, weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
