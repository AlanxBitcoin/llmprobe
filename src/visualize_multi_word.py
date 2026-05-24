from __future__ import annotations

import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(".mplconfig")
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir.resolve())

import matplotlib.pyplot as plt
import networkx as nx
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


def render_multi_word_dashboard(
    words: list[str],
    layer_strengths: np.ndarray,
    salient_paths: list[tuple[str, str, float]],
    output_path: str | Path,
    dpi: int = 160,
    cmap: str = "magma",
) -> None:
    _configure_font_fallback()
    fig = plt.figure(figsize=(18, 6), dpi=dpi)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_graph = fig.add_subplot(grid[0, 1])
    fig.suptitle(f"Multi Word Flow: {' '.join(words)}", fontsize=14)

    sns.heatmap(
        layer_strengths,
        cmap=cmap,
        ax=ax_heat,
        xticklabels=words,
        yticklabels=[f"L{i}" for i in range(layer_strengths.shape[0])],
    )
    ax_heat.set_title("Layer / Word Activation Heatmap")

    graph = nx.DiGraph()
    for source, target, weight in salient_paths:
        graph.add_edge(source, target, weight=weight)
    positions = {}
    for layer_idx in range(layer_strengths.shape[0]):
        for word_idx, word in enumerate(words):
            node = f"L{layer_idx}:{word}"
            positions[node] = (layer_idx, -word_idx)
            graph.add_node(node)
    edge_weights = [graph[u][v].get("weight", 0.1) for u, v in graph.edges()]
    max_weight = max(edge_weights) if edge_weights else 1.0
    edge_widths = [max(0.6, (weight / max_weight) * 6.0) for weight in edge_weights]
    edge_colors = [plt.cm.cividis(weight / max_weight) for weight in edge_weights]
    node_strength = {}
    for layer_idx in range(layer_strengths.shape[0]):
        for word_idx, word in enumerate(words):
            node = f"L{layer_idx}:{word}"
            node_strength[node] = float(layer_strengths[layer_idx, word_idx])
    max_node_strength = max(node_strength.values()) if node_strength else 1.0
    node_colors = [plt.cm.magma(node_strength[node] / max_node_strength) for node in graph.nodes()]
    node_sizes = [420 + 780 * (node_strength[node] / max_node_strength) for node in graph.nodes()]
    nx.draw_networkx(
        graph,
        pos=positions,
        ax=ax_graph,
        node_size=node_sizes,
        font_size=8,
        width=edge_widths,
        arrows=True,
        node_color=node_colors,
        edge_color=edge_colors,
        arrowstyle="-|>",
        arrowsize=14,
    )
    if edge_weights:
        strongest = max(salient_paths, key=lambda item: item[2])
        ax_graph.set_title(f"Dominant Flow: {strongest[0]} -> {strongest[1]}")
    else:
        ax_graph.set_title("Dominant Flow")
    ax_graph.axis("off")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
