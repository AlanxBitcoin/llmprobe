from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.analyze_command_word_layers import (
    build_html,
    render_diff_heatmap,
    render_diff_top_chart,
    render_dim_frequency_chart,
    render_layer_top10_lines,
)
from src.utils.utils import write_text


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render charts from layer_dim_abs_diff_top10.csv.")
    parser.add_argument("--run-dir", default="data/outputs/command word/layer32_4096_diff")
    parser.add_argument("--left-name", default="command word.txt")
    parser.add_argument("--right-name", default="no command word.txt")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = load_rows(run_dir / "layer_dim_abs_diff_top10.csv")
    render_diff_top_chart(rows, run_dir / "layer_top1_abs_diff.png")
    render_diff_heatmap(rows, run_dir / "layer_top10_abs_diff_heatmap.png")
    render_layer_top10_lines(rows, run_dir / "layer_top10_abs_diff_lines.png")
    render_dim_frequency_chart(rows, run_dir / "top_diff_dim_frequency.png")
    write_text(run_dir / "index.html", build_html(run_dir, args.left_name, args.right_name))
    print(run_dir / "index.html")


if __name__ == "__main__":
    main()
