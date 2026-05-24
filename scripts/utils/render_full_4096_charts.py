from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.visualize_color_experiment import (
    render_full_4096_mean_landscape,
    render_full_4096_top_signed_bars,
)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render charts from full 4096-dim stats CSV files.")
    parser.add_argument("--run-dir", default="data/outputs/color_words", help="Run directory containing full_4096 CSV files.")
    parser.add_argument("--layer", type=int, default=8, help="Layer index for chart titles.")
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    per_word_rows = load_rows(run_dir / "per_word_full_4096_dim_stats.csv")
    all_input_rows = load_rows(run_dir / "all_input_full_4096_dim_stats.csv")
    positional_rows = load_rows(run_dir / "positional_full_4096_dim_stats.csv")

    render_full_4096_mean_landscape(
        per_word_rows=per_word_rows,
        all_input_rows=all_input_rows,
        positional_rows=positional_rows,
        output_path=run_dir / "full_4096_mean_landscape.png",
        layer_index=args.layer,
        dpi=args.dpi,
    )
    render_full_4096_top_signed_bars(
        per_word_rows=per_word_rows,
        all_input_rows=all_input_rows,
        positional_rows=positional_rows,
        output_path=run_dir / "full_4096_top_signed_bars.png",
        layer_index=args.layer,
        dpi=args.dpi,
    )

    print(run_dir / "full_4096_mean_landscape.png")
    print(run_dir / "full_4096_top_signed_bars.png")


if __name__ == "__main__":
    main()
