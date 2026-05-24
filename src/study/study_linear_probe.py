from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Study layer wrapper for linear probe experiment runs.
# - Compose config/model/probe modules and export artifacts.
# - Avoid embedding probe algorithm details here.

from pathlib import Path
from typing import Any

from ..config import load_config
from ..model_loader import get_model_bundle
from ..probes.linear_probe import (
    build_probe_dataset,
    export_probe_results,
    load_labeled_words,
    train_linear_probe,
)


def run_study(
    config: dict[str, Any] | None = None,
    *,
    config_path: str | Path = "configs/custom.yaml",
    label_file: str | Path = "data/word_labels.csv",
    output_dir: str | Path = "data/outputs/probe",
) -> dict[str, Any]:
    cfg = config or load_config(config_path)
    bundle = get_model_bundle(cfg)
    rows = load_labeled_words(label_file)
    target_layer = int(cfg["analysis"]["target_layer"])
    dataset = build_probe_dataset(bundle, rows, target_layer, config=cfg)
    results = train_linear_probe(dataset, config=cfg)
    export_probe_results(output_dir, results)
    return {"output_dir": str(output_dir), "metrics": results}
