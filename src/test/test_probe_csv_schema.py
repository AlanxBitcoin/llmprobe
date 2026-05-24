from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Keep probe CSV output columns stable for downstream consumers.

from pathlib import Path

from src.probes.attribute_probe import export_attribute_probe_results
from src.probes.linear_probe import export_probe_results


def _read_header(path: Path) -> list[str]:
    first = path.read_text(encoding="utf-8").splitlines()[0]
    return first.split(",")


def test_linear_probe_csv_columns_are_stable(tmp_path):
    results = {
        "test_examples": [
            {
                "word": "apple",
                "truth": "fruit",
                "prediction": "fruit",
                "correct": True,
                "top_predictions": [{"label": "fruit", "score": 0.91}],
            }
        ]
    }
    export_probe_results(tmp_path, results)
    headers = _read_header(tmp_path / "probe_predictions.csv")
    assert headers == ["word", "truth", "prediction", "correct", "top1_label", "top1_score"]


def test_attribute_probe_csv_columns_are_stable(tmp_path):
    results = {
        "attributes": {
            "color": {
                "accuracy": 0.8,
                "support": 10,
                "classes": ["red", "blue"],
                "coefficients_shape": [2, 4096],
                "test_examples": [
                    {
                        "word": "apple",
                        "truth": "red",
                        "prediction": "red",
                        "correct": True,
                        "top_predictions": [{"label": "red", "score": 0.9}],
                    }
                ],
            }
        }
    }
    export_attribute_probe_results(tmp_path, results)
    summary_headers = _read_header(tmp_path / "attribute_probe_summary.csv")
    pred_headers = _read_header(tmp_path / "attribute_probe_predictions.csv")
    assert summary_headers == ["attribute", "accuracy", "support", "classes", "coef_rows", "coef_cols"]
    assert pred_headers == ["attribute", "word", "truth", "prediction", "correct", "top1_label", "top1_score"]
