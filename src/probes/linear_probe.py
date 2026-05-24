from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from ..extract_hidden import extract_word_hidden_states
from ..utils import ensure_dir, write_csv, write_json


@dataclass
class ProbeDataset:
    words: list[str]
    labels: list[str]
    features: np.ndarray


def load_labeled_words(label_file: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(label_file).open("r", encoding="utf-8") as fh:
        headers = fh.readline().strip().split(",")
        for line in fh:
            if not line.strip():
                continue
            values = line.strip().split(",")
            rows.append(dict(zip(headers, values)))
    return rows


def build_probe_dataset(bundle, label_rows: list[dict[str, str]], target_layer: int) -> ProbeDataset:
    words: list[str] = []
    labels: list[str] = []
    vectors: list[np.ndarray] = []
    for row in label_rows:
        hidden = extract_word_hidden_states(bundle, row["word"])
        vector = np.asarray(hidden["layers"][target_layer]["vector"], dtype=np.float32)
        words.append(row["word"])
        labels.append(row["primary_label"])
        vectors.append(vector)
    return ProbeDataset(words=words, labels=labels, features=np.stack(vectors, axis=0))


def train_linear_probe(dataset: ProbeDataset, config: dict[str, Any] | None = None) -> dict[str, Any]:
    if config is None:
        config = {"probe": {"linear_probe": {"random_state": 42, "test_size": 0.2, "solver": "lbfgs", "max_iter": 3000}}}
    
    probe_cfg = config.get("probe", {}).get("linear_probe", {})
    random_state = probe_cfg.get("random_state", 42)
    test_size = probe_cfg.get("test_size", 0.2)
    solver = probe_cfg.get("solver", "lbfgs")
    max_iter = probe_cfg.get("max_iter", 3000)
    
    x_train, x_test, y_train, y_test, words_train, words_test = train_test_split(
        dataset.features,
        dataset.labels,
        dataset.words,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.labels,
    )
    classifier = LogisticRegression(
        max_iter=max_iter,
        solver=solver,
        n_jobs=None,
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    probabilities = classifier.predict_proba(x_test)
    label_order = list(classifier.classes_)
    per_word = []
    for word, truth, pred, probs in zip(words_test, y_test, predictions, probabilities):
        ranked = sorted(
            [{"label": label, "score": float(score)} for label, score in zip(label_order, probs)],
            key=lambda item: item["score"],
            reverse=True,
        )[:5]
        per_word.append(
            {
                "word": word,
                "truth": truth,
                "prediction": pred,
                "correct": bool(truth == pred),
                "top_predictions": ranked,
            }
        )
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=label_order)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classes": label_order,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "test_examples": per_word,
        "coefficients_shape": list(classifier.coef_.shape),
    }


def export_probe_results(output_dir: str | Path, results: dict[str, Any]) -> None:
    output_root = ensure_dir(output_dir)
    write_json(output_root / "probe_metrics.json", results)
    rows = []
    for item in results["test_examples"]:
        top1 = item["top_predictions"][0]["label"] if item["top_predictions"] else ""
        top1_score = item["top_predictions"][0]["score"] if item["top_predictions"] else ""
        rows.append(
            {
                "word": item["word"],
                "truth": item["truth"],
                "prediction": item["prediction"],
                "correct": item["correct"],
                "top1_label": top1,
                "top1_score": top1_score,
            }
        )
    write_csv(output_root / "probe_predictions.csv", rows)
