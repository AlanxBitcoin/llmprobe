from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from ..extract_hidden import extract_word_hidden_states
from ..utils import ensure_dir, write_csv, write_json


def load_attribute_rows(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        headers = fh.readline().strip().split(",")
        for line in fh:
            if not line.strip():
                continue
            values = line.rstrip("\n").split(",")
            if len(values) < len(headers):
                values.extend([""] * (len(headers) - len(values)))
            rows.append(dict(zip(headers, values)))
    return rows


def build_feature_bank(bundle, rows: list[dict[str, str]], target_layer: int) -> dict[str, np.ndarray]:
    bank: dict[str, np.ndarray] = {}
    for row in rows:
        hidden = extract_word_hidden_states(bundle, row["word"])
        bank[row["word"]] = np.asarray(hidden["layers"][target_layer]["vector"], dtype=np.float32)
    return bank


def train_attribute_probes(
    feature_bank: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if config is None:
        config = {"probe": {"attribute_probe": {
            "random_state": 42, "solver": "lbfgs", "max_iter": 3000,
            "test_size_ratio": 0.25, "min_class_count": 2, "min_total_samples": 8,
            "min_unique_classes": 2
        }}}
    
    probe_cfg = config.get("probe", {}).get("attribute_probe", {})
    random_state = probe_cfg.get("random_state", 42)
    solver = probe_cfg.get("solver", "lbfgs")
    max_iter = probe_cfg.get("max_iter", 3000)
    test_size_ratio = probe_cfg.get("test_size_ratio", 0.25)
    min_class_count = probe_cfg.get("min_class_count", 2)
    min_total_samples = probe_cfg.get("min_total_samples", 8)
    min_unique_classes = probe_cfg.get("min_unique_classes", 2)
    
    attribute_names = [name for name in rows[0].keys() if name != "word"]
    results: dict[str, Any] = {"attributes": {}}
    for attribute in attribute_names:
        labeled = [row for row in rows if row.get(attribute, "").strip()]
        labels = [row[attribute] for row in labeled]
        label_counts = {label: labels.count(label) for label in set(labels)}
        labeled = [row for row in labeled if label_counts[row[attribute]] >= min_class_count]
        labels = [row[attribute] for row in labeled]
        label_counts = {label: labels.count(label) for label in set(labels)}
        if len(set(labels)) < min_unique_classes or len(labeled) < min_total_samples or min(label_counts.values()) < min_class_count:
            continue
        features = np.stack([feature_bank[row["word"]] for row in labeled], axis=0)
        words = [row["word"] for row in labeled]
        class_count = len(set(labels))
        test_size = max(class_count, int(round(len(labeled) * test_size_ratio)))
        if test_size >= len(labeled):
            test_size = class_count
        x_train, x_test, y_train, y_test, _w_train, w_test = train_test_split(
            features,
            labels,
            words,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
        clf = LogisticRegression(max_iter=max_iter, solver=solver)
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        probs = clf.predict_proba(x_test)
        classes = list(clf.classes_)
        examples = []
        for word, truth, pred, prob_vec in zip(w_test, y_test, preds, probs):
            ranked = sorted(
                [{"label": label, "score": float(score)} for label, score in zip(classes, prob_vec)],
                key=lambda item: item["score"],
                reverse=True,
            )[:5]
            examples.append(
                {
                    "word": word,
                    "truth": truth,
                    "prediction": pred,
                    "correct": bool(truth == pred),
                    "top_predictions": ranked,
                }
            )
        results["attributes"][attribute] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "classes": classes,
            "support": len(labeled),
            "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
            "test_examples": examples,
            "coefficients_shape": list(clf.coef_.shape),
        }
    return results


def export_attribute_probe_results(output_dir: str | Path, results: dict[str, Any]) -> None:
    root = ensure_dir(output_dir)
    write_json(root / "attribute_probe_metrics.json", results)
    rows: list[dict[str, Any]] = []
    for attribute, payload in results.get("attributes", {}).items():
        rows.append(
            {
                "attribute": attribute,
                "accuracy": payload["accuracy"],
                "support": payload["support"],
                "classes": " | ".join(payload["classes"]),
                "coef_rows": payload["coefficients_shape"][0] if payload["coefficients_shape"] else "",
                "coef_cols": payload["coefficients_shape"][1] if payload["coefficients_shape"] else "",
            }
        )
    write_csv(root / "attribute_probe_summary.csv", rows)

    prediction_rows: list[dict[str, Any]] = []
    for attribute, payload in results.get("attributes", {}).items():
        for item in payload["test_examples"]:
            top1 = item["top_predictions"][0] if item["top_predictions"] else {"label": "", "score": ""}
            prediction_rows.append(
                {
                    "attribute": attribute,
                    "word": item["word"],
                    "truth": item["truth"],
                    "prediction": item["prediction"],
                    "correct": item["correct"],
                    "top1_label": top1["label"],
                    "top1_score": top1["score"],
                }
            )
    write_csv(root / "attribute_probe_predictions.csv", prediction_rows)


def fit_full_attribute_probes(
    feature_bank: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if config is None:
        config = {"probe": {"attribute_probe": {
            "random_state": 42, "solver": "lbfgs", "max_iter": 3000,
            "test_size_ratio": 0.25, "min_class_count": 2, "min_total_samples": 8,
            "min_unique_classes": 2
        }}}
    
    probe_cfg = config.get("probe", {}).get("attribute_probe", {})
    solver = probe_cfg.get("solver", "lbfgs")
    max_iter = probe_cfg.get("max_iter", 3000)
    min_class_count = probe_cfg.get("min_class_count", 2)
    min_total_samples = probe_cfg.get("min_total_samples", 8)
    min_unique_classes = probe_cfg.get("min_unique_classes", 2)
    
    attribute_names = [name for name in rows[0].keys() if name != "word"]
    fitted: dict[str, Any] = {}
    for attribute in attribute_names:
        labeled = [row for row in rows if row.get(attribute, "").strip()]
        labels = [row[attribute] for row in labeled]
        label_counts = {label: labels.count(label) for label in set(labels)}
        labeled = [row for row in labeled if label_counts[row[attribute]] >= min_class_count]
        labels = [row[attribute] for row in labeled]
        if len(set(labels)) < min_unique_classes or len(labeled) < min_total_samples:
            continue
        features = np.stack([feature_bank[row["word"]] for row in labeled], axis=0)
        clf = LogisticRegression(max_iter=max_iter, solver=solver)
        clf.fit(features, labels)
        fitted[attribute] = {
            "classifier": clf,
            "classes": list(clf.classes_),
            "support": len(labeled),
        }
    return fitted


def predict_word_attributes(
    bundle,
    fitted_probes: dict[str, Any],
    word: str,
    target_layer: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if config is None:
        config = {"analysis": {"top_k_predictions": 5}}
    
    top_k = config.get("analysis", {}).get("top_k_predictions", 5)
    
    hidden = extract_word_hidden_states(bundle, word)
    vector = np.asarray(hidden["layers"][target_layer]["vector"], dtype=np.float32).reshape(1, -1)
    predictions: dict[str, Any] = {}
    for attribute, payload in fitted_probes.items():
        clf = payload["classifier"]
        probs = clf.predict_proba(vector)[0]
        ranked = sorted(
            [{"label": label, "score": float(score)} for label, score in zip(payload["classes"], probs)],
            key=lambda item: item["score"],
            reverse=True,
        )[:top_k]
        predictions[attribute] = {
            "top_prediction": ranked[0] if ranked else None,
            "top_candidates": ranked,
            "support": payload["support"],
        }
    return {
        "word": word,
        "target_layer": target_layer,
        "predictions": predictions,
    }
