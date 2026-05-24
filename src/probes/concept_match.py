from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Provide concept matching utilities for hidden-state interpretation.
# - Concept catalogs are data-driven and loaded from data/ files.
# - Keep matching logic reusable by pipeline/study layers.

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..utils.utils import load_yaml, read_lines


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


class ConceptMatcher:
    def __init__(
        self,
        bundle,
        concept_type_file: str | Path,
        concept_value_file: str | Path,
        catalog_path: str | Path = "data/concept_catalog.yaml",
    ) -> None:
        self.bundle = bundle
        self.concept_types = read_lines(concept_type_file)
        self.concept_values = read_lines(concept_value_file)
        self.catalog = load_yaml(catalog_path)
        self.value_to_type = self._build_value_to_type(self.catalog.get("types", {}))
        self.type_vectors = self._embed_terms(self.concept_types)
        self.value_vectors = self._embed_terms(self.concept_values)
        self.type_prototype_vectors = self._build_type_prototypes(self.catalog.get("types", {}))

    @staticmethod
    def _build_value_to_type(types: dict[str, list[str]]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for concept_type, values in types.items():
            for value in values:
                mapping[value] = concept_type
        return mapping

    def _embed_terms(self, terms: list[str]) -> dict[str, np.ndarray]:
        tokenizer = self.bundle.tokenizer
        model = self.bundle.model
        device = next(model.parameters()).device
        vectors: dict[str, np.ndarray] = {}
        for term in terms:
            encoded = tokenizer(term, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded, output_hidden_states=True)
            term_vector = outputs.hidden_states[-1][0, -1, :].detach().float().cpu().numpy()
            vectors[term] = l2_normalize(term_vector)
        return vectors

    def _build_type_prototypes(self, catalog_types: dict[str, list[str]]) -> dict[str, np.ndarray]:
        prototypes: dict[str, np.ndarray] = {}
        for concept_type, values in catalog_types.items():
            members = [self.value_vectors[value] for value in values if value in self.value_vectors]
            if concept_type in self.type_vectors:
                members.append(self.type_vectors[concept_type])
            if not members:
                continue
            stacked = np.stack(members, axis=0)
            prototypes[concept_type] = l2_normalize(stacked.mean(axis=0))
        return prototypes

    def match(self, hidden_vector: np.ndarray, top_k: int) -> dict[str, list[dict[str, Any]]]:
        normalized = l2_normalize(hidden_vector)
        top_values = self._top_matches(normalized, self.value_vectors, top_k)
        return {
            "types": self._top_matches(normalized, self.type_prototype_vectors or self.type_vectors, top_k),
            "values": self._annotate_values(top_values),
            "catalog_types": self._aggregate_type_scores(top_values, top_k),
        }

    def explain_top_dims(self, vector: np.ndarray, top_dims: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
        explanations: list[dict[str, Any]] = []
        for item in top_dims:
            dim = int(item["dim"])
            value = float(item["value"])
            direction = "positive" if value >= 0 else "negative"
            top_concepts, top_types = self.explain_dimension_axis(dim, direction=direction, scale=abs(value), top_k=top_k)
            explanations.append(
                {
                    "dim": dim,
                    "value": value,
                    "direction": direction,
                    "aligned_concepts": top_concepts,
                    "aligned_types": top_types,
                }
            )
        return explanations

    def explain_dimension_axis(
        self,
        dim: int,
        direction: str = "positive",
        scale: float = 1.0,
        top_k: int = 3,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        sign = 1.0 if direction == "positive" else -1.0
        concept_scores = []
        for term, concept_vector in self.value_vectors.items():
            coord = float(concept_vector[dim])
            alignment = sign * scale * coord
            concept_scores.append(
                {
                    "term": term,
                    "parent_type": self.value_to_type.get(term, "unmapped"),
                    "coord": coord,
                    "alignment": alignment,
                }
            )
        concept_scores.sort(key=lambda row: row["alignment"], reverse=True)
        top_concepts = concept_scores[:top_k]
        type_totals: dict[str, float] = {}
        for row in concept_scores[: max(top_k * 3, top_k)]:
            parent = row["parent_type"]
            type_totals[parent] = type_totals.get(parent, 0.0) + row["alignment"]
        top_types = [
            {"term": term, "score": score}
            for term, score in sorted(type_totals.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        ]
        return top_concepts, top_types

    def _top_matches(
        self,
        hidden_vector: np.ndarray,
        concept_vectors: dict[str, np.ndarray],
        top_k: int,
    ) -> list[dict[str, Any]]:
        scored = []
        for term, concept_vector in concept_vectors.items():
            score = float(np.dot(hidden_vector, concept_vector))
            scored.append({"term": term, "score": score})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def _annotate_values(self, top_values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        annotated = []
        for item in top_values:
            annotated.append(
                {
                    "term": item["term"],
                    "score": item["score"],
                    "parent_type": self.value_to_type.get(item["term"], "unmapped"),
                }
            )
        return annotated

    def _aggregate_type_scores(self, top_values: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for item in top_values:
            parent_type = self.value_to_type.get(item["term"])
            if not parent_type:
                continue
            totals[parent_type] = totals.get(parent_type, 0.0) + float(item["score"])
            counts[parent_type] = counts.get(parent_type, 0) + 1
        merged = [
            {
                "term": concept_type,
                "score": totals[concept_type],
                "match_count": counts[concept_type],
            }
            for concept_type in totals
        ]
        merged.sort(key=lambda item: item["score"], reverse=True)
        return merged[:top_k]
