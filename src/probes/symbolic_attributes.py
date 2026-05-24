from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import load_yaml


class SymbolicAttributeRegistry:
    def __init__(self, metadata_path: str | Path) -> None:
        payload = load_yaml(metadata_path)
        self.token_map: dict[str, dict[str, str]] = payload.get("tokens", {}) if payload else {}

    def has_token(self, token: str) -> bool:
        return token in self.token_map

    def get_token_profile(self, token: str) -> dict[str, Any]:
        attrs = self.token_map.get(token)
        if not attrs:
            return {
                "token_class": "other_symbolic_token",
                "attribute_mode": "disabled",
                "match_domain": "none",
                "attribute_probe_enabled": False,
                "reason": "No symbolic attribute mapping is available for this token.",
            }
        token_class = attrs.get("token_class", "other_symbolic_token")
        return {
            "token_class": token_class,
            "attribute_mode": "symbolic",
            "match_domain": "symbolic",
            "attribute_probe_enabled": True,
            "reason": f"Non-lexical symbolic attributes are enabled for {token_class}.",
        }

    def build_predictions(self, token: str) -> dict[str, Any]:
        attrs = self.token_map.get(token, {})
        predictions: dict[str, Any] = {}
        for key, value in attrs.items():
            if key == "token_class" or not value:
                continue
            predictions[key] = {
                "top_prediction": {"label": value, "score": 1.0},
                "top_candidates": [{"label": value, "score": 1.0}],
                "support": 1,
                "source": "symbolic_registry",
            }
        return predictions
