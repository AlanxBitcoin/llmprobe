from __future__ import annotations

# Design requirements (moved from PROJECT_DESIGN.md):
# - Define the minimal abstract contract for probe implementations.
# - Keep shared probe interface stable for study/pipeline callers.

from abc import ABC, abstractmethod
from typing import Any


class BaseProbe(ABC):
    """Base contract for probe implementations."""

    @abstractmethod
    def fit(self, features, labels) -> Any:
        """Train the probe and return self or model payload."""

    @abstractmethod
    def evaluate(self, features, labels) -> dict[str, Any]:
        """Evaluate trained probe and return metrics."""
