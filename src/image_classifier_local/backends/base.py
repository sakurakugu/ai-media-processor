from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import ClassificationResult


class BaseClassifierBackend(ABC):
    @abstractmethod
    def classify(self, image_path: Path) -> ClassificationResult:
        raise NotImplementedError
