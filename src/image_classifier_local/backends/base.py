from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import ClassificationResult


class BaseClassifierBackend(ABC):
    @abstractmethod
    def classify(self, image_path: Path) -> ClassificationResult:
        raise NotImplementedError

    def test_connection(self) -> str:
        return "当前后端不需要额外连接测试。"
