from __future__ import annotations

import hashlib
from pathlib import Path

from .base import BaseClassifierBackend
from ..models import ALLOWED_LABELS, ClassificationResult


class MockClassifierBackend(BaseClassifierBackend):
    def classify(self, image_path: Path) -> ClassificationResult:
        digest = hashlib.md5(str(image_path).encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % len(ALLOWED_LABELS)
        label = ALLOWED_LABELS[index]
        confidence = 0.45 + (int(digest[8:12], 16) % 40) / 100
        return ClassificationResult(
            image_path=image_path,
            label=label,
            confidence=min(confidence, 0.99),
            reason="mock backend result for GUI and batch flow validation",
            raw_response=digest,
        )
