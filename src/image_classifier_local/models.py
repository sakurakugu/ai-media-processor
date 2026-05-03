from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


ALLOWED_LABELS = [
    "screenshot_text",
    "cosplay",
    "anime_art",
    "meme",
    "other",
]


LABEL_DISPLAY_NAMES = {
    "screenshot_text": "截图/文字图",
    "cosplay": "Cosplay",
    "anime_art": "二次元美图",
    "meme": "表情包/梗图",
    "other": "其他",
}


def label_to_display_name(label: str) -> str:
    return LABEL_DISPLAY_NAMES.get(label, label)


@dataclass(slots=True)
class ClassificationResult:
    image_path: Path
    label: str
    confidence: float
    reason: str = ""
    raw_response: str = ""


@dataclass(slots=True)
class BackendConfig:
    backend_name: str
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    timeout_seconds: int = 120
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ClassifierRequest:
    image_path: Path
    labels: list[str] = field(default_factory=lambda: ALLOWED_LABELS.copy())
    prompt_hint: Optional[str] = None
