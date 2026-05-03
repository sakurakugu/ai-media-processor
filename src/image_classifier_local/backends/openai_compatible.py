from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path

import requests

from .base import BaseClassifierBackend
from ..models import ALLOWED_LABELS, BackendConfig, ClassificationResult


SYSTEM_PROMPT = """You are an image classification engine.
Classify the image into exactly one label from this closed set:
- screenshot_text
- cosplay
- anime_art
- meme
- other

Definitions:
- screenshot_text: screenshot, UI capture, document capture, chat screenshot, or text-heavy image.
- cosplay: a real human dressed as a fictional character.
- anime_art: anime image, manga-style illustration, non-photorealistic two-dimensional artwork.
- meme: meme image, reaction image, internet joke image, or text-caption humor image.
- other: anything unclear or not fitting the categories above.

Return JSON only in this exact schema:
{"label":"other","confidence":0.0,"reason":""}

Rules:
- label must be one of the five labels above.
- confidence must be between 0 and 1.
- if uncertain, use other.
- do not return markdown.
"""


class OpenAICompatibleBackend(BaseClassifierBackend):
    def __init__(self, config: BackendConfig):
        self.config = config

    def classify(self, image_path: Path) -> ClassificationResult:
        payload = self._build_payload(image_path)
        response = requests.post(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = self._parse_content(content)
        label = parsed.get("label", "other")
        if label not in ALLOWED_LABELS:
            label = "other"
        confidence = parsed.get("confidence", 0.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))
        reason = str(parsed.get("reason", ""))
        return ClassificationResult(
            image_path=image_path,
            label=label,
            confidence=confidence,
            reason=reason,
            raw_response=content,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.config.extra_headers}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _build_payload(self, image_path: Path) -> dict:
        image_url = self._to_data_url(image_path)
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Classify this image into one label and return JSON only.",
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def _to_data_url(self, image_path: Path) -> str:
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _parse_content(self, content: str) -> dict:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"label": "other", "confidence": 0.0, "reason": cleaned}
