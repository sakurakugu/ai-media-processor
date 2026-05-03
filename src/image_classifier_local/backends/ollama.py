from __future__ import annotations

import base64
import json
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


class OllamaBackend(BaseClassifierBackend):
    def __init__(self, config: BackendConfig):
        self.config = config

    def classify(self, image_path: Path) -> ClassificationResult:
        payload = self._build_payload(image_path)
        response = requests.post(
            f"{self._root_url()}/api/chat",
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        content = self._extract_content(data)
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

    def test_connection(self) -> str:
        response = requests.get(
            f"{self._root_url()}/api/tags",
            timeout=min(self.config.timeout_seconds, 15),
        )
        response.raise_for_status()
        data = response.json()
        models = [item.get("name", "") for item in data.get("models", []) if item.get("name")]
        if self.config.model and models and self.config.model not in models:
            return (
                f"Ollama 服务可访问，但未找到模型 {self.config.model}。"
                f" 当前已下载：{', '.join(models[:8])}"
            )
        return f"Ollama 连接成功。服务地址：{self._root_url()}"

    def _build_payload(self, image_path: Path) -> dict:
        encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Classify this image into one label and return JSON only.",
                    "images": [encoded_image],
                },
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0},
        }

    def _root_url(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        if base_url.endswith("/v1"):
            return base_url[:-3]
        return base_url

    def _extract_content(self, payload: dict) -> str:
        message = payload.get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            fragments = [item.get("text", "") for item in content if isinstance(item, dict)]
            return "\n".join(fragment for fragment in fragments if fragment)
        return str(content)

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
