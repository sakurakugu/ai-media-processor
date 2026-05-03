from __future__ import annotations

import base64
import json
from pathlib import Path

import requests

from .base import BaseClassifierBackend
from ..models import ALLOWED_LABELS, BackendConfig, ClassificationResult


SYSTEM_PROMPT = """你是一个图像分类引擎。
请从下面的封闭标签集合中，且只能选一个标签：
- screenshot_text
- cosplay
- anime_art
- meme
- other

标签定义：
- screenshot_text：截图、界面截屏、文档截图、聊天记录截图，或以文字内容为主的图片。
- cosplay：真人扮演虚构角色的照片。
- anime_art：动漫图片、漫画风插画、非写实的二维绘画作品。
- meme：表情包、梗图、反应图，或带文字梗的幽默图片。
- other：无法明确判断，或不属于以上任一类别。

只返回 JSON，且必须严格符合下面这个结构：
{"label":"other","confidence":0.0,"reason":""}

规则：
- label 必须是上面五个英文标签之一，不能翻译成中文。
- confidence 必须是 0 到 1 之间的数字。
- reason 必须使用简体中文，简洁说明判断依据。
- 如果不确定，使用 other。
- 不要返回 markdown，不要返回额外说明文字。
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
                    "content": "请对这张图片进行分类，只返回 JSON。label 保持英文枚举值，reason 使用简体中文。",
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
