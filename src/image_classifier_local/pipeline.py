from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .backends.base import BaseClassifierBackend
from .models import ClassificationResult, label_to_display_name


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def discover_images(paths: Iterable[Path]) -> list[Path]:
    discovered: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            discovered.append(path)
            continue
        if path.is_dir():
            for candidate in path.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                    discovered.append(candidate)
    return sorted(set(discovered))


def classify_images(
    backend: BaseClassifierBackend,
    image_paths: Iterable[Path],
) -> list[ClassificationResult]:
    results: list[ClassificationResult] = []
    for image_path in image_paths:
        results.append(backend.classify(image_path))
    return results


def export_results_csv(results: list[ClassificationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "label", "label_zh", "confidence", "reason", "raw_response"])
        for result in results:
            writer.writerow(
                [
                    str(result.image_path),
                    result.label,
                    label_to_display_name(result.label),
                    f"{result.confidence:.4f}",
                    result.reason,
                    result.raw_response,
                ]
            )


def export_results_json(results: list[ClassificationResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "image_path": str(result.image_path),
            "label": result.label,
            "label_zh": label_to_display_name(result.label),
            "confidence": round(result.confidence, 4),
            "reason": result.reason,
            "raw_response": result.raw_response,
        }
        for result in results
    ]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
