from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Iterable

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
    on_result: Callable[[ClassificationResult, int, int], None] | None = None,
) -> list[ClassificationResult]:
    image_list = list(image_paths)
    total = len(image_list)
    results: list[ClassificationResult] = []
    for index, image_path in enumerate(image_list, start=1):
        result = backend.classify(image_path)
        results.append(result)
        if on_result is not None:
            on_result(result, index, total)
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
