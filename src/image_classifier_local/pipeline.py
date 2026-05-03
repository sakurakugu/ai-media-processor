from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Callable, Iterable

from .backends.base import BaseClassifierBackend
from .models import ClassificationResult, label_to_display_name, label_to_folder_name


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


class ClassificationCancelled(Exception):
    pass


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
    should_stop: Callable[[], bool] | None = None,
) -> list[ClassificationResult]:
    image_list = list(image_paths)
    total = len(image_list)
    results: list[ClassificationResult] = []
    for index, image_path in enumerate(image_list, start=1):
        if should_stop is not None and should_stop():
            raise ClassificationCancelled(f"分类已停止，已完成 {len(results)}/{total} 张图片。")
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


def move_results_to_label_folders(
    results: list[ClassificationResult],
    output_dir: Path,
) -> list[ClassificationResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    moved_results: list[ClassificationResult] = []
    for result in results:
        source_path = result.image_path
        target_dir = output_dir / label_to_folder_name(result.label)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name
        if source_path.resolve() != target_path.resolve():
            target_path = _dedupe_target_path(target_path, source_path)
            shutil.move(str(source_path), str(target_path))
        moved_results.append(
            ClassificationResult(
                image_path=target_path,
                label=result.label,
                confidence=result.confidence,
                reason=result.reason,
                raw_response=result.raw_response,
            )
        )
    return moved_results


def _dedupe_target_path(target_path: Path, source_path: Path) -> Path:
    if not target_path.exists():
        return target_path
    try:
        if target_path.resolve() == source_path.resolve():
            return target_path
    except FileNotFoundError:
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    index = 1
    while True:
        candidate = target_path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1
