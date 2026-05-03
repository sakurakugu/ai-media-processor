from __future__ import annotations

import argparse
from pathlib import Path

from .backends.mock import MockClassifierBackend
from .backends.ollama import OllamaBackend
from .backends.openai_compatible import OpenAICompatibleBackend
from .gui import launch
from .models import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    BackendConfig,
    label_to_display_name,
)
from .pipeline import classify_images, discover_images, export_results_csv, export_results_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-classifier-local",
        description="本地图像分类工具，支持 GUI 和 CLI 两种模式。",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("gui", help="启动图形界面")

    cli_parser = subparsers.add_parser("cli", help="命令行批量分类")
    cli_parser.add_argument("inputs", nargs="+", help="图片文件或目录路径")
    cli_parser.add_argument(
        "--backend",
        choices=["mock", "ollama", "openai_compatible"],
        default="mock",
        help="后端类型，默认 mock",
    )
    cli_parser.add_argument(
        "--base-url",
        default="",
        help="后端服务地址；ollama 默认使用 http://127.0.0.1:11434",
    )
    cli_parser.add_argument("--model", default="", help="模型名称")
    cli_parser.add_argument("--api-key", default="", help="接口密钥")
    cli_parser.add_argument("--csv", default="", help="导出 CSV 路径")
    cli_parser.add_argument("--json", default="", help="导出 JSON 路径")
    cli_parser.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="未发现图片时返回非零退出码",
    )

    return parser


def create_backend(args) -> MockClassifierBackend | OllamaBackend | OpenAICompatibleBackend:
    if args.backend == "mock":
        return MockClassifierBackend()
    default_base_url = DEFAULT_OPENAI_BASE_URL
    default_model = DEFAULT_OPENAI_MODEL
    if args.backend == "ollama":
        default_base_url = DEFAULT_OLLAMA_BASE_URL
        default_model = DEFAULT_OLLAMA_MODEL
    config = BackendConfig(
        backend_name=args.backend,
        model=args.model.strip() or default_model,
        base_url=args.base_url.strip() or default_base_url,
        api_key=args.api_key.strip(),
    )
    if not config.base_url or not config.model:
        raise ValueError("使用远程模型后端时，--base-url 和 --model 不能为空。")
    if args.backend == "ollama":
        return OllamaBackend(config)
    return OpenAICompatibleBackend(config)


def run_cli(args) -> int:
    inputs = [Path(item) for item in args.inputs]
    image_paths = discover_images(inputs)
    if not image_paths:
        print("未发现可处理的图片。")
        return 2 if args.fail_on_empty else 0

    backend = create_backend(args)
    results = classify_images(backend, image_paths)

    for result in results:
        print(
            f"{result.image_path}\t{label_to_display_name(result.label)} ({result.label})\t{result.confidence:.2f}\t{result.reason}"
        )

    if args.csv:
        output_path = Path(args.csv)
        export_results_csv(results, output_path)
        print(f"\n已导出 CSV：{output_path}")

    if args.json:
        output_path = Path(args.json)
        export_results_json(results, output_path)
        print(f"\n已导出 JSON：{output_path}")

    print(f"\n处理完成，共 {len(results)} 张图片。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "gui"):
        launch()
        return 0
    if args.command == "cli":
        return run_cli(args)

    parser.print_help()
    return 1
