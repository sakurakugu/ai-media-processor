from __future__ import annotations

import queue
import threading
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, VERTICAL, W, filedialog, messagebox, ttk
import tkinter as tk

from PIL import Image, ImageTk

from .backends.mock import MockClassifierBackend
from .backends.openai_compatible import OpenAICompatibleBackend
from .models import BackendConfig, ClassificationResult, label_to_display_name
from .pipeline import classify_images, discover_images, export_results_csv, export_results_json


BACKEND_OPTIONS = {
    "模拟后端（无模型）": "mock",
    "OpenAI兼容接口": "openai_compatible",
}


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("本地图像分类工具")
        self.root.geometry("1200x760")

        self.items: list[Path] = []
        self.results: list[ClassificationResult] = []
        self.preview_image = None
        self.status_var = tk.StringVar(value="就绪")
        self.backend_var = tk.StringVar(value="模拟后端（无模型）")
        self.base_url_var = tk.StringVar(value="http://127.0.0.1:8000/v1")
        self.model_var = tk.StringVar(value="Qwen3.5-4B")
        self.api_key_var = tk.StringVar(value="")
        self.result_queue: queue.Queue = queue.Queue()

        self._build_layout()
        self.root.after(150, self._poll_queue)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=BOTH, expand=True)

        top = ttk.LabelFrame(container, text="后端配置", padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="后端类型").grid(row=0, column=0, sticky=W, padx=4, pady=4)
        ttk.Combobox(
            top,
            textvariable=self.backend_var,
            values=list(BACKEND_OPTIONS.keys()),
            state="readonly",
            width=20,
        ).grid(row=0, column=1, sticky=W, padx=4, pady=4)

        ttk.Label(top, text="服务地址").grid(row=0, column=2, sticky=W, padx=4, pady=4)
        ttk.Entry(top, textvariable=self.base_url_var, width=36).grid(
            row=0, column=3, sticky=W, padx=4, pady=4
        )

        ttk.Label(top, text="模型名").grid(row=0, column=4, sticky=W, padx=4, pady=4)
        ttk.Entry(top, textvariable=self.model_var, width=22).grid(
            row=0, column=5, sticky=W, padx=4, pady=4
        )

        ttk.Label(top, text="API Key").grid(row=0, column=6, sticky=W, padx=4, pady=4)
        ttk.Entry(top, textvariable=self.api_key_var, width=22, show="*").grid(
            row=0, column=7, sticky=W, padx=4, pady=4
        )

        action_bar = ttk.Frame(container, padding=(0, 10))
        action_bar.pack(fill="x")

        ttk.Button(action_bar, text="添加文件", command=self.add_files).pack(side=LEFT, padx=4)
        ttk.Button(action_bar, text="添加文件夹", command=self.add_folder).pack(side=LEFT, padx=4)
        ttk.Button(action_bar, text="清空", command=self.clear_items).pack(side=LEFT, padx=4)
        ttk.Button(action_bar, text="分类选中项", command=self.classify_selected).pack(side=LEFT, padx=4)
        ttk.Button(action_bar, text="全部分类", command=self.classify_all).pack(side=LEFT, padx=4)
        ttk.Button(action_bar, text="导出 CSV", command=self.export_csv).pack(side=LEFT, padx=4)
        ttk.Button(action_bar, text="导出 JSON", command=self.export_json).pack(side=LEFT, padx=4)

        body = ttk.Panedwindow(container, orient="horizontal")
        body.pack(fill=BOTH, expand=True)

        left = ttk.Frame(body, padding=(0, 0, 10, 0))
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        self.file_list = tk.Listbox(left, selectmode=tk.EXTENDED)
        self.file_list.pack(fill=BOTH, expand=True)
        self.file_list.bind("<<ListboxSelect>>", self._on_select_item)

        result_frame = ttk.LabelFrame(right, text="分类结果", padding=8)
        result_frame.pack(fill=BOTH, expand=True)

        columns = ("path", "label", "confidence")
        self.result_tree = ttk.Treeview(result_frame, columns=columns, show="headings", height=12)
        self.result_tree.heading("path", text="图片")
        self.result_tree.heading("label", text="标签")
        self.result_tree.heading("confidence", text="置信度")
        self.result_tree.column("path", width=360, anchor=W)
        self.result_tree.column("label", width=110, anchor=W)
        self.result_tree.column("confidence", width=90, anchor=W)
        self.result_tree.pack(fill=BOTH, expand=True)
        self.result_tree.bind("<<TreeviewSelect>>", self._on_select_result)

        detail_frame = ttk.LabelFrame(right, text="预览 / 详情", padding=8)
        detail_frame.pack(fill=BOTH, expand=True, pady=(10, 0))

        self.preview_label = ttk.Label(detail_frame, text="未选择图片")
        self.preview_label.pack(fill="x")

        self.reason_text = tk.Text(detail_frame, height=10, wrap="word")
        self.reason_text.pack(fill=BOTH, expand=True, pady=(8, 0))

        status = ttk.Label(container, textvariable=self.status_var, anchor=W)
        status.pack(fill="x", pady=(10, 0))

    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="选择图片",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.gif")],
        )
        self._add_paths([Path(path) for path in paths])

    def add_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择文件夹")
        if folder:
            self._add_paths(discover_images([Path(folder)]))

    def clear_items(self) -> None:
        self.items.clear()
        self.results.clear()
        self.file_list.delete(0, END)
        for row in self.result_tree.get_children():
            self.result_tree.delete(row)
        self.reason_text.delete("1.0", END)
        self.preview_label.configure(image="", text="未选择图片")
        self.status_var.set("已清空")

    def classify_selected(self) -> None:
        selected = self.file_list.curselection()
        if not selected:
            messagebox.showinfo("提示", "请先选择至少一张图片。")
            return
        paths = [self.items[index] for index in selected]
        self._run_classification(paths)

    def classify_all(self) -> None:
        if not self.items:
            messagebox.showinfo("提示", "请先添加图片。")
            return
        self._run_classification(self.items)

    def export_csv(self) -> None:
        if not self.results:
            messagebox.showinfo("提示", "当前没有可导出的结果。")
            return
        output = filedialog.asksaveasfilename(
            title="导出 CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not output:
            return
        export_results_csv(self.results, Path(output))
        self.status_var.set(f"已导出 CSV：{output}")

    def export_json(self) -> None:
        if not self.results:
            messagebox.showinfo("提示", "当前没有可导出的结果。")
            return
        output = filedialog.asksaveasfilename(
            title="导出 JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not output:
            return
        export_results_json(self.results, Path(output))
        self.status_var.set(f"已导出 JSON：{output}")

    def _run_classification(self, image_paths: list[Path]) -> None:
        backend = self._create_backend()
        self.status_var.set(f"正在分类，共 {len(image_paths)} 张图片…")
        thread = threading.Thread(
            target=self._classify_worker,
            args=(backend, image_paths),
            daemon=True,
        )
        thread.start()

    def _classify_worker(self, backend, image_paths: list[Path]) -> None:
        try:
            results = classify_images(backend, image_paths)
            self.result_queue.put(("results", results))
        except Exception as exc:
            self.result_queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                message_type, payload = self.result_queue.get_nowait()
                if message_type == "results":
                    self._merge_results(payload)
                elif message_type == "error":
                    self.status_var.set(f"错误：{payload}")
                    messagebox.showerror("分类失败", payload)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)

    def _merge_results(self, new_results: list[ClassificationResult]) -> None:
        index_by_path = {result.image_path: idx for idx, result in enumerate(self.results)}
        for result in new_results:
            if result.image_path in index_by_path:
                self.results[index_by_path[result.image_path]] = result
            else:
                self.results.append(result)
        self._refresh_result_tree()
        self.status_var.set(f"分类完成，本次处理 {len(new_results)} 张图片。")

    def _refresh_result_tree(self) -> None:
        for row in self.result_tree.get_children():
            self.result_tree.delete(row)
        for result in self.results:
            self.result_tree.insert(
                "",
                END,
                values=(
                    str(result.image_path),
                    f"{label_to_display_name(result.label)} ({result.label})",
                    f"{result.confidence:.2f}",
                ),
            )

    def _create_backend(self):
        backend_name = BACKEND_OPTIONS.get(self.backend_var.get(), "mock")
        if backend_name == "mock":
            return MockClassifierBackend()
        config = BackendConfig(
            backend_name=backend_name,
            model=self.model_var.get().strip(),
            base_url=self.base_url_var.get().strip(),
            api_key=self.api_key_var.get().strip(),
        )
        if not config.base_url or not config.model:
            raise ValueError("使用 OpenAI 兼容接口时，服务地址和模型名不能为空。")
        return OpenAICompatibleBackend(config)

    def _add_paths(self, paths: list[Path]) -> None:
        merged = sorted(set(self.items).union(paths))
        self.items = merged
        self.file_list.delete(0, END)
        for path in self.items:
            self.file_list.insert(END, str(path))
        self.status_var.set(f"已加载 {len(self.items)} 张图片。")

    def _on_select_item(self, _event=None) -> None:
        selected = self.file_list.curselection()
        if not selected:
            return
        self._show_preview(self.items[selected[0]])

    def _on_select_result(self, _event=None) -> None:
        selection = self.result_tree.selection()
        if not selection:
            return
        values = self.result_tree.item(selection[0], "values")
        image_path = Path(values[0])
        result = next((item for item in self.results if item.image_path == image_path), None)
        self._show_preview(image_path)
        self.reason_text.delete("1.0", END)
        if result:
            self.reason_text.insert(
                "1.0",
                f"标签：{label_to_display_name(result.label)} ({result.label})\n置信度：{result.confidence:.2f}\n\n原因：\n{result.reason}\n\n原始响应：\n{result.raw_response}",
            )

    def _show_preview(self, image_path: Path) -> None:
        try:
            image = Image.open(image_path)
            image.thumbnail((420, 320))
            self.preview_image = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview_image, text="")
        except Exception:
            self.preview_label.configure(image="", text=str(image_path))


def launch() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()
