from __future__ import annotations

import queue
import threading
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, VERTICAL, W, filedialog, messagebox, ttk
import tkinter as tk

from PIL import Image, ImageTk

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    DND_FILES = None
    TkinterDnD = None

from .backends.mock import MockClassifierBackend
from .backends.ollama import OllamaBackend
from .backends.openai_compatible import OpenAICompatibleBackend
from .models import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    BackendConfig,
    ClassificationResult,
    SkippedImage,
    label_to_display_name,
)
from .pipeline import (
    ClassificationCancelled,
    classify_images,
    discover_images,
    export_results_csv_with_skips,
    export_results_json_with_skips,
    move_results_to_label_folders,
    move_skipped_items_to_folder,
)


BACKEND_OPTIONS = {
    "模拟后端（无模型）": "mock",
    "本地 Ollama": "ollama",
    "OpenAI兼容接口": "openai_compatible",
}


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("本地图像分类工具")
        self.root.geometry("1200x760")

        self.items: list[Path] = []
        self.results: list[ClassificationResult] = []
        self.skipped_items: list[SkippedImage] = []
        self.result_tree_rows: dict[Path, str] = {}
        self.preview_image = None
        self.is_classifying = False
        self.stop_requested = False
        self.skipped_count = 0
        self.cancel_event = threading.Event()
        self.status_var = tk.StringVar(value="就绪")
        self.backend_var = tk.StringVar(value="本地 Ollama")
        self.base_url_var = tk.StringVar(value=DEFAULT_OLLAMA_BASE_URL)
        self.model_var = tk.StringVar(value=DEFAULT_OLLAMA_MODEL)
        self.api_key_var = tk.StringVar(value="")
        self.recursive_scan_var = tk.BooleanVar(value=True)
        self.result_queue: queue.Queue = queue.Queue()

        self._build_layout()
        self.root.after(150, self._poll_queue)

    def _build_layout(self) -> None:
        self._enable_drop_target(self.root)
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

        button_bar = ttk.Frame(top)
        button_bar.grid(row=1, column=0, columnspan=8, sticky=W, padx=4, pady=(6, 0))
        ttk.Button(button_bar, text="连接本地 Ollama", command=self.connect_local_ollama).pack(
            side=LEFT, padx=(0, 8)
        )
        ttk.Button(button_bar, text="测试连接", command=self.test_connection).pack(side=LEFT)

        action_bar = ttk.Frame(container, padding=(0, 10))
        action_bar.pack(fill="x")

        self.add_files_button = ttk.Button(action_bar, text="添加文件", command=self.add_files)
        self.add_files_button.pack(side=LEFT, padx=4)
        self.add_folder_button = ttk.Button(action_bar, text="添加文件夹", command=self.add_folder)
        self.add_folder_button.pack(side=LEFT, padx=4)
        ttk.Checkbutton(
            action_bar,
            text="递归子目录",
            variable=self.recursive_scan_var,
        ).pack(side=LEFT, padx=(4, 12))
        self.clear_button = ttk.Button(action_bar, text="清空", command=self.clear_items)
        self.clear_button.pack(side=LEFT, padx=4)
        self.classify_selected_button = ttk.Button(
            action_bar,
            text="分类选中项",
            command=self.classify_selected,
        )
        self.classify_selected_button.pack(side=LEFT, padx=4)
        self.classify_all_button = ttk.Button(action_bar, text="全部分类", command=self.classify_all)
        self.classify_all_button.pack(side=LEFT, padx=4)
        self.stop_button = ttk.Button(action_bar, text="停止分类", command=self.stop_classification)
        self.stop_button.pack(side=LEFT, padx=4)
        self.stop_button.configure(state="disabled")
        self.move_button = ttk.Button(action_bar, text="按分类移动图片", command=self.move_classified_images)
        self.move_button.pack(side=LEFT, padx=4)
        ttk.Button(action_bar, text="导出 CSV", command=self.export_csv).pack(side=LEFT, padx=4)
        ttk.Button(action_bar, text="导出 JSON", command=self.export_json).pack(side=LEFT, padx=4)

        body = ttk.Panedwindow(container, orient="horizontal")
        body.pack(fill=BOTH, expand=True)

        left = ttk.Frame(body, padding=(0, 0, 10, 0))
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        file_list_frame = ttk.Frame(left)
        file_list_frame.pack(fill=BOTH, expand=True)

        file_list_scrollbar = ttk.Scrollbar(file_list_frame, orient=VERTICAL)
        file_list_scrollbar.pack(side=RIGHT, fill="y")

        self.file_list = tk.Listbox(
            file_list_frame,
            selectmode=tk.EXTENDED,
            yscrollcommand=file_list_scrollbar.set,
        )
        self.file_list.pack(side=LEFT, fill=BOTH, expand=True)
        file_list_scrollbar.configure(command=self.file_list.yview)
        self.file_list.bind("<<ListboxSelect>>", self._on_select_item)
        self._enable_drop_target(self.file_list)

        result_frame = ttk.LabelFrame(right, text="分类结果", padding=8)
        result_frame.pack(fill=BOTH, expand=True)

        columns = ("path", "label", "confidence")
        result_tree_frame = ttk.Frame(result_frame)
        result_tree_frame.pack(fill=BOTH, expand=True)

        result_tree_scrollbar = ttk.Scrollbar(result_tree_frame, orient=VERTICAL)
        result_tree_scrollbar.pack(side=RIGHT, fill="y")

        self.result_tree = ttk.Treeview(
            result_tree_frame,
            columns=columns,
            show="headings",
            height=12,
            yscrollcommand=result_tree_scrollbar.set,
        )
        self.result_tree.heading("path", text="图片")
        self.result_tree.heading("label", text="标签")
        self.result_tree.heading("confidence", text="置信度")
        self.result_tree.column("path", width=360, anchor=W)
        self.result_tree.column("label", width=110, anchor=W)
        self.result_tree.column("confidence", width=90, anchor=W)
        self.result_tree.pack(side=LEFT, fill=BOTH, expand=True)
        result_tree_scrollbar.configure(command=self.result_tree.yview)
        self.result_tree.bind("<<TreeviewSelect>>", self._on_select_result)

        detail_frame = ttk.LabelFrame(right, text="预览 / 详情", padding=8)
        detail_frame.pack(fill=BOTH, expand=True, pady=(10, 0))

        self.preview_label = ttk.Label(detail_frame, text="未选择图片")
        self.preview_label.pack(fill="x")

        reason_frame = ttk.Frame(detail_frame)
        reason_frame.pack(fill=BOTH, expand=True, pady=(8, 0))

        reason_scrollbar = ttk.Scrollbar(reason_frame, orient=VERTICAL)
        reason_scrollbar.pack(side=RIGHT, fill="y")

        self.reason_text = tk.Text(
            reason_frame,
            height=10,
            wrap="word",
            yscrollcommand=reason_scrollbar.set,
        )
        self.reason_text.pack(side=LEFT, fill=BOTH, expand=True)
        reason_scrollbar.configure(command=self.reason_text.yview)

        skipped_frame = ttk.LabelFrame(right, text="跳过文件", padding=8)
        skipped_frame.pack(fill=BOTH, expand=False, pady=(10, 0))

        skipped_text_frame = ttk.Frame(skipped_frame)
        skipped_text_frame.pack(fill=BOTH, expand=True)

        skipped_text_scrollbar = ttk.Scrollbar(skipped_text_frame, orient=VERTICAL)
        skipped_text_scrollbar.pack(side=RIGHT, fill="y")

        self.skipped_text = tk.Text(
            skipped_text_frame,
            height=7,
            wrap="word",
            yscrollcommand=skipped_text_scrollbar.set,
        )
        self.skipped_text.pack(side=LEFT, fill=BOTH, expand=True)
        skipped_text_scrollbar.configure(command=self.skipped_text.yview)
        self._enable_drop_target(self.skipped_text)

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
            self._add_paths(
                discover_images([Path(folder)], recursive=self.recursive_scan_var.get())
            )

    def clear_items(self) -> None:
        if self.is_classifying:
            messagebox.showinfo("提示", "分类进行中，请等待当前任务完成。")
            return
        self.items.clear()
        self.results.clear()
        self.skipped_items.clear()
        self.result_tree_rows.clear()
        self.file_list.delete(0, END)
        for row in self.result_tree.get_children():
            self.result_tree.delete(row)
        self.reason_text.delete("1.0", END)
        self.skipped_text.delete("1.0", END)
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
        export_results_csv_with_skips(self.results, self.skipped_items, Path(output))
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
        export_results_json_with_skips(self.results, self.skipped_items, Path(output))
        self.status_var.set(f"已导出 JSON：{output}")

    def move_classified_images(self) -> None:
        if not self.results and not self.skipped_items:
            messagebox.showinfo("提示", "当前没有可移动的分类结果或跳过文件。")
            return
        output_dir = filedialog.askdirectory(title="选择分类输出文件夹")
        if not output_dir:
            return

        old_results = list(self.results)
        old_skipped_items = list(self.skipped_items)
        try:
            new_results = move_results_to_label_folders(old_results, Path(output_dir))
            new_skipped_items = move_skipped_items_to_folder(old_skipped_items, Path(output_dir))
        except Exception as exc:
            self.status_var.set(f"移动失败：{exc}")
            messagebox.showerror("移动失败", str(exc))
            return

        path_remap = {
            old_result.image_path: new_result.image_path
            for old_result, new_result in zip(old_results, new_results)
        }
        path_remap.update(
            {
                old_item.image_path: new_item.image_path
                for old_item, new_item in zip(old_skipped_items, new_skipped_items)
            }
        )
        self.results = new_results
        self.skipped_items = new_skipped_items
        self.items = sorted(path_remap.get(path, path) for path in self.items)
        self._refresh_file_list()
        self._refresh_result_tree()
        self._refresh_skipped_text()
        self.reason_text.delete("1.0", END)
        self.preview_label.configure(image="", text="未选择图片")
        self.status_var.set(
            f"已移动 {len(new_results)} 张分类图片，{len(new_skipped_items)} 个跳过文件到：{output_dir}"
        )

    def connect_local_ollama(self) -> None:
        self.backend_var.set("本地 Ollama")
        self.base_url_var.set(DEFAULT_OLLAMA_BASE_URL)
        self.model_var.set(DEFAULT_OLLAMA_MODEL)
        self.api_key_var.set("")
        self.status_var.set("已填入本地 Ollama 默认配置。")

    def test_connection(self) -> None:
        try:
            backend = self._create_backend()
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            self.status_var.set(f"配置错误：{exc}")
            return

        self.status_var.set("正在测试后端连接…")
        thread = threading.Thread(target=self._test_connection_worker, args=(backend,), daemon=True)
        thread.start()

    def _run_classification(self, image_paths: list[Path]) -> None:
        if self.is_classifying:
            messagebox.showinfo("提示", "已有分类任务在进行中。")
            return
        try:
            backend = self._create_backend()
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            self.status_var.set(f"配置错误：{exc}")
            return
        self.is_classifying = True
        self.stop_requested = False
        self.skipped_count = 0
        self.skipped_items.clear()
        self.cancel_event.clear()
        self._set_classification_controls(enabled=False)
        image_paths = list(image_paths)
        self.skipped_text.delete("1.0", END)
        self.status_var.set(f"正在分类，共 {len(image_paths)} 张图片…")
        thread = threading.Thread(
            target=self._classify_worker,
            args=(backend, image_paths),
            daemon=True,
        )
        thread.start()

    def _classify_worker(self, backend, image_paths: list[Path]) -> None:
        try:
            results = classify_images(
                backend,
                image_paths,
                on_result=self._queue_progress_result,
                on_skip=self._queue_skipped_item,
                should_stop=self.cancel_event.is_set,
            )
            self.result_queue.put(("classification_done", len(results)))
        except ClassificationCancelled as exc:
            self.result_queue.put(("classification_cancelled", str(exc)))
        except Exception as exc:
            self.result_queue.put(("error", str(exc)))

    def stop_classification(self) -> None:
        if not self.is_classifying or self.stop_requested:
            return
        self.stop_requested = True
        self.cancel_event.set()
        self.stop_button.configure(state="disabled")
        self.status_var.set("正在停止，等待当前图片处理完成…")

    def _queue_progress_result(
        self,
        result: ClassificationResult,
        completed: int,
        total: int,
    ) -> None:
        self.result_queue.put(("progress_result", (result, completed, total)))

    def _queue_skipped_item(
        self,
        item: SkippedImage,
        completed: int,
        total: int,
    ) -> None:
        self.result_queue.put(("skipped_item", (item, completed, total)))

    def _test_connection_worker(self, backend) -> None:
        try:
            message = backend.test_connection()
            self.result_queue.put(("connection_ok", message))
        except Exception as exc:
            self.result_queue.put(("connection_error", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                message_type, payload = self.result_queue.get_nowait()
                if message_type == "progress_result":
                    result, completed, total = payload
                    self._merge_result(result)
                    if self.stop_requested:
                        self.status_var.set(
                            f"正在停止，当前已完成 {completed}/{total} 张：{result.image_path.name}"
                        )
                    else:
                        self.status_var.set(
                            f"正在分类，已完成 {completed}/{total} 张：{result.image_path.name}"
                        )
                elif message_type == "skipped_item":
                    item, completed, total = payload
                    self.skipped_count += 1
                    self.skipped_items.append(item)
                    self._append_skipped_item(item)
                    self.status_var.set(
                        f"已跳过 {completed}/{total} 张：{item.image_path.name}；{item.reason}"
                    )
                elif message_type == "classification_done":
                    self._finish_classification()
                    self.status_var.set(
                        f"分类完成，本次处理 {payload} 张图片，跳过 {self.skipped_count} 个文件。"
                    )
                elif message_type == "classification_cancelled":
                    self._finish_classification()
                    self.status_var.set(payload)
                    messagebox.showinfo("已停止", payload)
                elif message_type == "error":
                    self._finish_classification()
                    self.status_var.set(f"错误：{payload}")
                    messagebox.showerror("分类失败", payload)
                elif message_type == "connection_ok":
                    self.status_var.set(payload)
                    messagebox.showinfo("连接成功", payload)
                elif message_type == "connection_error":
                    self.status_var.set(f"连接失败：{payload}")
                    messagebox.showerror("连接失败", payload)
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)

    def _merge_result(self, new_result: ClassificationResult) -> None:
        index_by_path = {result.image_path: idx for idx, result in enumerate(self.results)}
        if new_result.image_path in index_by_path:
            self.results[index_by_path[new_result.image_path]] = new_result
        else:
            self.results.append(new_result)
        self._upsert_result_row(new_result)

    def _refresh_result_tree(self) -> None:
        self.result_tree_rows.clear()
        for row in self.result_tree.get_children():
            self.result_tree.delete(row)
        for result in self.results:
            item_id = self.result_tree.insert(
                "",
                END,
                values=self._result_row_values(result),
            )
            self.result_tree_rows[result.image_path] = item_id

    def _upsert_result_row(self, result: ClassificationResult) -> None:
        values = self._result_row_values(result)
        item_id = self.result_tree_rows.get(result.image_path)
        if item_id is None:
            item_id = self.result_tree.insert("", END, values=values)
            self.result_tree_rows[result.image_path] = item_id
            return
        self.result_tree.item(item_id, values=values)

    def _result_row_values(self, result: ClassificationResult) -> tuple[str, str, str]:
        return (
            str(result.image_path),
            f"{label_to_display_name(result.label)} ({result.label})",
            f"{result.confidence:.2f}",
        )

    def _set_classification_controls(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.add_files_button.configure(state=state)
        self.add_folder_button.configure(state=state)
        self.clear_button.configure(state=state)
        self.classify_selected_button.configure(state=state)
        self.classify_all_button.configure(state=state)
        self.move_button.configure(state=state)
        self.stop_button.configure(state="disabled" if enabled else "normal")

    def _finish_classification(self) -> None:
        self.is_classifying = False
        self.stop_requested = False
        self.cancel_event.clear()
        self._set_classification_controls(enabled=True)

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
            raise ValueError("使用远程模型后端时，服务地址和模型名不能为空。")
        if backend_name == "ollama":
            return OllamaBackend(config)
        return OpenAICompatibleBackend(config)

    def _add_paths(self, paths: list[Path]) -> None:
        merged = sorted(set(self.items).union(paths))
        self.items = merged
        self._refresh_file_list()
        self.status_var.set(f"已加载 {len(self.items)} 张图片。")

    def _refresh_file_list(self) -> None:
        self.file_list.delete(0, END)
        for path in self.items:
            self.file_list.insert(END, str(path))

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

    def _append_skipped_item(self, item: SkippedImage) -> None:
        self.skipped_text.insert("end", f"{item.image_path}\n{item.reason}\n\n")
        self.skipped_text.see("end")

    def _refresh_skipped_text(self) -> None:
        self.skipped_text.delete("1.0", END)
        for item in self.skipped_items:
            self._append_skipped_item(item)

    def _enable_drop_target(self, widget) -> None:
        if DND_FILES is None or not hasattr(widget, "drop_target_register"):
            return
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind("<<Drop>>", self._on_drop_files)

    def _on_drop_files(self, event) -> str:
        dropped_items = [Path(item) for item in self.root.tk.splitlist(event.data) if item]
        discovered = discover_images(dropped_items, recursive=self.recursive_scan_var.get())
        if discovered:
            self._add_paths(discovered)
        else:
            self.status_var.set("拖拽内容中未发现可处理的图片。")
        return "break"


def launch() -> None:
    if TkinterDnD is not None:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    App(root)
    root.mainloop()
