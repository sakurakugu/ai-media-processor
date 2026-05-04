# Local Image Classifier

一个本地图片分类小工具的最小骨架，当前目标是先把流程跑通：

- 选择单张图片或一个目录
- 按固定类别分类
- 在桌面 GUI 里查看结果
- 后续再接 OCR、细分类、梗图拆分等能力

当前固定类别：

- `screenshot_text`
- `cosplay`
- `anime_art`
- `meme`
- `other`

当前界面与导出同时提供：

- 英文原始标签
- 中文标签说明

当前支持输入格式：

- `JPG / JPEG`
- `PNG`
- `WEBP`
- `BMP`
- `GIF`
- `HEIC / HEIF`
- 对扩展名错误但文件头可识别的图片，也会自动纳入处理

当前提供三种后端：

- `mock`：无模型依赖，便于先验证 GUI 和批处理流程
- `ollama`：直接连接本地 Ollama，适合当前这台 Windows 机器快速落地
- `openai_compatible`：连接本地 OpenAI 兼容视觉服务，适合后续接 `Qwen3.5-4B` 或其他视觉模型

## 推荐路线

机器是 `16GB VRAM`，第一版建议：

- 先用 `Qwen3.5-4B`
- 先做固定 5 分类
- 先跑批处理与人工复核
- 后续再加 OCR 和更细分类

## 目录结构

```text
image-classifier-local/
├─ app.py
├─ main.py
├─ requirements.txt
├─ README.md
├─ docs/
│  └─ architecture.md
└─ src/
   └─ image_classifier_local/
      ├─ gui.py
      ├─ models.py
      ├─ pipeline.py
      └─ backends/
         ├─ base.py
         ├─ mock.py
         ├─ ollama.py
         └─ openai_compatible.py
```

## 快速启动

### 1. 创建环境

```powershell
cd D:\elric\Code\image-classifier-local
# python -m venv .venv
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```powershell
# 拉取模型
ollama pull qwen3.5:4b
```

```powershell
# 启动模型并确认本地服务正常
ollama run qwen3.5:4b
# ollama stop qwen3.5:4b # 记得关闭
```

### 2. 运行 GUI

```powershell
python app.py
```

默认后端是 `mock`，不需要模型即可运行。

如果你已经安装了 Ollama，GUI 顶部现在可以直接：

- 选择 `本地 Ollama`
- 点击 `连接本地 Ollama`
- 点击 `启动 Ollama`，会先检查 `11434` 端口服务是否可用，未启动时自动执行 `ollama serve`
- 点击 `开启模型` / `关闭模型`，可把当前填写的模型载入或卸载出内存
- 点击 `测试连接`
- 按需勾选 `递归子目录`
- 也可以直接拖拽图片或文件夹到窗口里导入

也可以用统一入口启动：

```powershell
python main.py gui
```

## CLI 用法

### 最小示例

```powershell
python main.py cli D:\images
```

如果只想扫描当前目录、不递归子目录：

```powershell
python main.py cli D:\images --no-recursive
```

### 导出 CSV

```powershell
python main.py cli D:\images --csv D:\images\result.csv
```

### 导出 JSON

```powershell
python main.py cli D:\images --json D:\images\result.json
```

### 接本地视觉模型服务

```powershell
python main.py cli D:\images --backend ollama --model qwen3.5:4b --csv D:\images\result.csv
```

CLI 输出格式：

- 图片路径
- 中文标签 + 英文标签
- 置信度
- 原因

## 接入本地视觉模型

这个项目不直接绑定某一个推理框架，当前既支持直接连接 Ollama，也支持通过 OpenAI 兼容接口接其他后端，目的是先把工具层稳定下来。

你后续可以接：

- 本地 `Ollama`
- 本地 `vLLM` 服务
- 本地 `SGLang` 服务
- 其他支持 OpenAI 兼容 `/chat/completions` 的服务

GUI 里要填的字段：

- `Base URL`：Ollama 通常是 `http://127.0.0.1:11434`，OpenAI 兼容服务例如 `http://127.0.0.1:8000/v1`
- `Model`：Ollama 例如 `qwen3.5:4b`，OpenAI 兼容服务例如 `Qwen/Qwen3.5-4B`
- `API Key`：Ollama 可留空，某些 OpenAI 兼容服务要求填任意字符串

本地服务接入文档见：`docs/qwen35_local_setup.md:1`

## 当前适合的使用方式

- 先在 `mock` 模式下确认 GUI、目录扫描、CSV 导出都正常
- 再切到本地模型服务
- 先人工抽样 100 张图检查分类效果
- 最后决定是否需要二阶段分类或 OCR

## 输出说明

每条结果包含：

- 图片路径
- 分类标签
- 中文标签
- 置信度
- 分类原因
- 原始响应

支持 GUI 和 CLI 两种方式，并可导出为 `CSV` / `JSON`。
