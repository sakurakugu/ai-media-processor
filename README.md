# AI Media Processor

一个本地 AI 图片/视频处理工具：

- 当前只能分类图片和视频

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
- `AVIF`
- `MP4 / MOV / MKV / AVI / WEBM / M4V`
- 对扩展名错误但文件头可识别的图片，也会自动纳入处理
- 视频会先抽帧，再按帧分类并投票得到最终标签

当前提供三种后端：

- `mock`：无模型依赖，便于先验证 GUI 和批处理流程
- `ollama`：直接连接本地 Ollama，适合当前这台 Windows 机器快速落地
- `openai_compatible`：连接本地 OpenAI 兼容视觉服务，适合后续接 `Qwen3.5-4B` 或其他视觉模型

## 快速开始

```bash
# 先进入目录
pip install -r requirements.txt
# 没安装 ollama 的输入: irm https://ollama.com/install.ps1 | iex
ollama pull qwen3.5:4b
python main.py
```

## 目录结构

```text
ai-media-processor/
├─ app.py
├─ main.py
├─ requirements.txt
├─ README.md
├─ docs/
│  └─ architecture.md
└─ src/
   ├─ gui.py
   ├─ models.py
   ├─ pipeline.py
   └─ backends/
      ├─ base.py
      ├─ mock.py
      ├─ ollama.py
      └─ openai_compatible.py
```

## GUI 用法

### 1. 创建环境

```powershell
cd D:\ai-media-processor
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
python main.py
```

默认后端是 `mock`，不需要模型即可运行。

如果你已经安装了 Ollama，GUI 顶部现在可以直接：

- 选择 `本地 Ollama`
- 点击 `连接本地 Ollama`
- 点击 `启动 Ollama`，会先检查 `11434` 端口服务是否可用，未启动时自动执行 `ollama serve`
- 点击 `开启模型` / `关闭模型`，可把当前填写的模型载入或卸载出内存
- 点击 `测试连接`
- 按需勾选 `递归子目录`
- 也可以直接拖拽图片、视频或文件夹到窗口里导入
- 可设置“视频抽帧数”控制每个视频抽取多少帧参与分类

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

### 处理图片和视频混合目录

```powershell
python main.py cli D:\media --backend ollama --model qwen3.5:4b --video-frame-count 5 --json D:\media\result.json
```

CLI 输出格式：

- 文件路径
- 中文标签 + 英文标签
- 置信度
- 原因

## 接入本地视觉模型

这个项目不直接绑定某一个推理框架，当前既支持直接连接 Ollama，也支持通过 OpenAI 兼容接口接其他后端。

## 输出说明

每条结果包含：

- 图片路径
- 分类标签
- 中文标签
- 置信度
- 分类原因
- 原始响应

支持 GUI 和 CLI 两种方式，并可导出为 `CSV` / `JSON`。
