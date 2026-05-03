# Qwen3.5-4B 本地服务接入

这份文档对应当前项目的 `openai_compatible` 后端。

目标：

- 在本地启动一个兼容 OpenAI Chat Completions API 的视觉模型服务
- 让 GUI 和 CLI 直接连接该服务
- 优先保证“先跑起来”

## 结论

对你这台 `16GB VRAM` 机器，第一版建议：

- 模型：`Qwen/Qwen3.5-4B`
- 服务框架：优先 `vLLM`
- 运行环境：优先 `WSL2 Ubuntu`

说明：

- 这是工程建议，不是官方强制要求
- 官方模型卡明确给了 `vLLM`、`SGLang`、`transformers serve` 三种服务方式
- 但在 Windows 本地实际落地时，`WSL2 + vLLM` 通常更稳

## 官方信息

- `Qwen3.5-4B` 模型卡说明它兼容 `Transformers`、`vLLM`、`SGLang` 等框架
- 模型卡说明：`Qwen3.5` 默认会先输出 thinking 内容
- 模型卡提供了关闭 thinking 的 OpenAI 兼容调用方式
- 模型卡给出了 `vLLM`、`SGLang`、`transformers serve` 的启动命令

项目当前后端已经默认按分类场景关闭 thinking。

## 方案一：`vLLM`，推荐

### 适用

- 你要本地 OpenAI 兼容接口
- 你后续可能继续做批量分类
- 你接受先在 `WSL2` 里部署

### 官方安装命令

`Qwen3.5-4B` 模型卡给出的 `vLLM` 安装方式是：

```bash
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
```

### 官方启动命令

模型卡给出的标准启动示例是：

```bash
vllm serve Qwen/Qwen3.5-4B --port 8000 --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser qwen3
```

### 针对本项目的简化建议

下面这部分是工程建议，不是官方原样命令。

原因：

- 你当前只是做单图分类
- 不需要超长上下文
- `16GB VRAM` 没必要先背 `262144` 上下文长度

建议先从更保守的配置开始，例如：

```bash
vllm serve Qwen/Qwen3.5-4B --port 8000 --tensor-parallel-size 1 --max-model-len 8192 --reasoning-parser qwen3
```

如果显存和稳定性都没问题，再逐步试：

- `--max-model-len 16384`
- `--max-model-len 32768`

### 服务地址

启动后，本项目里填：

- `Base URL`：`http://127.0.0.1:8000/v1`
- `Model`：`Qwen/Qwen3.5-4B`
- `API Key`：可填 `EMPTY`

### 用 GUI 连接

- 后端类型：`OpenAI兼容接口`
- 服务地址：`http://127.0.0.1:8000/v1`
- 模型名：`Qwen/Qwen3.5-4B`
- API Key：`EMPTY`

### 用 CLI 连接

```powershell
python main.py cli D:\images --backend openai_compatible --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3.5-4B --api-key EMPTY --csv D:\images\result.csv --json D:\images\result.json
```

## 方案二：`transformers serve`，用于快速验证

### 适用

- 你只想快速验证接口
- 你暂时不追求更高吞吐

### 官方安装命令

模型卡给出的安装方式是：

```bash
pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
```

模型卡还提醒同时安装 `torchvision` 和 `pillow`。

### 官方启动命令

```bash
transformers serve --force-model Qwen/Qwen3.5-4B --port 8000 --continuous-batching
```

### 说明

- 这是更轻量的验证路线
- 对“先跑起来”很有帮助
- 但长期做批处理，我还是建议转回 `vLLM`

## 方案三：`SGLang`

模型卡同样提供了 `SGLang` 启动方式，适合更进阶的服务场景。

官方标准命令：

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B --port 8000 --tp-size 1 --mem-fraction-static 0.8 --context-length 262144 --reasoning-parser qwen3
```

如果你当前目标只是把分类工具先跑通，没必要优先走这条线。

## 为什么项目里默认关闭 thinking

这是为了分类场景做的工程取舍。

原因：

- 分类任务不需要长推理链
- 开着 thinking 会更慢
- 输出更容易带额外内容，增加 JSON 解析失败概率

项目当前发送到兼容接口的请求里，会附带：

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

## 排错建议

### 启动即 OOM

- 降低 `--max-model-len`
- 先用 `8192`
- 关闭其他占显存程序

### 请求超时

- 先用单张图测试
- 先确认服务端已完全启动
- 先确认模型名与服务端一致

### 返回内容不是 JSON

- 本项目后端已经做了解析失败回退
- 但你仍应优先检查服务端是否真的使用了 `Qwen3.5-4B`
- 再检查是否成功关闭了 thinking

## 推荐的实际落地顺序

1. 先跑 `mock` 后端确认 GUI/CLI 正常
2. 再用 `transformers serve` 或 `vLLM` 启动本地服务
3. 先用 10 张图验证输出格式
4. 再批量跑目录
5. 最后抽样检查误判

## 参考链接

- Qwen3.5-4B 模型卡：https://huggingface.co/Qwen/Qwen3.5-4B
- vLLM OpenAI 兼容服务：https://docs.vllm.ai/en/latest/serving/openai_compatible_server/
- vLLM 多模态输入：https://docs.vllm.ai/en/stable/features/multimodal_inputs/
