# vLLM 实时转写引擎 —— 落地参数与协议实测（2026-10-14）

> 目的：把 `vllm-asr` 从 0 副本/0.6B 换成 **1.7B + 真流式（/v1/realtime）**，
> 并记下所有**实测出来的、不写下来下次一定重踩**的东西。

---

## 一、最终生效的配置（vllm-asr Deployment args）

```
--model=/models/qwen3-asr-1.7b-official
--served-model-name=qwen3-asr
--host=0.0.0.0 --port=8000
--max-model-len=4096
--limit-mm-per-prompt={"audio": 1}
--trust-remote-code
--log-config-file=/logcfg/uvicorn.json
--gpu-memory-utilization=0.80          ← 见第二节
--cudagraph-capture-sizes=64
--quantization=fp8                     ← ⭐ 8GB 卡上必须
--hf-overrides={"architectures": ["Qwen3ASRRealtimeGeneration"]}   ← ⭐ 不指过去 WS 路由不注册
```
replicas=1；startupProbe 原本就有（failureThreshold 180 × 5s = 容忍 15 分钟）。

## 二、两个"必须"（都是实测撞出来的）

### ① 必须 `--quantization=fp8`
8GB 卡上 **1.7B bf16 装不下**，与现有服务无法共存：

| 项 | 实测值 |
|---|---|
| 1.7B bf16 权重加载 | **4.48 GiB**（`Model loading took 4.48 GiB`） |
| 非权重占用（vllm-embed + merged-intent + CUDA context + 激活） | **≈ 2.8 GiB** |
| `util=0.62` | `Available KV cache memory: -2.46 GiB` → EngineCore 起不来（`_check_enough_kv_cache_memory`） |
| `util=0.74` | 仍为 `-1.43 GiB` |
| 换算 | 即便 util 拉到 **0.98** 也不够 → 只能量化 |
| **`--quantization=fp8` + util 0.80** | 权重 4.48 → **3.13 GiB**，KV **+1.45 GiB** ✓ 起来了 |

`--quantization=fp8` 走的是 `Fp8PerTensorOnlineLinearMethod` + `CutlassFP8ScaledMMLinearKernel`（日志可见）
→ sm_120 支持 ✓。**精度实测没问题**：15.5s 八指令长句，离线端点转写逐字正确。

### ② 必须 `--hf-overrides` 把架构指到 realtime 类
- 模型目录 `config.json` 写的是 `architectures: ['Qwen3ASRForConditionalGeneration']`（离线类，**不实现 SupportsRealtime**）
- 真正实现 realtime 的是另一个注册类：`"Qwen3ASRRealtimeGeneration": ("qwen3_asr_realtime", "Qwen3ASRRealtimeGeneration")`，
  它 `class Qwen3ASRRealtimeGeneration(Qwen3ASRForConditionalGeneration, SupportsRealtime)` —— 权重通用
- 路由注册条件在 `entrypoints/serve/middleware/register.py:49`：`if "realtime" in supported_tasks`
- **不指过去的症状**：WS 握手 **HTTP 403**（Starlette 对未匹配的 WS 路由回 403，看起来像鉴权失败，其实路由根本没注册）
- 指过去之后日志：
  ```
  Resolved architecture: Qwen3ASRRealtimeGeneration
  Supported tasks: ['generate', 'transcription', 'realtime']
  OpenAIServingRealtime initialized for task: realtime
  Route: /v1/realtime, Endpoint: realtime_endpoint
  ```
- ✅ **同一个引擎两用**：`/v1/realtime`（WS 流式）+ `/v1/audio/transcriptions`（离线整段）都可用

## 三、协议实测（`_ws_probe.py` 在真机上跑的结论）

### 事件与字段（权威来自源码 `realtime/protocol.py`，已实测确认）
| 方向 | 事件 | 字段 |
|---|---|---|
| C→S | `input_audio_buffer.append` | `audio`: base64 **PCM16 / 16kHz / mono** |
| C→S | `input_audio_buffer.commit` | `final`: bool（默认 false） |
| C→S | `session.update` | **扁平 `{type, model}`** —— 不是嵌套在 `session` 里（写错会收到 error） |
| S→C | `session.created` | `id`, `created` |
| S→C | `transcription.delta` | **`delta`**（注意不是 `text`） |
| S→C | `transcription.done` | `text`, `usage` |
| S→C | `error` | `error`, `code` |

### 关键行为（三条实测）
1. **`append` 不触发解码，`commit` 才是触发点**：连续 append 60 片没有任何输出；一发 `commit`
   就开始吐该段音频的 delta。→ 客户端按"连续 append + 每隔 ~0.5~1s commit 一次"工作。
2. **delta 是"只追加"**（本样本）：`打开`→`知乎`→`，`→`搜索`→… 逐片追加，**没有回头改写前缀** ✓
   → 切分器可以简单做；但仍建议保留"稳定片段"门槛（流式本就会抖）。
3. **每次 commit 的输出都带 `language Chinese<asr_text>` 前缀**，客户端必须剥掉
   （离线路径已有同样的剥离逻辑）。

### ⚠️ 流式精度低于离线（实测同一段音频）
```
离线整段：打开 DeepSeek 网页版，搜索 DeepSeek 网页版…
流式 delta：打开 Discuz 的网页版，搜索 Discuz 的网页版…      ← 听错了
```
→ 这正是"流式抢跑 + 整段权威复核/回滚"这套设计的价值所在，不是可选优化。

## 四、当前部署状态与坑

| 项 | 状态 |
|---|---|
| `vllm-asr` | ✅ 1/1 Running，1.7B fp8 + realtime 路由，NodePort **30805** |
| `llama-asr` | ⚠️ **缩到 0**（显存放不下两个 ASR）；要回滚先缩 vllm-asr |
| app 的 ASR | ✅ 已切到 `http://172.20.71.101:30805/v1/audio/transcriptions`（`intent_config.json`，备份 `intent_config.json.bak-preurl`） |
| **portproxy 30805** | ❌ **加不上（需要管理员）** → 现用 **WSL IP 直连**；加了 portproxy 后可改回 `127.0.0.1:30805` |
| 延迟变化 | vLLM 离线转写 3.82s 音频 = **282ms**（llama-asr 同段约 120ms）→ 命令路略慢，可接受 |

回滚（纯配置）：
```bash
kubectl -n default scale deploy vllm-asr  --replicas=0
kubectl -n default scale deploy llama-asr --replicas=1
# 再把 intent_config.json 的 asr_http_url 改回 http://127.0.0.1:30803/v1/audio/transcriptions
# 备份：/tmp/vllm-asr.backup.yaml, /tmp/llama-asr.backup.yaml, intent_config.json.bak-preurl
```

## 五、下一步（客户端）

`agent_stream.py`（Windows 侧 app 用）要做的：
1. `append` 连续送音频（100ms/片，base64 PCM16 16k）+ 每 ~0.5~1s `commit` 一次
2. 收 `delta` 累积成文本，剥 `language Chinese<asr_text>` 前缀
3. 交给 `intent_split.Splitter` → 段形状检查 → A 路（规则直通 + 可撤销 + 稳定 2 次）
4. 松手后照旧整段 ASR → B 路权威 → 对账 / 回滚
5. 回退链：WS 不可用 → 离线 HTTP（现状）→ 进程内 GGUF

⚠️ Windows 侧的 `E:\voice_agent_build` venv **没有 websocket 库**
→ 二选一：`pip install websockets`（推荐，一行），或用标准库手写 WS（~80 行，零依赖）。
