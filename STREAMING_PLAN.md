# 流式转写落地方案（二者分流 + 引擎侧 vLLM realtime）

> 目标：**听写**边说边打字（延迟 ~150~300ms），**命令**仍走整段音频（准确性优先）。
> 结论先行：引擎侧不需要新东西 —— 你们镜像 `dsh/vllm-openai-patched:1.1` 里
> **已经有官方 vLLM 的 Qwen3-ASR realtime 实现**（`vllm/model_executor/models/
> qwen3_asr_realtime.py`，实现 `SupportsRealtime`），即 vLLM 的
> **实时 API（WebSocket `/v1/realtime`）**。

---

## 一、为什么必须"二者分流"

| 路径 | 用什么 | 为什么 |
|---|---|---|
| **命令**（打开/关闭/清空/站内搜索/听写开关…） | **整段音频 → 离线转写**（现有 HTTP 路径 ✓） | 命令要求**准确**：流式会丢跨段上下文 ✗（「打开DeepSeek网页版」被切成两段 → 意图判错 ✗）。而且命令很短，等一下无妨 ✓ |
| **听写**（非命令的自由说话） | **流式**（WebSocket realtime ✓） | 只要"文字尽快出现"✓，个别字词出错可接受 ✓；且听写**不触发任何动作**，错了没有副作用 ✓✓ |

分流判据：**先按现有流程判定**（规则直通 → 句向量 kNN 意图 → 实体链接）
- 判为命令 → 走离线路径（现状不变 ✓）
- 判为 `unknown`（非命令）→ 本来就要听写 ✓ → **改走流式** ✓✓

> 实现上更省事的一步：**音频侧就一直往流式引擎送**（它只产出文本 ✓ 不执行动作 ✓），
> 命令判定仍用**离线整段**结果 ✓ —— 两条路并行，互不干扰 ✓（代价：GPU 上多一份并发负载 ✓）

---

## 二、引擎侧（需要改的只有部署）

1. 现在 `vllm-asr` 是 **0 副本**（我曾按你要求缩容 ✓），它原来的配置是 **0.6B fp16** ✗
   → 改成 **1.7B 官方 fp16**（权重在 WSL 里是**完整**的 ✓ `/home/user/npc-models/qwen3-asr-1.7b-official/`，4.59GB ✓）
   ```
   --model=/models/qwen3-asr-1.7b-official
   --gpu-memory-utilization=0.62      # 4.7GB 权重 + KV，显存 8GB 够（需先把 llama-asr 或 vllm-embed 让出）
   --max-model-len=4096 --trust-remote-code --limit-mm-per-prompt={"audio":1}
   ```
2. **显存预算**（RTX 5060 = 8GB）：
   - 1.7B fp16 ≈ 4.7GB 权重 + ~0.6GB KV + 激活 ≈ **5.5GB**
   - 结论：**同一时间只能有一个 ASR 占 GPU** ✓
     → 建议：**vllm-asr(1.7B realtime) 取代 llama-asr(Q4)** ✓（fp16 质量 ≥ Q4 ✓，且 realtime+offline 一个服务全包 ✓）
     → `llama-asr` 保留为 fallback（缩容 0 ✓ 需要时再起 ✓）
3. **启动慢**：vLLM 加载 fp16 1.7B 实测约 **2~4 分钟** ✗（期间 `/health` 不通 ✓）
   → 必须给 Deployment 加 **`startupProbe`**（`failureThreshold: 30, periodSeconds: 10` ✓），否则探针会把它打成 `0/1` 来回重启 ✗（这是它上次 CrashLoop 之外的老毛病 ✓）
4. **WSL 一关就全停** ✗ → 客户端必须有回退（见第四节 ✓）

---

## 三、客户端侧（我们自己写，`agent_stream.py`）

```
麦克风(16k PCM16, 每 30~100ms 一块)
   └─ 命令判定用：累积整段（现状 ✓，段结束 → 离线 HTTP 30803 ✓）
   └─ 听写用：并发地 append 到 WebSocket /v1/realtime
                ├─ 收 "…transcription.delta"  → typein.type_text(delta)  ← 边说边出字 ✓
                ├─ 收 "…transcription.completed" → 补空格 / 结束该段 ✓
                └─ 断线 → 自动重连；连续失败 → 回退离线路径 ✓
```
关键点：
- **打字必须走 `typein.type_text`**（已实现 UIA 判定 + SendInput ✓ + 事后验证 ✓）
- **听写模式为 `off` 时不发流** ✓（省 GPU ✓）
- **收到 delta 要检查焦点仍在输入框** ✓（焦点跑了就别乱打 ✓）
- **丢弃处理**：模型可能修订前面的 delta ✗ → 简化做法：**只追加、不修订** ✓（听写场景可接受 ✓）；要修订就用 `typein.send_backspaces` ✓（已实现 ✓）

---

## 四、回退链（必须实现，按顺序自动降级）

1. **vLLM realtime（WS）** —— 首选（同模型、真流式 ✓）
2. **离线 HTTP（现用 ✓ 30803 llama-server / 或 vLLM 的 `/v1/audio/transcriptions`）** —— WS 不可用时 ✓
3. **进程内 CPU GGUF（`asr_backend=gguf` ✓ 已实现 ✓）** —— WSL 整个挂了时 ✓

`_make_asr()` 里加一次健康探测即可（现有 http 分支已有此模式 ✓）。

---

## 五、验收标准（都可用现有工具测）

| 指标 | 现状 | 目标 |
|---|---|---|
| 听写感知延迟（说完 → 文字出现） | ~0.83s（静音 704ms + ASR 90ms ✓） | **< 0.3s**（字随声出 ✓） |
| 命令准确率 | 语料 val 0.9793 ✓ / 端到端待测 | **不下降** ✓（命令走离线 ✓） |
| 误打风险 | 命令永远优先 ✓（已修 ✓） | 保持 ✓ |
| 引擎可用性 | llama-asr 1/1 ✓ | vLLM realtime 1/1 + startupProbe ✓ |

工具：`perf_monitor.py`（实时指标 ✓）、`stage_breakdown.py`（逐句拆解 ✓）、
`ab_sensevoice.py`（外部引擎 A/B 模板 ✓）。

---

## 六、执行顺序（每步可独立验证）

1. **起 vllm-asr（1.7B fp16, offline 先跑通）** → 用 `ab_*` 思路比 Q4 vs fp16（同模型，质量应 ≥ ✓）
2. **验证 vLLM realtime API 可用**（`/v1/realtime` WS 握手 + `input_audio_buffer.append` ✓）
3. **写 `agent_stream.py`**（WS 客户端 + delta 打字 ✓）→ 单测：喂一段 wav，看是否逐字打进输入框 ✓
4. **接入 `run_mic`**（听写走流式，命令走离线 ✓ = 二者分流 ✓）
5. **加回退链 + 压测**（断 WSL ✓ 看是否自动降级 ✓）

---

## 七、已知风险

- **WSL 重启后 WSL IP 变** ✗ → 用 `netsh portproxy`（现在 30803 那条 ✓）或改 `.wslconfig` 的 mirrored（实测对 k3s NodePort 无效 ✗）
- **VRAM 只有一个 ASR 的位置** ✗ → 需要 tmux/运维约定：**要么 vllm-asr 要么 llama-asr** ✓
- **vLLM 启动 2~4 分钟** ✗ → `startupProbe` 必须配 ✓，否则看起来像"挂了" ✗
- **流式的 delta 修订** ✗ → 先"只追加" ✓，避免复杂化
