# voice-intent-agent

常开流式的**全离线**语音意图管线：`语音 → ASR 文本 → 句向量 → kNN 意图 → 执行`。

## 核心设计

**意图识别走「零训练」路线**：加一个意图 = 加示例 + 重建索引，**不碰模型权重**。
意图是数据问题，不是训练问题。

- 句向量 + **kNN 检索**（不用质心：质心会把同一意图的多种说法平均掉，实测 **0.913 → 0.964**）
- **开集拒识**：阈值 + 「最近 vs 次近」间隔（等价于 Rasa 的 `nlu_threshold` / `ambiguity_threshold`）
- **ASR 走进程内 llama.cpp（C++）**：**666 MB vs torch fp32 3130 MB（4.7×）**，不占显存

## 关键实测数字

| 项 | 值 |
|---|---|
| 意图索引 | 1,144 条 / 9 类 / 768 维 |
| kNN vs 质心 | 0.913 → **0.964** |
| 量化对照 | Q4_K_M / Q8_0 / bf16 三档精度**同为 100%** → 瓶颈不在量化 |
| `n_ctx` 4096 → 512 | 省 **394 MB**，速度不变 |
| 错例驱动修复 | 127 个错例中 **116 个是「→ unknown」** → 定位到「置信度门排在规则直通之前」，改优先级后修复 |
| 回归集 | 324 条真实录音 |

## 目录

| 目录/文件 | 内容 |
|---|---|
| `voice_agent.py` | 主服务 |
| `actions.py` | 意图执行层 |
| `asr_*.py` | 三种 ASR 落点（llama.cpp / vLLM / HTTP） |
| `intent_*.py` | 句向量、意图配置、kNN 路由 |
| `build_dataset.py` / `build_en_dataset.py` | 语料与索引构建 |
| `exp_*.py` / `ab_*.py` | 实验与 A/B 脚本 |
| `_*_selftest*.py` | 自测 |
| `命令表.md` | 意图与命令对照 |

## 复现

```bash
# 1. 建索引（需要句向量服务；索引文件不进 git）
python _build_idx.py

# 2. 自检
python _wiring_smoke.py

# 3. 启动
python voice_agent.py --doctor
```

## 已知边界

- **无关人声仍会被切句送判**：已建 324 条回归集但未根治
- **ASR 三种落点（本地 CPU / 本地 GPU / 云 GPU）的延迟-成本-隐私对比尚未做**
- Windows 侧无 Triton；8 GB 卡上无法同时驻留多个服务
