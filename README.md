# 语音意图 Agent（voice-intent-agent）

常开流式监听麦克风 → 意图识别 → **自动执行** Python 动作。
说「启动记事本」就真的弹出记事本；说「打开B站」就用默认浏览器打开。

```
麦克风 → VAD 分句 → [A(18层) → ctc_head → V1帧级混合 + V3分词查表 → B] → 意图头
                 → 分句规则解析参数 → Python 动作
```

---

## 本仓库说明

这是**源码仓库**，用于展示架构与实现。以下内容**不包含**（体积大或与本机环境绑定）：

| 不包含 | 原因 |
|---|---|
| `model/intent.pt`、`model/bundle.safetensors`（283 MB） | 训练产物，可复现 |
| `*.exe`（打包产物） | 见「打包」一节自行构建 |
| `wavs/`、`utt/`（录音与合成语音） | 数据，不入库 |
| `intent_index.json`、`*/profiles/*/index.json` | 派生索引，由脚本重建 |
| 真实意图配置里的**本机绝对路径** | 见 `intent_config.json` 的 `apps[].path` |

复现步骤见文末「复现」。

---

## 快速开始

```powershell
# 看所有能执行的命令（不启动监听）
python voice_agent.py --list

# 干跑（只打印会做什么，不真的启动程序/开浏览器）—— 建议先跑这个
python voice_agent.py --dry-run

# 正式常开监听（Ctrl+C 退出）
python voice_agent.py

# 不用麦克风，直接测音频文件
python voice_agent.py --wav wavs\train_launch_app_0000.wav

# 指定麦克风设备
python voice_agent.py --device 2
```

---

## ⭐ 意图配置文件在哪 —— `intent_config.json`

**这就是你要改的文件，和主程序放在同一目录。**

```
voice_agent\
  voice_agent.py           ← 主程序
  intent_config.json       ← ⭐ 改这里
  model\
    intent.pt              ← 意图头（训练产物）
    bundle.safetensors     ← A + B + 词嵌入表（283 MB）
```

### 改了【不用重训】的（改完重启程序即生效）

| 段 | 能改什么 |
|---|---|
| `apps[].path` | 程序路径。换路径、加参数都行 |
| `apps[].aliases` | 别名。说「开一下XX」里的 XX 认哪些写法 |
| `urls[].url` | 网址 |
| `files[].path` | 文件路径 |
| `search.web_engine` | 搜索用哪个引擎（`{q}` 是占位符）|
| `search.file_manager_root` | 文件管理器搜索的起点目录 |
| `triggers.*` | 触发词表（「打开」「搜索」「精确」…）|
| `behavior.*` | VAD 灵敏度、置信度阈值、dry_run 等 |

### 改了【需要重训】的（因为类别数变了）

- **增删 `apps` / `urls` / `files` 里的条目**
- **增删 `intents` 段里的意图**

重训：

```powershell
python build_dataset.py      # 从配置生成训练数据（含 TTS 任务表）
python tts_batch.ps1         # 合成语音（用 Windows SAPI）
python train_intent.py       # 训练（约 15 分钟）
python package_model.py      # 打包 bundle
```

> 💡 只想加一个程序/网址？也可以走**增量添加**：新意图只训头的那一行 + 400 条老数据回放，
> **0.2 秒**搞定，不用全量重训（见 `incremental_test.py`）。

---

## 能执行什么

跑 `python voice_agent.py --list` 会打印完整清单。当前配置下：

### 1. 启动程序（24 个闭集）
「启动记事本」「打开任务管理器」「运行Blender」「帮我打开Ollama」…
→ `subprocess.Popen(exe路径)`

### 2. 打开网址（16 个闭集）
「打开B站」「打开B站网页」「上知乎」「访问GitHub」…
→ 默认浏览器打开

### 3. 打开文件（7 个闭集）
「打开CUDA编程指南文件」「打开并行规约」…
→ `os.startfile(路径)`（用系统默认程序）

### 4. 参数化命令（用转写文本，不查闭集）

| 说法 | 动作 |
|---|---|
| 「打开 xxx」 | xxx 不在上面任何闭集 → 默认浏览器打开（像网址就直开，否则搜索）|
| 「搜索 xxx」 | 默认浏览器搜索 |
| 「搜索文件 xxx」 | **文件管理器搜索**（Windows Search 协议）|
| 「精确查找 xxx」 | **必须含「精确」** → 用**转写原文**搜索，不纠错不联想 |

### 5. 不执行
其它任何话 → `unknown`，什么都不做。

---

## 分句规则（`triggers` 段控制）

```
精确查找 xxx   → exact_search    ← 最高优先，且【必须】含"精确"
搜索文件 xxx   → file_search     ← 含"文件"才是文件搜索
搜索 xxx       → web_search
打开 xxx网页   → open_url        ← 后缀强制按网址闭集匹配
打开 xxx文件   → open_file       ← 后缀强制按文件闭集匹配
打开 xxx       → 先查闭集（程序/网址/文件），查不到才 open_web
其它           → unknown
```

**关键区分（模型必须学会的）**：

```
打开B站        → open_url    （在闭集）
打开github.com → open_web    （不在闭集）
搜索python     → web_search
搜索文件python → file_search  ← 只差"文件"两个字
精确查找python → exact_search ← 只差"精确"两个字
```

---

## 架构

```
wav → A (Qwen3-ASR audio_tower, 18层, 冻结)  →  h [T,896]
    → ctc_head (896→6629, 可训 5.95M)
    ├─ V1: p=softmax(logits/τ) → z=pᵀ·E_pad → 时间压缩 r=3 → [CLS,z] → B → CLS1
    └─ V3: CTC贪心解码 → 字符串 → dmeta分词 → 精确词嵌入行 → B → CLS2
    → concat(CLS1,CLS2) 1536维 → intent_head → 8 类
```

- **V1 提供软容错**：转写错一点，混合体只偏一点，还能救
- **V3 提供对齐**：与纯文本路径的 cos 达 **0.95**（V1 只有 0.35），所以加意图时可以用文本
- **两者拼起来最优**：实测 V1 **97.87%** / V3 **95.51%** / **V5 集成 98.11%**

模型总量 **288.56 M**（A 186.38 + B 85.05 + 词嵌入 16.23 + 头 0.9），fp32 781 MB。

### 为什么 ctc_head 和 intent_head 是分开的

- `ctc_head` 让 A 的连续特征落到**字符**上（可解释、可对齐）
- `intent_head` 在 B 的语义空间上做分类
- A 和 B **全程冻结**，所以特征可缓存 → 增量加意图只要 0.2 秒

---

## 常开流式

Qwen3-ASR 的 audio_tower **本来就是 chunk 化的**：

```
n_window       = 50   → chunk = 100 mel 帧 = 1.0 秒
n_window_infer = 800  → 注意力窗口 = 800 mel 帧 = 8.0 秒
卷积逐 chunk 独立（chunk-local）
```

所以流式不需要改架构。实测**听到句子的 60%（1.7 秒）就已经满精度**。

主程序用 **VAD 分句**（webrtcvad，静音 700 ms 判停）+ 逐句推理。
单条推理 ~100 ms（CPU）。

---

## 测试

```powershell
python actions.py --list --selftest   # 规则分句自测（不需要模型）
python vad.py wavs                    # VAD 分句自测
python test_agent.py                  # 端到端测试（模型 + 动作）
```

---

## 打包

```powershell
.\build_exe.ps1
```

用只含 CPU 版 torch 的独立环境打包，产物在 `dist\`。

---

## 复现

```powershell
# 1. 装依赖（torch CPU 版足够）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. 改 intent_config.json 里的 apps[].path / urls[].url / files[].path 为你的本机路径

# 3. 生成训练数据 → 合成语音 → 训意图头（约 15 分钟）
python build_dataset.py
python tts_batch.ps1
python train_intent.py
python package_model.py

# 4. 先干跑验证，再正式监听
python voice_agent.py --dry-run
python voice_agent.py
```

---

## 已知边界

- **闭集意图**：程序/网址/文件是固定集合，加新目标要重训（或走 0.2 秒增量）
- **中文为主**：ASR 用 Qwen3-ASR，英文长句切分不稳
- **Windows 绑定**：`os.startfile`、Windows Search 协议、SAPI 合成，Linux/容器里跑不了动作层
- **无关人声**：VAD 会把旁人说话切句送判，靠 `unknown` 兜住，但没有说话人分离
