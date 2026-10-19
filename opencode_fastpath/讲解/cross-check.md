# 交叉检查清单（cross-check.md）

对象：《语音意图识别全参数详解》`main.pdf`（61 页）+ `exam.json`（56 题）。
口径：只信「代码 + 可复现实验」；作者文档过期处按代码修正并在正文/附录标注。

## 一、自查（逐条回源）

| # | 断言 | 出处 | 结论 |
|---|---|---|---|
| 1 | `intent_registry` 45 条；system 16 / 日常 7 / 游戏 1 / 运维 21 | `intent_config.json` 逐条统计（脚本复算） | 核实 |
| 2 | `enabled=false` 恰 3 条：dsh_new / dsh_run / open_web | 同上 | 核实 |
| 3 | says=271，pinyin=271，0 条不一致 | 同上 + `intents._check_pinyin` | 核实 |
| 4 | `len(PATCHES)=35`，`len(MARKERS)=33`，voice_agent.py 里 33 marker 各 1 次 | `patch.py` + `grep -c` | 核实 |
| 5 | selfcheck 41 项 | 逐项计数 `ok(` | 核实 |
| 6 | 三份 profile：misses=16、prob=0.30、hits=1、smooth=2、max_gain=80、target=0.05 | 三份 json | 核实 |
| 7 | aoe4 log_level=debug、其余 info；aoe4/ops min_prefix=2、共享 3 | 三份 json | 核实 |
| 8 | `vad_action._finish()` 用未定义名 `b`/`target` → NameError | 静态 + 合成音频复现（patch 掉 `vad_adapt.save` 避免写盘） | 核实（新发现） |
| 9 | reply IDLE_S=1.5 / PENDING_IDLE_S=0.8；micwatch 250ms/30s；typein 30ms/250ms | 对应源文件 | 核实 |
| 10 | 热词默认关（env `OPENCODE_ASR_BIAS=1` 才开）；LIMIT=120 | `asr_vllm.py:62-63`、`hotwords.py:23` | 核实 |
| 11 | 折叠表使「补农」=「捕农」同掩码；旧 ASCII 残余字段 8 位饱和（Blender/PowerShell 同 0xff） | `pinyin.py` + 独立脚本复算 | 核实 |
| 12 | 日常域（minp=3）1330 触发器 / 851 整串键 / 2207 前缀键；MatchStore 20B、8192 行 | json + `pinyin.Encoder` 独立脚本复算 | 核实 |
| 13 | 起音被吞两轮数值（100/105/250ms 与 184/220/560ms）都写明，且都记 7/10 | `onset_cmp.py` 复跑输出 + 用户记录 | 核实 |
| 14 | silero 甜点两轮数值都进正文 | `silero_probe.py` 复跑输出 + 用户记录 | 核实 |
| 15 | PDF 可检索「参数总表 / 推导 / 方块类比 / 原文注释」 | `miktex-pdftotext -enc UTF-8 main.pdf` | 核实 |

## 二、独立复核（换方法重算，避免同源盲区）

1. **掩码与表规模**：不调用仓库 `intents.load`（避免 actions/torch 依赖），改为
   直接解析 json、用 `opencode_fastpath.pinyin.Encoder` 复算触发器/整串键/前缀键；
   并用仓库旧公式复算 ASCII 残余字段（Blender/PowerShell/Steam/Hearthstone 均 0xff）。
2. **VAD 灵敏度**：用 `vad_sens_sweep.py` 把 25 条真实录音拼成连续流独立复跑，
   结果写入正文（并明确标注该批录音电平已够，不能用来否定现场经验）。
3. **起音量测**：用 `onset_cmp.py` 独立复跑，与用户第一轮结果并列呈现（数值不同、
   结论一致：7/10 段有起音被吞）。
4. **单句校准缺陷**：不信任文档描述，用合成音频直接调用 `_finish()` 复现 NameError。
5. **PDF 文本与标题**：用 `miktex-pdftotext` 抽全文，检索章节关键词，确认可搜。
6. **限制说明**：本环境禁止嵌套 subagent（`subagent depth limit reached`），
   因此「他人独立复核」由上述 5 个换方法的独立脚本/提取流程替代，未做 LLM 二次审阅。

## 三、未核实/未复跑项（正文已显式标注）

| 项 | 原因 | 正文标注位置 |
|---|---|---|
| 热词 A/B（无热词 / 补农 / 15 词 α=6/12/24） | 需 vLLM ASR 服务，遵守「不启动被测程序」 | ch14 §实测、ch09 §9.1 |
| `hs_launch_ab.py`（炉石两姿势） | 会启动本地游戏 | ch10 §10.3 |
| B 站「打开第 N 条」按需取页 | 需真实网络请求 | ch15 §作者风险 A3 |
| DSH-Voice-Input-Plugin 参数 | 外部对照，本仓内无源 | ch14 §14.4 |

## 四、勘误（材料过期数字，不静默修改）

- `patch.py` docstring「8 处 / markers 16」→ 实际 35 / 33。
- `patch.py` MARKERS 后注释「27 / 30」→ 实际 33 / 35。
- README / 现状与风险.md「33 处补丁 / 31 marker」→ 实际 35 / 33。
- 现状与风险.md「停止窗 19 帧 608ms」→ 三份 profile 均 16 帧 512ms。
- `vad_profiles.json` default misses：文档写 19，文件实际 16（why=用户指定 500ms）。
- COMMANDS.md「vad_calib_seconds 默认 12」「读两遍」→ 代码/配置 8s、单句口径
  （且 `_finish` 当前会 NameError）。
- 现状与风险.md 风险 A7「麦克风监控 info 看不到」→ `micwatch` 显式传 `level="info"`，
  info 可见。
- `opencode_fastpath/table.json`（42 条 / 738 触发器 / 408+1176 行）为过期产物；
  当前唯一开关在内存重建（日常域 1330 / 851 / 2207）。

## 五、排版级检查

- `xelatex` 三遍：**0 Error、0 Missing character、0 Undefined reference**。
- 剩余警告：Overfull 74 条 / Underfull 27 条，全部为表格/长标识符的毫米级溢出
  （≤约 120pt，多为 <20pt），不影响内容与数值正确性；已在正文尽量避免
  （`\sloppy`、`emergencystretch`、缩小 `\tabcolsep`、拆分长公式）。
- 体积：`main.pdf` ≈ 696 KB；61 页。
- `exam.json` 56 题：概念 23 / 反例 16 / 数值 17，难度 1:4、2:32、3:20；
  每题含 id/kind/q/answer/evidence/difficulty，id 无重复；evidence 均回指
  讲解章节或 annotated 片段。
