# opencode_fastpath 命令说明

> **OPENCODE** ｜ 2026-10-19 ｜ 回滚细节见 `ROLLBACK.md`

## 0. 改名 + 唯一意图开关（先看这个）

**产品名：语音意图识别**（原「帝国时代4 语音指挥」；窗口标题/横幅/文档都已改）

**唯一开关 = 一份 json**：`voice_agent/intent_config.json` → **第一个 key** `intent_registry`

```jsonc
{"intent": "show_commands",        // 意图
 "says": ["命令列表", "有哪些命令"],  // 说法（空 = 由 apps/urls/files 自动展开）
 "pinyin": ["ming ling lie biao", "you na xie ming ling"],  // 说法的拼音标注（加载时校验）
 "desc": "打开这份意图清单",         // 具体作用
 "enabled": true,                  // 是否开启
 "tag": "system"}                  // 意图域
```

- `enabled`：`true`=参与 ｜ `false`=不参与（**任何路径都不执行**）；**不用 null**
- **参与匹配 = `enabled` 为 true 且（`tag == 当前域` 或 `tag == system`）**
- `tag`：`system` / `office` / `aoe4` / `ops`（`system` 任何域都生效）
- `pinyin`：说法的拼音标注（写明白；加载时自动核对，不一致会打 WARN）
- **动态加载**：每句匹配前检查 json 的 `mtime+size`，改完保存**立即生效**（不用重启）；
  也能说「重新加载意图」强制刷新；日志打一行 `意图清单：域=… ｜ 启用 N / 禁用 M ｜ …`
- 三个 profile（`intent_config.json` / `ops/...` / `aoe4/...`）都通过
  `behavior.intent_registry_file` 指向**同一份**；profile 里只留行为参数（ASR/VAD/apps）
- 旧表机制（`table_*.json` / `table_system.json`）**不再参与匹配**（文件保留备查）
- **「命令列表」= 打开这份 json**（不再生成 .md）
- 关掉后的表现：`❌ 意图 xxx 已关闭（… enabled=false …改成 true 即生效，无需重启）`

## 0.11 说错了/听不懂时怎么办（任何域都能说）

| 你说 | 意图 | 做什么 |
|---|---|---|
| 听不清 / 听不见 / 声音小 / 提高灵敏度 | `sens_up` | ASR 前增益 ×1.5（上限 600×）+ 起判 prob −0.02，**写回 profile**（重启仍在）|
| 太吵了 / 声音太大 / 降低灵敏度 | `sens_down` | 增益 ÷1.5（下限 60×）+ prob +0.02 |
| 灵敏度 / 当前增益 / 识别参数 | `sens_show` | 报当前参数（不改） |
| 识别不准 / 总听错 / 转写有问题 | `asr_diag` | 报参数 + 下一步建议 |
| 调整静音判定 / 静音判定 / 校准静音 | `vad_calibrate` | 现场校准静音窗（读两遍测试文字） |
| 命令列表 / 有哪些命令 / 能做什么 | `show_commands` | 打开意图清单 |
| 重新加载意图 / 刷新意图 | `reload_intents` | 强制重载意图清单 |
| 开始输入 / 打开输入 | `dictation_on` | 进听写模式（**终端也必须先说这句**才往里打） |
| 终止输入 / **关闭输入** / 关闭听写 | `dictation_off` | 退出听写模式 |
| 清空输入 / 清空输入框 | `clear_input` | 清空焦点输入框（终端里不做） |

> 终端窗口：**auto 模式不自动听写**（避免随便说话被打进 shell）；说「开始输入」才会往终端打。

```powershell
# 例：只留系统命令（把 aoe4 的农民训练关掉）
#   打开 voice_agent\intent_config.json → intent_registry → 找到 aoe4_train_villager
#   把 "enabled": null 改成 "enabled": false，保存即可（下一句就生效）
```

## 0.1 三个入口

| 命令 | 作用 |
|---|---|
| `.\run_src_voice_agent.cmd` | **原入口**（不挂快路径，行为与以前完全一致） |
| `.\run_opencode_shadow.cmd` | 影子：部分文本/整句都只**记录**，且加 `--dry-run`（动作也不执行） |
| `.\run_opencode_live.cmd` | live：部分文本命中即**提前执行**（破坏性动作仍拦截） |

> live 就是执行开关，不依赖 `intent_config.json` 里的 `stream_exec`。

## 1. 常用（在 `voice_agent` 目录下）

```powershell
set PY=E:\voice_agent_build\Scripts\python.exe

# 自检（16 项判据，不需要模型/麦克风）
%PY% -X utf8 -m opencode_fastpath.selfcheck

# 影子汇总（rescue 率 / 快路径耗时）
%PY% -X utf8 -m opencode_fastpath.report

# 只跑演示（影子/学习/live 语义）
%PY% -X utf8 -m opencode_fastpath.demo

# 从 intent_config.json 重建基表（42 条 / 738 触发器）
%PY% -X utf8 -m opencode_fastpath.build_table
```

## 2. 学新说法（教师产物入库）

```powershell
# additive=1（默认）: 只在"规则判 unknown"时接管
%PY% -X utf8 -m opencode_fastpath.learn --intent launch_app --arg 记事本 `
      --text "开一下记事板" --text "打开记事板"

# 想看 live 接管破坏性意图（慎用）
$env:OPENCODE_FASTPATH_DESTRUCTIVE=1
```

破坏性意图（默认拦截）：`close_app` `dsh_run` `dsh_new` `clear_input` `dictation_on` `dictation_off`

## 2.5 静音窗（VAD）校准与自适应

```powershell
# 换人时：拿这个人的真实录音跑校准（默认目标：被切断的句子 ≤3%）
E:\voice_agent_build\Scripts\python.exe -X utf8 -m opencode_fastpath.vad_calib `
  --glob "utt/*_mic_*.wav" --n 200                 # 只看推荐值
# 写进说话人档案（运行时自动加载）
... --save-profile alice
# 直接写进 profile 配置
... --apply aoe4        # 或 ops / office
```

算法：逐 32ms 帧取 Silero 概率（本仓 PreGain，阈 0.4，平滑 3）→ 只统计**句中停顿**
→ 取"切碎率 ≤ 目标"的**最小静音窗**。判停延迟 = 静音窗。

运行时自适应（`vad_adapt.py`，已接在 [M]/[N]）：
- 发生一次「与上一段拼接」= 上次被切碎 → 静音窗 **+1 帧**（每两次拼接加 1，防抖）
- 连续 **20 段**无拼接 → **-1 帧**（控延迟）
- 区间 = 以档案值为中心 ±8 帧（[9,25] 帧 ≈ 288~800ms）；档案存 `opencode_fastpath/vad_profiles.json`
- 说话人：`set OPENCODE_SPEAKER=alice`（默认 `default`）

## 2.6 语音校准静音窗（意图动作 / 任何领域可用）

对麦说：**「调整静音判定」**（同义词：校准静音 / 静音窗 / 调整静音窗 / 调整静音）
→ 日志提示「请读出测试文字（12 秒，读两遍，中间停一下）」→ 读出提示文字
→ 采集结束自动校准：静音窗 = clamp(1.15×句内最大停顿, 0.45×句间停顿, 0.8×句间停顿)
→ 写入运行中的 VAD + 说话人档案 `vad_profiles.json`，日志给出 `X→Y 帧（…ms）`

- 该动作走【系统命令表】`table_system.json`，**任何领域都生效**（AoE4 域表关着也认）
- 走【域名表】时（办公/运维）也包含该意图（`vad_calibrate`）
- 相关参数（写进 profile 的 `behavior`）：`vad_calib_seconds`(默认 12)、`vad_calib_prompt`(默认内置测试文字)

## 2.7 预滚（VAD 的 640ms 怎么砍 —— 可校准 + 在线自适应）

VAD 固定把 **640ms 预滚**放进段里；运行时决定**砍掉多少**（减法）：

```
keep = min(640 − ahead_trim_ms, 总长 × ahead_frac)      再在语音起点前留 onset_margin_ms
```

| 键 | 默认 | 含义 |
|---|---|---|
| `ahead_trim_ms` | `640 − ahead_cap_ms` | **从 640ms 里砍掉多少**（0~500；keep 至少 140ms） |
| `ahead_frac` | 0.2 | 再按总长比例设上限（防长段送太多静音） |
| `onset_margin_ms` | 100 | 语音起点前额外保留的余量 |

**在线自适应**（`vad_adapt.on_segment_head`，每段都测"段首能量比"= 段首100ms RMS / 整段 RMS）：
- 比值 **>0.35** → 起音被切 → `ahead_trim_ms −40`（少砍）
- 比值 **<0.06 连续 15 段** → 段首全静音 → `ahead_trim_ms +40`（多砍、更快）
- 每次调整写日志 + 落档案；`ahead_cap_ms` 同步保持 `640 − trim`

实测（真机）：`预滚裁剪 8→0ms（VAD 640ms → 保留 640ms；段首能量比 0.46（起音被切））`

## 2.8 日志级别

- 默认 **info**：只打印 **转写 / 耗时 / 动作结果 / 错误**（另有 banner 与一行级别声明）
- **debug**：再加 VAD 参数、拼音标注、裁剪明细、自适应调整、VAD 段入队、系统表加载…
- 切换：`set VOICE_AGENT_LOG=debug`（临时）｜或 profile `behavior.log_level: "debug"`（持久）
- 代码侧：`from dlog import log, debug; log(..., level="debug")` / `debug(...)`；WARN/ERR 永远显示

## 2.9 听写耗时怎么读（typein.py 验证）

- `文字进框 Xms`：所有字符 SendInput 完成的那一刻 —— **这才是"用户看见字"的延迟**
- `日志写出 Yms`：函数返回、日志打出来的时刻（Y > X，差额 = `其中验证确认 Zms`）
- 验证做什么：打完字后读一次输入框，要求**出现次数增加**（防"假成功"：焦点不在可编辑控件时
  SendInput 返回 2 但字被丢掉）
- 实测：原固定 `sleep(0.15)` → 验证 ≈153ms；现改为**轮询**（30ms 一次，读到新增即停，
  上限 250ms）→ 正常 ≈30ms；字没进去时才等满 250ms 再判失败
- 想省掉整个验证：`typein.type_text(..., verify=False)`（不推荐，会回到假成功）

## 2.10 系统意图「命令列表」（任何领域都生效）

- 说：**命令列表 / 命令清单 / 有哪些命令 / 有什么命令 / 能做什么 / 能干什么 /
  能说哪些话 / 有什么功能 / 使用说明 / 怎么用**
- 做什么：**打开 `voice_agent\intent_config.json`**（唯一意图清单：意图→说法→作用→是否开启）
- 触发机制：intent_registry 里的 system 标签（任何意图域都参与）
- 想临时看渲染版：`python -m opencode_fastpath.intents <profile.json>`
- 重新生成清单：无（清单就是配置本身；改 json 即改清单）

## 2.11 听写也能打进终端

原来终端被判成"不在输入框"（终端没有 Win32 caret，UIA 读不到输入行）→ 听写不打。现在：

- `typein.TERM_CLASSES` 认这些终端：`ConsoleWindowClass`（cmd/conhost）、
  `CASCADIA_HOSTING_WINDOW_CLASS` + `Windows.UI.Input.InputSite.WindowClass`（Windows Terminal）、
  `mintty`（Git Bash）、`conemu`/`TabWindowClass`、`alacritty`/`wezterm`/`putty`
- 判定：`caret_state()["is_terminal"]` → `in_text_field()` 直接返回"可输入"
- **跳过回读验证**：终端 UIA 给的是整个滚动缓冲区，回读会把"写进去了"误报成"没写进去"
  → `type_text` 在终端里不验证，消息写成「终端窗口：键盘已发送，不做回读验证」
- `focus_input()`（「开始输入」）在终端里直接算成功（终端本来就"焦点在哪往哪打"）
- 终端里**不加尾随空格**（`type_text` 在终端里 `rstrip()`）：shell 命令前后多空格没意义
- **「清空输入」在终端里拒**：Ctrl+A / Delete 在 shell / REPL / TUI 里是别的东西
  （PSReadLine 移至行首、删字符、程序自己的快捷键）→ `clear_field()` 明确返回
  「终端里没有可清空的输入框」。听写能进终端，"清空"不适用
- 「清空输入」「开始输入/终止输入」「关闭X」是**只在 triggers 里、不在 intents 数组**的
  命令（靠 voice_agent 的规则直通跑）；命令清单现在也把它们列出来（办公域有，运维域没配）

## 2.12 统一返回（每个意图都有 message；一句一个「总返回」）

`opencode_fastpath/reply.py`（启动时由 `[OPENCODE-G]` 安装）：

- **保证 message**：`dispatch()` 返回空消息时自动补「已完成/未执行（intent 没有返回消息）」
- **总返回**：一句里所有意图（含 A 路抢跑、听写）汇总成一行
  `⇒ 总返回：2 成功 / 1 失败 ｜ a→✅ …；b→❌ …`，并存到 `agent.last_reply`
  （`{"ok", "executed", "failed", "items", "text", "ts"}`）
- **异步也能等**：听写是异步线程（不能阻塞音频），`reply` 会把它记成 pending；
  收账由"空闲 1.5s 无新账"定时器触发，`pending` 未清零不结账
  → 外部调用方（工具/MCP 桥）可以：
  ```python
  agent.handle(wav)
  rep = reply.wait_utterance(agent)     # 阻塞到本句真的处理完（含听写）
  print(rep["ok"], rep["text"])
  ```
- 回滚：`reply.uninstall()`（或删掉 `[OPENCODE-G/W]` 后 `patch --apply`）

## 3. 补丁管理

```powershell
%PY% -X utf8 -m opencode_fastpath.patch --status     # 期望 markers: 5
%PY% -X utf8 -m opencode_fastpath.patch --apply      # 幂等（已打过会先从备份还原再重打）
%PY% -X utf8 -m opencode_fastpath.patch --rollback   # 从备份还原 voice_agent.py
```

## 4. k8s realtime 开关（启用/回滚 `/v1/realtime`）

```bash
# 在 WSL 里
bash "/mnt/c/Users/user/Desktop/ds harness/voice_agent/opencode_fastpath/k8s/apply.sh"
bash "/mnt/c/Users/user/Desktop/ds harness/voice_agent/opencode_fastpath/k8s/rollback.sh"
```

不加 `--hf-overrides` 的症状：WS 握手 **HTTP 403**（路由未注册，不是鉴权问题）。

## 5. 环境变量

| 变量 | 值 | 含义 |
|---|---|---|
| `OPENCODE_FASTPATH` | `shadow` / `live` / `off` | 快路径模式（默认 off；两个 cmd 会设） |
| `OPENCODE_FASTPATH_DESTRUCTIVE` | `1` | 放行破坏性意图的 live 接管 |
| `PYTHONIOENCODING` | `utf-8` | 控制台中文不乱码（cmd 已设） |

## 6. 日志与文件

| 路径 | 内容 |
|---|---|
| `run_opencode_live.log` / `run_opencode_shadow.log` | 主日志（Tee 出来的完整输出） |
| `opencode_fastpath/logs/decisions-*.jsonl` | 每次决策：text/pinyin/key/fp/model/延迟 |
| `opencode_fastpath/logs/rescue-*.jsonl` | 规则 unknown 但快路径能救回的样本 |
| `opencode_fastpath/logs/a_path-*.jsonl` | A 路/流式早火：命中或 `would_exec` |
| `opencode_fastpath/logs/pending-*.jsonl` | 教师候选（规则与快路径都没命中） |
| `opencode_fastpath/table.json` | 快路径表（条目+音节 id；运行期产物） |
| `opencode_fastpath/OPENCODE_MANIFEST.json` | 部署清单（24 文件 sha256 + 补丁/k8s 信息） |

## 7. 一键全回滚

```powershell
cd "C:\Users\user\Desktop\ds harness\voice_agent"
E:\voice_agent_build\Scripts\python.exe -X utf8 -m opencode_fastpath.patch --rollback
wsl kubectl patch deploy vllm-asr --type=json --patch-file "/mnt/c/Users/user/Desktop/ds harness/voice_agent/opencode_fastpath/k8s/rollback_args.json"
Remove-Item -Recurse -Force ".\opencode_fastpath"
Remove-Item -Force ".\run_opencode_shadow.cmd",".\run_opencode_live.cmd",".\OPENCODE_DEPLOY_20261018.md"
```

## 8. 现象速查

| 现象 | 含义 / 处理 |
|---|---|
| 日志出现 `⚡ OPENCODE 快路径提前执行 xxx` | 流式/段级早火命中（live） |
| 日志出现 `OPENCODE 快路径执行未成功（…）→ 交回主路径` | 命中但动作失败（如闭集里没有该条目）→ 自动回退 |
| 日志出现 `流式不可用 → A 路（抢跑）关闭` | realtime 没连上（先跑第 4 节 apply.sh） |
| 想只看不执行 | 用 `run_opencode_shadow.cmd` |
| 说了一句没反应 | 看 `pending-*.jsonl`，用第 2 节 `learn` 补条目 |
