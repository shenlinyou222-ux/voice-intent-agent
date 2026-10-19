# ROLLBACK — opencode 部署回滚说明

> **OPENCODE** ｜ session `ses_eb3637506ffeoeQ5UIVdyl40q4` ｜ 2026-10-18

## 变更范围

### 新增

| 路径 | 类型 |
|---|---|
| `voice_agent/opencode_fastpath/` | 新增目录（模块、基表、日志） |
| `voice_agent/run_opencode_shadow.cmd` | 新增启动器（影子） |
| `voice_agent/run_opencode_live.cmd` | 新增启动器（live） |
| `voice_agent/opencode_fastpath/cmd_list.py` | 新增：「命令列表」system 意图（写文件+打开；触发靠 table_system.json） |
| `voice_agent/opencode_fastpath/reply.py` | 新增：统一返回（保证 message / 一句一个总返回 / 异步可等） |
| `voice_agent/opencode_fastpath/intents.py` | 新增：**唯一意图开关**（intent_config.json → intent_registry）+ mtime 动态热加载 + ACTIONS 门槛 |
| `voice_agent/语音命令列表.md` | 运行期生成（说「命令列表」重写一遍；可随时删） |
| `voice_agent/OPENCODE_DEPLOY_20261018.md` | 新增标记 |

### 修改（可一键还原）

| 文件 | 改动 | 备份 |
|---|---|---|
| `voice_agent/voice_agent.py` | **29 处 `[OPENCODE-*]` 接入点**（A 拼音识别日志 / B 部分文本 / C realtime 启动 / D 音频块推送 / E 段边界 / F PTT 崩溃修复 / G 唯一开关+统一返回注册 / H 标点→空格 / I 注入文本日志 / J 听写拼接窗 / K 听写拼接开关 / L 拼接状态存原始转写 / M 静音窗档案+自适应启动 / N 自适应信号 / O 注册语音校准动作 / P 校准音频 tee / Q 复用日志静音 / R 预滚参数化(减法) / S 预滚在线信号 / T ASR 前增益上限可配 / U 起判自适应信号 / V 裁剪行表述 / W 统一返回·一句开始 / X 日志级别 / Y 听写耗时表述 / Z 游戏横幅 / A2 产品改名+横幅 / B2·C2·D2 文案） | `voice_agent.py.bak-opencode-20261018` |
| `voice_agent/asr_vllm.py` | **1 行**：`HOTWORD_ENABLED` 改为 env 驱动（`OPENCODE_ASR_BIAS=1` 才用配置里的 hotwords/bias；默认与原行为一致） | `asr_vllm.py.bak-opencode-20261019` |
| `voice_agent/dlog.py` | **日志级别**：`debug<info<warn<error`；INFO 只留转写/耗时/动作结果/错误；`set_level()` + `VOICE_AGENT_LOG` | `dlog.py.bak-opencode-20261019` |
| `voice_agent/typein.py` | **听写验证**：固定 `sleep(0.15)`（≈153ms）→ 轮询确认（30ms/次，读到"新增"即停，上限 250ms；字没进去才等满）；**终端**：认 `ConsoleWindowClass` / `Cascadia` / `mintty`… → 可输入、跳过回读、去尾随空格；终端里拒「清空输入」 | `typein.py.bak-opencode-20261019` |
| 改名（纯文案） | `aoe4/run_voice_aoe4.ps1`、`aoe4/run_voice_aoe4_opencode.ps1`、`openra/run_voice_game.ps1`、`run_src_voice_agent.cmd`、`aoe4/profiles/aoe4-v1/profile.json`、`aoe4/README_设计与使用.md`、`aoe4/.../intent_config.aoe4.json` 描述 → **语音意图识别**；桌面快捷方式 `帝国时代4语音指挥.lnk` → `语音意图识别.lnk` | 各文件旁 `.bak-opencode-rename` |
| `voice_agent/aoe4/profiles/aoe4-v1/intent_config.aoe4.json` | **3 行**：`enable_dictation: false→true`；`dictation_punct_to_space: true`；**`vad_required_misses: 5→17`**（静音窗 160→544ms，真实录音校准） | `intent_config.aoe4.json.bak-opencode-20261019` |
| `voice_agent/intent_config.json`（通用） | **1 行**：新增 `dictation_punct_to_space: true` | `intent_config.json.bak-opencode-20261019` |
| `default/vllm-asr`（k8s） | args 追加 `--hf-overrides={"architectures":["Qwen3ASRRealtimeGeneration"]}`（启用 `/v1/realtime`） | 原始 args 全量数组在 `k8s/rollback_args.json` |

新增环境变量（全部默认关，不影响原行为）：

| 变量 | 值 | 作用 |
|---|---|---|
| `OPENCODE_MIN_PREFIX` | `2` | 两字命令（AoE4 三农/补农）必须 2 |
| `OPENCODE_ASR_BIAS` | `1` | 用 profile 的 hotwords/bias（默认 0 = 原样禁用） |

> 周期抢跑 worker（`OPENCODE_PREEMPT_MS=400`）已按用户要求**回滚**：补丁 C2/G 已移除，
> `preempt.py` 已删除；`OPENCODE_PREEMPT_*` 不再生效。

其余既有文件（`actions.py` / `phon_match.py` / `run_src_voice_agent.cmd` / `intent_config.json` …）**未改动**。

## 回滚步骤（顺序不能反）

```powershell
cd "C:\Users\user\Desktop\ds harness\voice_agent"

# 1) 先还原 voice_agent.py（patch.py 在待删目录里，所以必须先跑）
E:\voice_agent_build\Scripts\python.exe -X utf8 -m opencode_fastpath.patch --rollback
#    可选校验: python -m opencode_fastpath.patch --status   （应输出 markers: 0）

# 1b) 还原 asr_vllm.py（只有 1 行改动）
Copy-Item ".\asr_vllm.py.bak-opencode-20261019" ".\asr_vllm.py" -Force

# 1b2) 还原 dlog.py（日志级别）与 typein.py（听写验证轮询）
if (Test-Path ".\dlog.py.bak-opencode-20261019")   { Copy-Item ".\dlog.py.bak-opencode-20261019"   ".\dlog.py"   -Force }
if (Test-Path ".\typein.py.bak-opencode-20261019") { Copy-Item ".\typein.py.bak-opencode-20261019" ".\typein.py" -Force }

# 1c) 还原 AoE4 配置（enable_dictation / dictation_punct_to_space）
Copy-Item ".\aoe4\profiles\aoe4-v1\intent_config.aoe4.json.bak-opencode-20261019" ".\aoe4\profiles\aoe4-v1\intent_config.aoe4.json" -Force
# 1d) 还原通用配置（dictation_punct_to_space 一行）
if (Test-Path ".\intent_config.json.bak-opencode-20261019") { Copy-Item ".\intent_config.json.bak-opencode-20261019" ".\intent_config.json" -Force }

# 2) 再删新增文件
Remove-Item -Recurse -Force ".\opencode_fastpath"
Remove-Item -Force ".\run_opencode_shadow.cmd",".\run_opencode_live.cmd",".\OPENCODE_DEPLOY_20261018.md"
```

**k8s（realtime 路由）单独回滚**（不跑上面两步也可以先做）：

```bash
# 在 WSL 里
kubectl patch deploy vllm-asr --type=json \
  --patch-file "/mnt/c/Users/user/Desktop/ds harness/voice_agent/opencode_fastpath/k8s/rollback_args.json"
kubectl rollout status deploy/vllm-asr --timeout=300s
```

⚠️ 若先删了 `opencode_fastpath/`，请从 `OPENCODE_MANIFEST.json` 或本说明重建 `rollback_args.json`；
更好的顺序是**先回滚 k8s，再删目录**。

如果不方便跑 Python，也可手工恢复：

```powershell
Copy-Item ".\voice_agent.py.bak-opencode-20261018" ".\voice_agent.py" -Force
# 然后删除上面列出的新增路径
```

## 运行时产物（可单独保留/删除）

`opencode_fastpath/logs/`、`run_opencode_shadow.log`、`run_opencode_live.log`、`c_stderr_opencode.log`。

## 校验

- `python -m opencode_fastpath.patch --status` → `markers: 2`（已部署）/ `0`（已回滚）
- `opencode_fastpath/OPENCODE_MANIFEST.json` 记录新增文件与备份文件的 sha256
