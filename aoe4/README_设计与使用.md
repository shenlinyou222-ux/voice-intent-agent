# 语音意图识别 · 意图域 aoe4 —— 设计与使用（现状版）

> 阶段状态：**闭集跑通** ✓。后续计划：专门剪枝/蒸馏一个闭集小模型（ASR 430ms → 目标 50~100ms）。
> 链路：常开监听 → Silero VAD 分段 → 进程内 ASR → 闭集对齐 → 写命令文件 → 游戏内 SCAR 脚本执行。

---

## 1. 当前架构（唯一路径，没有 A/B 双路）

```
麦克风(设备1, 16kHz)
  → 【AGC 前置】= Silero 引擎自带 PreGain（10s 窗口 95 分位自适应，非手调）
  → 【Silero VAD 分段】= 分段器本体（概率判据，自带 640ms 环形预滚 pre_buffer）
  → 段
  → 【静音裁剪 + 动态预滚上限】头上限 = min(640ms, 段总长/5)，两侧留 100ms
  → 【队列】串行 worker（防丢段：VAD 串行，处理上一句时下一句会排队而不是被丢）
  → 【ASR = vLLM(GPU)】集群 vllm-asr:30805（官方 Qwen3-ASR-0.6B **bf16 原版**，CUDA Graph）
  → 【闭集强制对齐】x农 + 谐音归一（三龙→三农）
  → 写命令文件 vb_cmd_*.txt
  → 游戏内 SCAR 脚本 0.1s 轮询 → 城镇中心训练 N 个农民
```

**ASR 后端演进（实测对比，同一段 1.05s「三农」录音）**：

| 方案 | 延迟 | 转写 | 原因 |
|---|---|---|---|
| CPU 进程内 llama.cpp 1.7B-Q4 | 430 ms | 「三农」✓ | 无 CUDA Graph ✗ |
| transformers GPU 0.6B（逐 token） | 227 ms | 「三龙」✗ | 无图加速 ✗ |
| GGUF 0.6B-Q4 进程内 | ~230 ms | 「三龙」✗ | 量化损失 ✗ |
| **vLLM GPU 0.6B-bf16** ✓✓ | **131 ms** | 「**三农**」✓✓ | **官方权重 + piecewise CUDA Graph** |

⚠️ **vLLM 接口与 llama.cpp 不同**（`85-vllm-asr.yaml` 里明确警告过）：
- vLLM **没有** `/v1/audio/transcriptions` ✗
- 走 **chat/completions 多模态输入**：`{"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"<base64 wav>","format":"wav"}}]}]}`
- 客户端：`voice_agent/asr_vllm.py`（`asr_backend=vllm`）

**部署要点**（`E:\habitica-setup\k8s\live\85-vllm-asr.yaml`）：
- `--cudagraph-capture-sizes=64` ★ **只抓 1 个桶**（默认 51 个会卡在 Capturing 0/51 → startupProbe 超时 → SIGKILL 137 → CrashLoopBackOff）
- `--gpu-memory-utilization=0.55`（0.30 时权重就吃掉 1.82 GiB → encoder cache/KV 不够 → 被 OOM 杀，无 traceback）
- 更新部署：`kubectl apply -f k8s/live/85-vllm-asr.yaml`

**已废弃/删除**（别再往回加）：
- ✗ HTTP ASR（`llama-server:8124`）—— 多一跳，且那个 server 是**纯 CPU 构建**（无 `ggml-cuda*.dll`，`-ngl` 被忽略）
- ✗ 自研机械切块（`stream_P_s/stream_k/stream_thr/stream_chunk_*`）—— **VAD 本身就是分段器**
- ✗ 自研环形预滚（`stream_preroll_s`）—— 引擎自带 640ms `pre_buffer`，叠加是重复
- ✗ 能量 VAD（`vad.py`）—— Silero 概率判据更稳（能量 VAD 的参数仍在配置里但**不生效**）
- ✗ SSE→WS 适配器（`aoe4/asr_stream_adapter.py`）—— 架构错误产物，未使用

---

## 2. 延迟账（实测，毫秒级）

| 环节 | 时长 | 说明 |
|---|---|---|
| 检测（说完 → VAD 判定关闭） | **~150ms** | = `required_misses×30`(90) + `smoothing×30×0.5`(30) + 块对齐(≤30) |
| ASR（**vLLM GPU 0.6B-bf16**） | **~131ms**（1.05s 音频） | CUDA Graph + 官方权重；CPU 版要 430ms |
| 意图（正则短路） | **~0ms** | 闭集命中直接返回，不打 embedding |
| 动作（写命令文件） | **~2.6ms** | 进程检查缓存已**启动预热**（否则首次 tasklist 145ms ✗） |
| 游戏侧轮询 | **≤100ms** | 脚本 0.1s 轮询 |
| **合计** | **≈0.38s** ✓ | 检测150 + ASR131 + 意图0 + 动作3 + 轮询≤100 |

**关键结论**：
- ASR 时间 ≈ **固定开销**（预填+首token）×模型规模 + **输出 token 数** × 80ms
- **"三农"两个字真的有 0.88s 语音**（起音辅音+字音+收尾拖音，用录音实测过）
- **前缀复用基本无效**：音频在 chat 模板**中间**（`<|im_start|>user<|audio_start|>[音频]<|audio_end|><|im_end|><|im_start|>assistant`），音频一变后面全作废 → 只能复用开头 ~5 token
- **GPU 对这个 ASR 没有优势**（实测 227 vs 228ms；瓶颈是音频编解码+逐 token 前向，不是算力）

---

## 3. VAD 参数（Silero，全部生效 ✓）

| 参数 | 值 | 含义 |
|---|---|---|
| `vad_prob_threshold` | 0.4 | 平滑后语音概率阈值 |
| `vad_db_threshold` | **0** | **必须 0** ✗ —— 硬电平门会把 prob=0.999 的轻语音整段拦掉（血债） |
| `vad_required_hits` | **2** | 连续 2 次命中 → 开始计时（3 太严，短句概率抖 → 整句丢 ✗） |
| `vad_required_misses` | 3 | 连续 3 次未命中 → 收段（≈90ms，端点主旋钮） |
| `vad_smoothing_window` | 2 | 平滑窗（5 时拖尾 ~75ms ✗） |
| `vad_max_speech_ms` | 6000 | 单段最长 |
| `vad_target_rms` / `vad_max_gain` | 0.03 / 60 | 自适应预增益目标与上限 |
| 预滚 | **动态 min(640ms, 总长/5)** | 段 = [640ms 预滚]+[语音]+[尾静音]；头部保留 = min(总长/5, 语音起点−100ms)（两者取小 ✓ 安全阀）；阈值必须**相对峰值**（AGC 抬高底噪 → 绝对阈值失效 ✗） |

> ⚠️ 配置里 `_VAD_NOTE` 键写明了**哪些参数属于 Silero 路（生效）**、**哪些属于能量 VAD 分支（不生效）** —— 防止再踩"改了不生效"的坑（`smoothing_window` 曾经就没被工厂转发 ✗）。

---

## 4. 闭集（意图层）

```
闭集：<数字词>农     数字词 ∈ {一,二,两,三,四,五,六,七,八,九,十} ∪ 阿拉伯数字
输出：aoe4_train_villager(n) → 城镇中心训练 n 个农民
其它一律拒识（动词式/长句/闲聊）
```
**三层对齐**（`aoe4/game_profile.py`）：
1. **谐音/异体字归一化**：二龙→二农、仨农→三农、五浓→五农、衣农→一农、酒农→九农 …
2. **句尾锚定正则**：整句 ≈ `<数字>农`（允许 1 个语气尾巴）→ 长句里的不认
3. **音节模糊兜底**：只看句尾恰好两字，分数 <0.5 拒识

自测 **35/35 通过** ✓（含 12 个应拒识用例）

---

## 5. 游戏侧（mod）

- mod：`…\Documents\My Games\Age of Empires IV\mods\extension\local\voicecontroller\voicecontroller.sga`（编辑器 F7 烧录）
- 挂载：`data+1:` 覆盖层
- 脚本：`voicecontroller.scar`，0.1s 轮询 12 个槽位 + 主通道
- 动作：`LocalCommand_EntityBuildSquad(pid, 城镇中心EGroup, scar_villager蓝图)`（一次调用=一个单位）
- 反馈：`UI_CreateEventCue` 中文事件提示

**血债（脚本侧）**：
1. `UI_SystemMessageShow(裸文本)` **直接崩游戏** → 只用 `UI_CreateEventCue`
2. `Rule_Add*` 的**回调必须是具名全局函数**（匿名会 `FATAL SCAR ERROR` 并暂停整个 SCAR 系统）
3. 引擎**按路径缓存**文件读取 → 多槽轮转（`vb_cmd_1..12.txt` + `#序号`）
4. 脚本内 `print` **不进** `warnings.log`；SCAR 报错**会进**（带行号）
5. `Misc_WriteFile` 游戏内**写不进去** → 桥是**单向**的（外部写、游戏读）
6. 命令解析抓**末尾数字**（`(%d+)%s*$`），别用 `^(%a+)%s*(%d*)`

---

## 6. 日志（毫秒级，干净）

```
[hh:mm:ss.mmm] MIC  ⚡分段已启用：Silero VAD 概率判据（prob 0.4 / hits 2 / misses 3 / 平滑 2）…
[hh:mm:ss.mmm] MIC  VAD 段 1.70s → 入队（队列 1）
[hh:mm:ss.mmm] ASR  裁剪静音 0.16s（头上限 min(640ms,总长/5)=339ms）→ 送 ASR 1.54s
[hh:mm:ss.mmm] ASR  VAD 段转写 1.54s → 「三农。」  430ms
[hh:mm:ss.mmm] INT  意图=aoe4_train_villager  conf=1.000  参数='3'  0ms
[hh:mm:ss.mmm] ACT  aoe4_train_villager('3') -> 训练农民 × 3（已发送 3 条）
[hh:mm:ss.mmm] MIC  ⇒ 说完→游戏生效 680ms（检测 … + ASR … + 意图与动作 …；游戏侧轮询 ≤100ms）
```

**红字（NativeCommandError）已消除** ✓：llama.cpp 把信息日志写 stderr，PowerShell 把任何 stderr 当错误 → 启动器改用 `cmd /c "… 2>&1"` 在**进程层**合并 ✓

---

## 7. 启动与测试

```
桌面快捷方式「语音意图识别」
  → aoe4\run_voice_aoe4.cmd → run_voice_aoe4.ps1
     （PYTHONWARNINGS=ignore 屏蔽第三方 UserWarning；cmd 层合并 stderr 消红字）
游戏：单人 → 遭遇战 → 游戏模式 voicecontroller → 开始
说话：「三农」「二农」…（常开监听，不用按键）
```

**不用麦克风测**：`python voice_agent.py --config …\intent_config.aoe4.json --wav <16k单声道wav>`

---

## 8. 后续（用户计划）

1. **闭集剪枝 / 蒸馏**一个小 ASR（目标 ASR 430ms → 50~100ms）
2. 可能扩闭集（更多单位/建筑）→ 加意图 + 语料 + 重建索引（`aoe4/tools/build_index.py`）
3. 流式抢跑**已放弃**（VAD 判定整段一次请求更准；若将来要抢跑，需重新设计且要接受模型争用）

---

## 9. 扩展设计：部队编队 + 动作树（调研结论，2026-10-16）

> 目标：语音「1队冲锋」「1队去XX地」→ 部队到位后**自动按动作树作战**（自动找敌、远程拉扯、近战顶上）。
> 出处：`<游戏>\scardocs\api\Essence_ScarFunctions.api`（官方函数表）+ `Essence_Constants.api`（枚举）
> + `E:\aoe4-modding\attrib\ai\`（数据模板）+ `AOE4-TSTL\types\*.d.ts`（类型）。

### 9.1 命令 API（`LocalCommand_*`，官方文档收录 ✓）

| 动作 | 函数 | 备注 |
|---|---|---|
| 移动到坐标 | `LocalCommand_SquadMovePos(PID,SGID,Pos,queued,reverseMove,split,prox)` | API:1058 |
| **攻击移动** | `LocalCommand_SquadAttackMovePos(...,SCMD_AttackMove,Pos,planName,queued,split)` | API:1050 |
| 攻击目标 | `LocalCommand_SquadEntityAttack(PID,SGID,EGID,bCheckFOW,bStationary,planName,queued)` | API:1053 |
| 攻击地点 | `LocalCommand_SquadPositionAttack(...)` / `SquadSquadAttack` | API:1065 / 1071 |
| 巡逻 | `LocalCommand_SquadPath(...LoopType,bAttackMove,pauseTime...)` | API:1061 |
| 撤退 | `LocalCommand_SquadRetreatPos(PID,SGID,Pos,queued,...)` | API:1066 |
| 驻守建筑 | `LocalCommand_SquadEntityLoad(...,SCMD_Load,...)` | API:1056 |
| 停止 | `LocalCommand_Squad(...,SCMD_Stop,...)` | API:1048 |
| **姿态** | `LocalCommand_SquadSetCombatStance(PID,SGID,StanceType,queued)` | API:1067 |
| 放技能 | `LocalCommand_SquadAbility(PID,SGID,ScarAbilityPBG,...)` | API:1049 |

姿态枚举（**只有 3 种** ✗ 没有 aggressive/defensive）：`STANCE_Attack` / `STANCE_CeaseFire` / `STANCE_StandGround`（CONST:792-795）

### 9.2 编队 / 单位查询

- **「1队」= `Misc_GetControlGroupContents(index, sgroup, egroup)`**（API:1141，index 0..9 → "1队"=0）
- 配套：`Misc_GetSquadControlGroup` / `Misc_SetSquadControlGroup` / `UI_SetControlGroupSelectedCallback`（捕获玩家按热键选队）
- 遍历：`SGroup_ForEachEx(SGID, fn, spawned, despawned)`（API:1443）
- 近战/远程：`Squad_IsOfType(squad, "melee"/"ranged")`（API:1638；类型表 `attrib\type_squad_type.json`）
- 坐标：`Squad_GetPosition`（API:1572）；到位判定：`SGroup_HasDestination`（API:1605）
- **找最近敌人：无专用函数** ✗ → `World_GetSquadsNearPoint(PID,SGID,Pos,radius,OwnerType)`（API:1952）
  + `Player_ObserveRelationship`（`R_ENEMY`）+ `World_DistanceSGroupToPoint`（API:1906）自己排序

### 9.3 地图标注（地名词表来源 ✓✓）

```
Marker_FromName / AllMarkers_FromName   ← 按名字取标记
Marker_GetPosition                      ← 取坐标 → 直接喂 LocalCommand_SquadMovePos
Marker_GetStringAttribute / GetNumberAttribute   ← 标记可带自定义属性（资源点/要道/敌基地）
Marker_InProximity / GetProximityRadius  ← 邻近判定（"到位"）
Marker_Create / Destroy / SetVisible     ← 没有标注就自己标
Marker_StartAction / StopAction          ← 标记能触发动作
```
**结论**：地名词表**优先读地图自带标注** ✓；没有就用 `Marker_Create` 自建 ✓

### 9.4 动作树（两条路线）

**路线 A —— 纯 SCAR 自写 tick（兜底 ✓ 立刻可做）**
```
Rule_AddInterval(小队动作树_Tick, 0.3)          -- 必须具名全局函数（血债）
  ① Misc_GetControlGroupContents(队号, sg, eg)
  ② SGroup_HasDestination(sg) == false → 已到位 → 进入动作树
  ③ World_GetSquadsNearPoint 找敌 + Player_ObserveRelationship 筛 R_ENEMY
  ④ Squad_IsOfType 分近战/远程
     远程 → LocalCommand_SquadMovePos（后撤到 keepRange）+ SquadSetCombatStance(STANCE_Attack)
     近战 → LocalCommand_SquadEntityAttack（顶上）
  ⑤ 血线 < retreatHp → LocalCommand_SquadRetreatPos
```
风险 ✗：每 0.3s 下命令会**持续打断玩家手动指令**（SCAR **无命令优先级仲裁**，未找到"软指令"API）

**路线 B —— 复用游戏 AI（"高级模式"，需实测 ⚠️）**
```
AI_CreateEncounter(PID, encounterType) → AIEncounter_SetDebugName
AIEncounter_TargetGuidance_SetTargetPosition / SetTargetArea / SetTargetLeash
AIEncounter_ResourceGuidance_SquadGroup(enc, sg_team1)          -- 把 1 队交进去
AIEncounter_FormationTaskStateGuidance_SimpleMelee(enc, prio, minSq, maxSq, maxTasks, "melee", prioPBG)
AIEncounter_FormationTaskStateGuidance_MinRange(enc, ..., repositionIntervalSecs, useTactics, "ranged", prioPBG)  ← ★ 拉扯模板
AIEncounter_FallbackGuidance_SetRetreatHealthPercentage(enc, 0.35)
AIEncounter_Trigger(enc)     -- 结束：ForceComplete / Cancel
```
风险 ✗：这些 API 原本服务 skirmish AI，**能否接管人类玩家的 squad 文档未说明** ⚠️（但 `Squad_IsInAIEncounter` API:1631 说明机制上允许）→ **必须实测**
**成就 / 多人模式影响：文档完全没提** ✗（未找到）

### 9.5 既定动作树数据模板（确实存在 ✓，用户猜测正确）

```
attrib\ai\ai_statemodel_tunings\skirmish_pinged_attack.json     ← ★「点地图 → 进攻」现成模板
attrib\ai\ai_statemodel_tunings\skirmish_pinged_defend.json     ← ★「点地图 → 防守」
attrib\ai\ai_formation_coordinator\ai_formation_coordinator_basic.json   （15KB 顶层协调器）
attrib\ai\ai_formation_target_priority\archer_target_priority.json
      └─ time_to_target_scale / time_to_kill_scale / movement_speed_scale / exclude_unit_types
attrib\ai\ai_settings\default_skirmish_standard.json
```
**"远程自动拉扯"不是技能、不是姿态，而是 `TACTIC_MinRange` + encounter 数据** ✓
`AITacticType` 枚举含 `TACTIC_MinRange` / `TACTIC_Hold` / `TACTIC_ForceAttack` / `TACTIC_Avoid` …

### 9.6 需要新增的数据文件（纯 Lua 表，放 `assets\scar\winconditions\`，避开 sga 重打包）

```
vb_groups.lua     语音队号 ↔ control group index（"全军" = 遍历 0-9 求并集）
vb_landmarks.lua  地点名 ↔ MarkerID / 坐标（优先读地图标注 ✓）
vb_doctrines.lua  动作树模板 {ranged={mode="kite",keepRange=8,retreatHp=0.35}, melee={mode="charge"}}
```
路线 B 的 `targetPriorityPBG` 可直接复用 attrib 的 `*_target_priority.json` ✓

### 9.7 实施顺序

1. **第一步（立刻可做 ✓ 不依赖 AI 接管）**：「1队去XX」+「1队冲锋」+「1队固守/停火/撤退/驻守」
2. **第二步（需实测 ⚠️）**：`AI_CreateEncounter` + `FormationTaskStateGuidance_MinRange/SimpleMelee` → 到位后交给游戏 AI

### 9.8 未找到项（明确记录，避免重复挖）

- 「找最近敌人」专用函数 ✗
- `AI_CreateEncounter` 的 `encounterType` 整数枚举 ✗（`attrib\ai_military_engagement_type.json` 是 engagement 类型，不是它）
- SCAR 命令优先级 / "软指令" API ✗
- 成就 / 多人模式影响说明 ✗
- `Cmd_*`（40+ 高层封装，`dump.txt:31912` 实测存在）的定义源文件 ✗（在 `Data.sga` 内未解包）→ **建议优先用官方 `LocalCommand_*`**

