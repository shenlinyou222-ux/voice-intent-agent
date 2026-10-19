# 红警（OpenRA）语音实时控制 —— 设计与落地

> 目标（用户原话拆解）：
> 1. 把 `voice_agent` 这个程序 + 后台推理服务**接到红警的 API** 上；
> 2. **检查所有其暴露的、玩家一般调用的 API**，做统计和设计；
> 3. 实现 **语音 → 意图识别 → 实时控制**；
> 4. **热词、权重等用不同版本加以控制与区分**；原有意图识别可以完全丢掉，
>    游戏意图**专门作为新模型权重部署**；
> 5. 全程**只变两样**：模型相关热词 + 后面 embedding 的重训练；以及控制器把指令送进游戏。

**状态：已打通并实测通过（真游戏 + 真语音）。**

---

## 0. 结论速览

| 项 | 结果 |
|---|---|
| 玩家可控 API 清点 | Lua 脚本 API **209** 方法 / 指令(order) **46** 有实现者、**38** 被代码发出 / 聊天命令 **30** / Bot 模块 **42** / Trait **594**（`API_INVENTORY.md`） |
| 游戏侧新增 | `VoiceBridge` trait（1 个 C# 文件 + 4 个 mod 的 `World:` 注册），编译进 `E:\OpenRA\bin` |
| 控制通路 | 本地 TCP NDJSON → trait → `world.IssueOrder()` —— **与玩家点击完全同一条代码路径** |
| 语音侧新增 | `openra/` 包：客户端 / 版本化 profile / 20 个游戏动作 / 6 个工具 |
| 版本化 | profile 目录即版本；`hotwords_version` / `intent_version` / `index_version` 三套戳独立演进 |
| 游戏部署 | `intent_config.openra.json`：20 个游戏意图，桌面意图 **0** 个（按要求完全丢掉） |
| 真机实测 | 联调 **13/13 通过**：选中→移动→展开基地车→生产→放置，现金/电力真实变化 |
| 真语音实测 | SAPI 合成中文口令 → ASR → 意图 → 游戏执行：**端到端 186~260ms**，单位真的动了 |

---

## 1. 红警暴露了哪些「玩家一般会调用」的 API

清点脚本 `openra/tools/api_inventory.py`（只读源码），产物 `API_INVENTORY.md` / `api_inventory.json`。

| API 面 | 数量 | 说明 | 能否实时控制 |
|---|---|---|---|
| **Lua 脚本 API** | 209 方法 / 55 对象 | `Scripting/Global/*`（Trigger 37、Map 13、Media 13、Utils 12…）+ `Scripting/Properties/*`（Move/Attack/Harvest/Produce/Deploy…） | ❌ 只在**地图脚本**里跑，进程内无外部入口 |
| **指令 order** | 46 有实现者 / 38 被发出 | `Move` `Attack` `AttackMove` `Guard` `Stop` `Scatter` `Harvest` `DeployTransform` `Enter` `CaptureActor` `Sell` `RepairBuilding` `StartProduction` `CancelProduction` `PauseProduction` `SetUnitStance` `SetRallyPoint` `PlaceBuilding` … | ✅ **这才是玩家真正"下命令"的接口** |
| **聊天命令** | 30（`/help` `/pause` `/surrender` + DevCommands…） | `INotifyChat` → `IChatCommand` | ⚠️ 只有固定几个，不能控制部队 |
| **Bot/AI 模块** | 42 | `HarvesterBotModule` `SquadManagerBotModule`… 也是 `QueueOrder` | ✅ 现成范式：自动化 = 用 order 控制 |
| **Trait 面** | 594（Common 508 / Cnc 72 / D2k 14） | 模组可挂组件 | — |

**关键结论**：玩家"下命令"的唯一通用接口是 **order**，正常入口是
`World.IssueOrder()` → `OrderManager` → 网络帧 → `UnitOrders.ProcessOrder`
→ `ValidateOrder`（校验 `Subject.Owner.ClientIndex == 发出者 clientId`）→ `Actor.ResolveOrder`。
红警**没有**对外控制入口（无 HTTP/socket API），所以必须在游戏内加一个**入口 trait**
把外部指令翻成 order —— 这样同步、校验、规则全部沿用原逻辑，不绕过任何东西。

---

## 2. 控制通路

```
 麦克风 / wav
   │  VAD 分句
   ▼
 Qwen3-ASR 1.7B (vLLM, 172.20.71.101:30805)
   │   system 提示词 = 版本化热词 (hotwords.txt)          ← 只改这里(+索引)
   ▼
 意图判定：① 规则直通 profile.triggers（确定性，优先）
           ② 版本化 embedding kNN 索引 index.json (dmeta-small 768 维)   ← 只改这里
   │  (game_move, "60,40")
   ▼
 动作层 openra/game_actions.py（20 个 game_* 动作）
   │  NDJSON over TCP 127.0.0.1:38900
   ▼
 游戏内 VoiceBridge trait（C#，World trait）
   │  world.IssueOrder(new Order("Move", actor, Target.FromCell(...)))
   ▼
 OrderManager → 网络帧 → ValidateOrder → Actor.ResolveOrder → 单位动起来
   （与玩家点右键同一条路）
```

### 2.1 游戏侧：`VoiceBridge` trait

文件 `E:\OpenRA\OpenRA.Mods.Common\Traits\VoiceControl\VoiceBridge.cs`
注册 `mods/{ra,cnc,d2k,ts}/rules/world.yaml` 的 `World:` → `VoiceBridge:`

要点（每一条都是实测踩出来的）：
* `[TraitLocation(SystemActors.World)]`；`IWorldLoaded` 起监听，`ITick` 在主线程消费指令
  （游戏状态只能在主线程碰 → `ConcurrentQueue` + `ManualResetEventSlim` 同步请求/响应）
* 只控制 `world.LocalPlayer` 自己的单位；专用服务器（无本地玩家）自动禁用
* **每条连接一个线程**（早先一次只服务一个连接 → 第二个客户端排在 backlog 里没人理，客户端超时）
* 每 tick 最多 8 条；请求等待 15s（窗口失焦时主循环掉帧，3s 会偶发超时）
* **item 名必须验证 + 归一化**：`ProductionQueue.ResolveOrder` 里是 `rules.Actors[TargetString]`，
  名字不存在会抛异常**把整局游戏带崩**（实测踩过）；而且规则集里 actor 名是**小写**（`powr`/`e1`/`2tnk`），
  语音层大小写都可能发过来 → 统一转小写 + 给相近候选
* **没有显式选中时按"指令能力"兜底**（Harvest→只有矿车；Sell/Repair→只有建筑；
  Move/Attack→只有可移动单位）—— 早先直接回退"所有空闲单位"，于是「矿车去采矿」把指令发给了基地和电厂
* 执行后把中文回执打进游戏聊天框 + 雷达 ping（`EchoToChat`），玩家能看见"它听懂了什么"

### 2.2 协议（v1，NDJSON）

请求 `{"id":<int>,"cmd":"<name>","token":<str?>,"args":{...}}`
响应 `{"id":<int>,"ok":true,"data":{...}}` / `{"id":<int>,"ok":false,"err":"..."}`

| cmd | args | 翻成的 order / 作用 |
|---|---|---|
| `hello` | — | 探活（mod/玩家/clientIndex/tick） |
| `state` | — | 现金/电力/我方单位/视野内敌人/生产队列/选中 |
| `buildable` | — | **让游戏自己报**本局能造什么（避免名字猜错） |
| `deployable` | — | 当前单位能否原地展开；不能则给附近可展开格子 |
| `select` | `mode,filter,ids,x,y,r` | 选择（同时镜像到游戏内选择） |
| `move` / `attack` / `attackmove` / `guard` | `x,y` 或 `id` | `Move` / `Attack` / `AttackMove` / `Guard` |
| `stop` / `scatter` | — | `Stop` / `Scatter` |
| `stance` | `stance` | `SetUnitStance` |
| `harvest` / `deploy` / `sell` / `repair` | — | `Harvest` / `DeployTransform` / `Sell` / `RepairBuilding` |
| `enter` / `capture` | `id` | `EnterTransport` / `CaptureActor` |
| `build` / `cancel` / `pause` | `item,count,pause` | `Order.StartProduction/CancelProduction/PauseProduction` |
| `place` | `item,x,y,variant` | `PlaceBuilding`（放不下会在附近螺旋找位置） |
| `rally` / `ping` / `chat` | `x,y` / `text` | `SetRallyPoint` / 雷达 ping / `Order.Chat` |
| `raw` | `order,id,x,y,target_string,extra_data,queued` | 任意 order（逃生舱口，`target_string` 也做存在性校验） |

`select` filter：`all` / `idle` / `mobile` / `building` / `harvester` / `armed` /
`type:<actor>` / `queue:<Infantry|Vehicle|Building|Defense|Ship|Aircraft>` / `near:x,y,r`。

---

## 3. 版本化设计（用户要求 4）

三样东西**各自独立版本化**，目录即版本：

```
openra/profiles/openra-v1/
    profile.json                 总配置 + 三套版本戳 + 别名表 + 单位表
    hotwords.txt                 热词（ASR system 提示词 + 纠错候选）→ hotwords_version
    intents.jsonl                意图语料（161 条示例 / 20 意图）    → intent_version
    index.json                   embedding kNN 索引（768 维）        → index_version
    intent_config.openra.json    游戏部署专用配置（自动生成）
```

| 版本戳 | 由什么决定 | 变了会怎样 |
|---|---|---|
| `hotwords_version` | 热词表内容 SHA256 前 12 位 | 只影响 ASR 偏置，**不碰权重**，重启即生效 |
| `intent_version` | 意图语料内容 SHA256 | 语料变了 → 必须重建索引 |
| `index_version` | 语料条数 × 维度 + intent 哈希 | 意图分类用的向量索引版本 |

* 换域/换版本 = **换一个 `profiles/<id>/` 目录**；旧版本原样留着，可 A/B、可秒级回滚
* 三套戳打进日志，可用 `python openra/game_profile.py` / `actions.game_status()` 查
* "原有意图识别可以完全丢掉" → 游戏配置里 `apps/urls/files` 全空、`intents` 全是 `game_*`
* "游戏意图专门作为新模型权重部署" → 意图后端指向**游戏专用索引**
  （`intent_index: openra/profiles/openra-v1/index.json`），与桌面版 `intent_index.json` 互不干扰

**为什么"只变热词 + embedding 重训"就够**：A/B 主干（Qwen3-ASR audio_tower + 语义编码器）全程冻结，
意图是**向量近邻分类**（kNN-NLU），换域 = 换语料重建索引（秒级，`build_profile_index.py`），
不需要反向传播 —— 与仓库里 `intent_embed.py` 的设计一致。

---

## 4. 语音 → 意图 → 指令（20 个意图，规则直通实测样例）

| 说法 | 意图 | 参数 | 桥指令 |
|---|---|---|---|
| 选中所有坦克 | `game_select` | `queue:Vehicle` | `select{mode:set,filter:queue:Vehicle}` |
| 移动到 60 40 / 移动到六十 四十 | `game_move` | `60,40` | `move{x:60,y:40}` |
| 攻击最近的敌人 | `game_attack` | `nearest` | `state` → 最近敌人 → `attack{x,y}` |
| 攻击前进到 70 70 | `game_attackmove` | `70,70` | `attackmove{x,y}` |
| 造两个矿车 | `game_build` | `HARVx2` | `build{item:HARV,count:2}` |
| 放置电厂 | `game_place` | `auto` | `place{item:powr,x,y}` |
| 取消建造坦克 | `game_cancel` | `2TNK` | `cancel{item:2TNK,count:1}` |
| 姿态防御 | `game_stance` | `Defend` | `stance{stance:Defend}` |
| 矿车去采矿 | `game_harvest` | — | `harvest`（只发给真有矿车的） |
| 基地车展开 | `game_deploy` | — | `deployable` → 必要时先挪位 → `deploy` |
| 设置集结点 40 40 | `game_rally` | `40,40` | `rally{x,y}` |
| 现在有多少钱 | `game_state` | — | `state` → 回中文句子（可接 TTS 播报） |
| 喊话进攻 | `game_chat` | `进攻` | `chat{text}` |

单位别名表（59 条，中/英/actor 三向）：矿车→`HARV`、重坦→`3TNK`、猛犸→`4TNK`、电厂→`POWR`、
战车工厂→`WEAP`、谭雅→`E7`、雌鹿→`HIND` …（`profile.json: items`）

**中文数字解析**（语音报坐标几乎必然用中文）：`六十 四十`→(60,40)、`三十 二十`→(30,20)、
`十五`→15、`一百 五十`→(100,50)，并且能处理 ASR 把停顿吃掉的情况（`六十四十`→(60,40)）。

---

## 5. 文件清单

**游戏侧（E:\OpenRA）**
```
OpenRA.Mods.Common/Traits/VoiceControl/VoiceBridge.cs    新增 trait（入口 + 协议 + order 翻译 + 校验）
mods/ra|cnc|d2k|ts/rules/world.yaml                      World: 节点注册 VoiceBridge
```

**语音侧（voice_agent/openra/）**
```
bridge_client.py                 客户端（stdlib、自动重连、探活）
game_profile.py                  版本化 profile 运行时：热词/别名/中文数字/规则直通
game_actions.py                  20 个动作（注册进 actions.ACTIONS）
profiles/openra-v1/*             一整套版本化产物
tools/api_inventory.py           OpenRA 玩家 API 清点
tools/make_profile.py            生成 profile（内容变了就出新版本）
tools/build_profile_index.py     重建 embedding 索引（= 换域"重训练"）
tools/make_game_config.py        生成游戏部署专用 intent_config
tools/selftest_bridge.py         假游戏协议/动作自测（23 项）
tools/live_test.py               真游戏联调（13 项）
test_wavs/*.wav                  SAPI 合成的中文口令（语音闭环测试用）
API_INVENTORY.md / DESIGN.md     报告与本文档
```

**对既有代码的最小改动（可回退）**
```
actions.py       hotword_context() 增加游戏模式分支；parse() 前置 profile 规则直通；
                 resolve_closed() 放行 game_*；末尾注册 GAME_ACTIONS + 三个辅助函数
voice_agent.py   规则直通链里加 game_* 分支（① 取参数 ② 判可执行）
```

---

## 6. 怎么跑

```powershell
# ① 编译游戏（带 VoiceBridge）
cd E:\OpenRA
$env:PATH = 'E:\dotnet10;' + $env:PATH        # 需要 .NET 10 SDK（装在 E:\dotnet10）
dotnet build -c Release -p:TargetPlatform=win-x64

# ② 起游戏（直接进地图，不用点菜单）
dotnet bin\OpenRA.dll -- Engine.EngineDir=".." Game.Mod=ra Launch.Map=a-nuclear-winter

# ③ 探活 / 联调
cd "C:\Users\user\Desktop\ds harness\voice_agent"
python openra\bridge_client.py                 # 打印 hello
python openra\tools\live_test.py               # 真机联调 13 项
python openra\tools\selftest_bridge.py         # 假游戏 23 项（不需要开游戏）

# ④ 游戏模式启动语音 agent（只加载游戏意图）
python voice_agent.py --config "openra\profiles\openra-v1\intent_config.openra.json"
#    想先用录音文件验证：
python voice_agent.py --config "openra\profiles\openra-v1\intent_config.openra.json" --wav openra\test_wavs\game_place.wav
```

改内容 → 出新版本：
```powershell
python openra\tools\make_profile.py --id openra-v2          # 改 tools/make_profile.py 后
python openra\tools\build_profile_index.py --id openra-v2   # 重建索引（秒级）
python openra\tools\make_game_config.py --id openra-v2      # 生成对应配置
```

---

## 7. 实测证据

| 验证 | 方法 | 结果 |
|---|---|---|
| 游戏编译 | `dotnet build -c Release -p:TargetPlatform=win-x64` | ✅ 0 警告 0 错误 |
| trait 被 mod 识别 | `OpenRA.Utility.dll {ra,cnc,d2k,ts} --check-yaml` | ✅ 4/4 exit=0 |
| 假游戏协议自测 | `python openra\tools\selftest_bridge.py` | ✅ **23/23**（含断线重连、未知指令、智能展开、放置） |
| 规则直通全链路 | `python openra\game_actions.py`（dry-run） | ✅ 17/17 |
| embedding 索引 | `build_profile_index.py`（真调 dmeta-small） | ✅ 161×768，留一自检 77.6% |
| **真机联调** | `python openra\tools\live_test.py` | ✅ **13/13** |
| **真语音闭环** | SAPI 中文口令 → `voice_agent.py --wav` | ✅ 见下 |

真机联调细节（游戏：Red Alert / A Nuclear Winter，玩家 Commander）：
```
[1] select all → 选中 mcv            ✓
[1] move      → (90,24) → (88,24) 真的动了 ✓
[2] deploy    → mcv 变成 fact（建造厂）  ✓
[3] buildable → Building: [powr] Defense: [fenc, brik] ✓（游戏自报）
[4] build powr → 生产队列 remaining 180→0 → ready ✓
[4] place powr → 电厂落到 (90,26)，现金 5000→4700，电力 0→100 ✓
[5] ping / chat → 雷达标记 + 游戏内聊天 ✓
[6] 非法条目名 → 干净报错（不再崩游戏）✓
```

真语音闭环（SAPI 合成 → ASR → 意图 → 游戏）：
```
「建造电厂。」   → game_build  POWRx1  → 生产 发电厂 x1        ⇒ 220ms（ASR 163）
「放置电厂。」   → game_place  auto    → 放置 powr 到 92,27    ⇒ 232ms
「现在有多少钱？」→ game_state         → 现金 4400，电力 200/0，单位 3 个 ⇒ 212ms
「矿车去采矿。」  → game_harvest       → 0 辆矿车（没矿车就不乱发）      ⇒ 182ms
「移动到六十，四十」→ game_move 60,40   → mcv (90,24)→(59,40) 真的走过去  ⇒ 186ms
```

---

## 8. 已知限制 / 下一步

1. **坐标的 ASR 容错**：说「六十 四十」被转成「六十四十」已能解析（见 §4）；
   但「四十三时」这种丢字仍只能判成单个数 → 动作会明确回「移动需要坐标」而不是乱动（这是对的）。
2. **TTS 回执**：`game_state` 已返回中文句子，接 `eva-voice-broadcast` 即可语音播报战况。
3. **"这里/这边"指代**：没有屏幕坐标 → 用选中部队重心兜底；真正的"鼠标指哪打哪"要再加屏幕坐标通道。
4. **多人局**：桥只控制本机玩家（`world.LocalPlayer`），符合预期。
5. **意图索引留一自检 77.6%**：`攻击/攻击前进/移动` 天然近义 → 靠规则直通兜底；
   提精度只需加语料/触发词再重建索引（不改代码、不动权重）。
6. **部署到集群**：profile 目录 + 索引可以随 `voice_agent` 一起发到 Jenkins/k8s，
   版本戳保证"装的是哪一版热词/权重"可追溯。
