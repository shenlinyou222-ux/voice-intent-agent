# 红警(OpenRA) AI 的指挥/攻击机制分析 + 我们的战术层方案

> 结论先说：**引擎里的 AI 不是行为树（BT）**，而是
> 「**并发 BotModule + 小队(Squad) 分层状态机 + 模糊/效用决策 + 编组订单(groupedActors)**」。
> 下面每一条都有源码出处；再给出我们语音战术层的同构方案。

---

## 一、引擎 AI 是怎么指挥攻击的（源码实证）

### 1.1 三层结构

```
① 决策层：玩家 actor 上挂一堆 BotModule（都是 IBotTick，各自节流，互相独立）
   SquadManager / UnitBuilder / BaseBuilder / Harvester / SupportPower /
   CaptureManager / McvManager / BuildingRepair / PowerDown / ResourceMap
        │
② 组织层：SquadManagerBotModule 把"闲置单位"收进 Squad（小队）
   SquadType = Assault | Air | Rush | Protection | Naval
   Squad = { Units, Target/TargetActor, FuzzyStateMachine }
        │
③ 执行层：状态机 → 一次性把命令下给【整队】
   bot.QueueOrder(new Order("AttackMove", null, target, false, groupedActors: units))
```

**关键点：命令是"编组订单"**（`groupedActors`）——一条 order 带一串单位，引擎内部按单位展开。
这是 BT 里没有的语义（BT 通常一个实体一棵树）。

### 1.2 调度节奏（源码常量，`SquadManagerBotModuleInfo`）

| 参数 | 值 | 含义 |
|---|---|---|
| `SquadSize` | 8（+随机 0~30） | 攒够多少单位才组进攻小队 |
| `AssignRolesInterval` | 50 tick = **2 s** | 给闲置单位分配角色的间隔 |
| `AttackForceInterval` | 75 tick = **3 s** | 小队状态更新间隔（并**分摊到多个 tick**跑：`Exts.IntegerDivisionRoundingAwayFromZero`，注释标了 PERF） |
| `RushInterval` | 600 tick = **24 s** | 尝试速攻的间隔 |
| `IdleScanRadius` | 10 格 | 空闲时扫敌人的半径 |
| `AttackScanRadius` | 12 格 | 进攻中重新找目标的半径 |
| `DangerScanRadius` | 10 格 | 判断"要不要跑"的半径 |
| `ProtectionScanRadius` | 8 格 | 护家小队的扫描半径 |

### 1.3 状态机（不是 BT）

`StateMachine.cs` 只有 34 行：

```csharp
interface IState { void Activate(Squad); void Tick(Squad); void Deactivate(Squad); }
class StateMachine { IState currentState;
  public void Update(Squad s) => currentState?.Tick(s);
  public void ChangeState(Squad s, IState n) { currentState?.Deactivate(s); ... currentState?.Activate(s); } }
```

地面小队四态（`GroundStates.cs`）：

| 状态 | 干什么 | 转移 |
|---|---|---|
| `GroundUnitsIdleState` | 校验目标 → 找最近敌人 → 扫 `IdleScanRadius` | `AttackOrFleeFuzzy.CanAttack()` 成立 → 发**整队 AttackMove** 转 AttackMove；否则 → Flee |
| `GroundUnitsAttackMoveState` | 跟踪领队位置/目标变化；掉队的重整（见下）；进 `AttackScanRadius` 就转 Attack | 63 tick(2.5 s) 没进展 → 回 Idle（防卡死的 HACK）；`ShouldFlee` → Flee |
| `GroundUnitsAttackState` | 对每个"没在打"的单位发 `AttackMove` 打目标 | 同上 |
| `GroundUnitsFleeState` | 所有单位 Move 到**随机一个自家建筑** | 立刻回 Idle；`Deactivate` 时注销小队 |

另外还有 `ProtectionStates`（Idle/Attack/Flee）、`AirStates`（Idle/Attack/Flee）。

**领队选举**（`GroundStateBase.Leader/NewLeader`）：挑 locomotor 地形通过性最"挑"（`TerrainSpeeds.Count` 最少）的单位里、离队形质心最近的 —— 目的是让跟随者跟得上。
**重整**：掉出 `WDist.FromCells(单位数)/3` 的单位 → 领队 `Stop` + 其余 `AttackMove` 到领队格。
**卡死保护**：63 tick 无进展回 Idle（注释明说是为了避免每 tick 都做昂贵寻路）。

### 1.4 决策是**模糊推理**，不是 if-else / BT 条件节点

`AttackOrFleeFuzzy.cs`（依赖 NuGet `OpenRA-FuzzyLogicLibrary`）用 **Mamdani 模糊系统**：

- 4 个输入：`OwnHealth`、`EnemyHealth`、`RelativeAttackPower`、`RelativeSpeed`（都归一化到 0~100/1000）
- 输出：`AttackOrFlee`，判据 `CanAttack = attackChance < 30`（输出是"逃"的隶属度，所以小于 30 才是进攻）
- 规则按自己血量分三组（Normal / Injured / NearDead），共 ~30 条，例如：
  `if (OwnHealth is Normal) and (EnemyHealth is NearDead|Injured|Normal) and (RelativeAttackPower is Weak|Equal|Strong) ... then AttackOrFlee is Attack`
- 隶属度用梯形函数，如 `NearDead = Trapezoid(0,0,20,40)`、`Injured = (30,50,50,70)`、`Normal = (50,80,100,100)`

**火力/血量/速度怎么算**（这部分最值得抄）：

| 指标 | 算法 |
|---|---|
| 血量 | `ΣHP / ΣMaxHP × 100` |
| 火力 | 对每个 `Armament`：`Σ(DamageWarhead.Damage × Burst / (ReloadDelay + BurstDelays[0]×(Burst-1)).Clamp(1,200) × 100)`，我方合计 ÷ 敌方合计 × 100 |
| 速度 | `Mobile.Info.Speed` 的均值之比 × 100 |
| 敌我比 | 统一 `relative = f(own)/f(enemy) × normalize`，clamp 到 0~999 |

### 1.5 其它模块用的是别的范式（也值得知道）

- **支援技能** `SupportPowerDecision`：**效用评分**（utility）——`Considerations[]` 每条有 `Attractiveness=100`、`TargetMetric = Health|Value|None`（按血量/价值加权）、`CheckRadius = 5c`，总分 ≥ `MinimumAttractiveness` 才用；扫描 `CoarseScanRadius=20` → `FineScanRadius=2` 粗到细。
- **造建筑** `BaseBuilderQueueManager`：队列 + **放置点搜索**（`world.CanPlaceBuilding` + 螺旋/评分找位）。
- **采矿** `HarvesterBotModule`：空闲矿车扫描（50 / 433 tick 两个间隔）+ 资源图分块（`ResourceMapBotModule` 把地图切成指标块，算"缺多少矿车/有没有敌人"）。
- **造兵** `UnitBuilderBotModule`、**占领** `CaptureManagerBotModule` 各自独立。

### 1.6 为什么引擎不写成行为树（我的判断，附证据）

1. **订单是异步的**：`QueueOrder` 只是排进队列，单位要几秒后才到位；BT 的"Sequence 等子节点成功"在 RTS 里会卡住整棵树。状态机天然是"每 tick 看一眼现状再决定"。
2. **一次命令要给整队**：`groupedActors` 是引擎的原生语义，BT 通常没有"群组"概念。
3. **要能存档**：`Squad.Serialize/Deserialize` 存的是 `{Type, Units, ActorToTarget, TargetOffset}` —— 状态机+小队很容易序列化，整棵 BT 的运行时状态很难。
4. **算力可控**：小队状态更新按 `AttackForceInterval=75` 分摊到多个 tick（源码有 PERF 注释）；模糊推理只在决策点跑一次。
5. **模块并发**：经济/建造/侦察/进攻是并行推进的，用一棵 BT 串起来反而别扭。

> 所以：**要做"战术"，跟着引擎的范式（小队 + 状态机 + 评分）比自己发明一棵 BT 更省事、更稳、更能和引擎行为对齐。**

---

## 二、我们的方案：一层"战术小队"（同构，但服务于人类玩家的语音指挥）

### 2.1 定位

引擎的 SquadManager 只服务 **bot 玩家**；我们要指挥的是**人类玩家**自己的单位，所以：
- 仍然走 `world.IssueOrder(...)`（和玩家点击同一条路，规则/同步全部沿用）；
- 但**结构上照抄引擎**：`TacticalSquad = Units + Anchor + Target + StateMachine`；
- 复用引擎的判断函数（`AttackOrFleeFuzzy` 的思想、`CanAttackTarget`、`BusyAttack`、领队/重整）。

> 已经落地的一半：`VoiceTasks.cs` 里已经有 `BridgeTask`（World trait 上按 `Interval` 节流的循环任务）和 **`HarassTask`**（骚扰），以及 bridge 的 `task start/stop/list` 命令与 `group`（控制组）目标解析。

### 2.2 四个语音战术 = 四个状态（+ 共享的 Approach/Disengage）

| 语音意图 | 状态 | 行为要点 | 退出条件 |
|---|---|---|---|
| **骚扰 harass** | `KiteState` | 目标区附近纠缠；保持距离 `d = min(我的视野, 敌方射程 + extra)`；`HoldFire`（不还击）；只在锚点半径内 | 目标消失→Approach；被打崩→Disengage；超时 |
| **集火 focus** | `FocusState` | 锁定一个 `TargetActor`，全员 `Attack`/`AttackMove` 打它；目标死了自动换最近的 | 目标清光→Idle；评分不利→Disengage |
| **清扫战场 sweep** | `SweepState` | 在锚点半径内巡逻（复用探索的路点生成），见敌→AttackMove→清完继续巡 | 时长到 / 区域清干净 |
| **渐次撤退 retreat** | `BoundingState` | 分两班：A 班退到下一个集结点，B 班原地 `HoldFire` 掩护；到点换班，交替后撤 | 到达最终集结点 / 脱离危险半径 |

状态机就用引擎那套三段式（`Activate/Tick/Deactivate`），转移条件用**引擎同款评分**（血量比、火力比、速度比 → 阈值）+ 任务参数（时长、半径、extra、单位损失率）。

### 2.3 命令协议（bridge，向后兼容现有 5 条）

```
{"cmd":"task","args":{"op":"start","kind":"harass|focus|sweep|retreat",
   "group":1,                  // 控制组 1~9（"对一号编队"）
   "ids":[...],"filter":"...","parts":[...],   // 也支持显式指定
   "x":60,"y":40,              // 目标区（"在某某地方"）；缺省=单位当前位置
   "radius":12,                // 活动半径
   "extra":1,                  // 骚扰专用：敌射程外多留几格（就是用户说的 +x）
   "seconds":60}}              // 可选时长；0=一直做
{"cmd":"task","args":{"op":"list"}}     // 返回每个任务的状态：目标/standoff/距离/state/note
{"cmd":"task","args":{"op":"stop","id":1}}   // 或 op:stop 不带 id = 全停
```

### 2.4 语音层（继续"收缩意图、一点点加"）

- 新建 **openra-v3** profile：v2 的 5 个意图（下单/放置/取消/编队/探索）**加 1 个** `game_tactic`，参数里带 `kind` —— 这样只加一个意图就覆盖四种战术。
- 说法示例：
  - 「**对一号编队在 60 40 骚扰**」「一号编队去 80 60 那边骚扰」「骚扰，距离多留两格」
  - 「一号编队集火那个炮台」「一号编队集火 60 40」
  - 「一号编队清扫战场」「一号编队清扫 40 40 附近」
  - 「一号编队渐次撤退到 30 30」
- 回执：开始/结束都往游戏聊天框发一行（含 standoff 与实测距离），`task list` 可随时问「现在骚扰得怎么样」。

### 2.5 实施顺序与验收

| 步骤 | 内容 | 验收 |
|---|---|---|
| ① 收尾 harass | 语音入口（`game_tactic`）+ 真机验证 | 真机：下骚扰 → 单位进场 → 距离稳定在 `min(视野, 敌射程+extra)`；`task list` 数值合理 |
| ② focus | `FocusState`（纯 Attack 锁定 + 自动换目标） | 真机：3 辆车集火同一目标，目标死后换下一个 |
| ③ sweep | `SweepState`（复用 `AutoWaypoints`） | 真机：区域内巡逻→接敌→清完继续巡 |
| ④ retreat | `BoundingState`（分班+掩护+换班） | 真机：两班交替后撤，掩护班不开火不前进，最终到点 |

每步都加进 `openra/tools/live_test_*.py`（真机脚本），保持"改一处、验一处"。

### 2.6 约束与已知坑

- **必须在主线程下订单**（我们已经是：World trait 的 `ITick` + 队列）。
- **订单是异步的** → 用"节流（`Interval` 12~15 tick）+ 迟滞（±1 格）"避免每 tick 重下命令抖动（引擎自己也是这么干的）。
- **迷雾**：只能对 `!world.FogObscures(a)` 的敌人决策；骚扰要保持"看得见"，所以 `d ≤ 我的视野`（这正是 `min(视野, …)` 的意义）。
- **HoldFire 会关掉自动开火**：骚扰要的就是不还击；集火/清扫结束时要恢复姿态（记下原姿态并还原）。
- **不要和引擎 AI 抢同一批单位**：bot 玩家有 SquadManager，人类玩家没有 → 我们只控制 `world.LocalPlayer` 的单位，安全。
- **订单合法性**：所有订单仍过 `ValidateOrder`（subject 必须属于本客户端），不会绕过规则。

---

## 三、建议的下一步

1. 把 **harass 收尾**（语音入口 + 真机验证）—— 代码已完成 ~80%。
2. 然后按 ②③④ 顺序加 focus / sweep / retreat，每个都独立可回滚（v3 → v4 → …）。
