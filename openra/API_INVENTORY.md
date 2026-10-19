# OpenRA 玩家可控 API 清点（自动生成）

源码：`E:\OpenRA`　生成脚本：`openra/tools/api_inventory.py`

## 0. 总量

| 面 | 数量 |
|---|---|
| Lua 脚本 API（global）| 123 个方法 / 21 个对象 |
| Lua 脚本 API（actor）| 81 个方法 / 30 个对象 |
| Lua 脚本 API（cnc）| 5 个方法 / 4 个对象 |
| **Lua 脚本 API 合计** | **209** |
| 指令(order)字符串（有实现者）| 46 |
| 指令(order)字符串（代码里被发出）| 38 |
| 聊天命令（/xxx）| 30 |
| Bot 模块 | 42 |
| Trait（模组 API 面）| 594 |

## 1. Lua 脚本 API（玩家/地图脚本直接调用）

### 1.1 global（123 方法）

| 对象 | 文件 | 方法数 | 方法 |
|---|---|---|---|
| TriggerGlobal | OpenRA.Mods.Common\Scripting\Global\TriggerGlobal.cs | 37 | `GetScriptTriggers`, `AfterDelay`, `OnPassengerEntered`, `OnPassengerExited`, `OnIdle`, `OnDamaged`, `OnKilled`, `OnAllKilled`, `OnAnyKilled`, `OnProduction`, `OnAnyProduction`, `OnPlayerWon`, `OnPlayerLost`, `OnObjectiveAdded`, `OnObjectiveCompleted`, `OnObjectiveFailed`, `OnBuildingPlaced`, `OnAddedToWorld`, `OnRemovedFromWorld`, `OnAllRemovedFromWorld`, `OnCapture`, `OnKilledOrCaptured`, `OnAllKilledOrCaptured`, `OnEnteredFootprint` … (+13) |
| MapGlobal | OpenRA.Mods.Common\Scripting\Global\MapGlobal.cs | 13 | `ActorsInCircle`, `ActorsInBox`, `RandomCell`, `RandomEdgeCell`, `ClosestEdgeCell`, `ClosestMatchingEdgeCell`, `CenterOfCell`, `TerrainType`, `LobbyOption`, `LobbyOptionOrDefault`, `NamedActor`, `IsNamedActor`, `ActorsWithTag` |
| MediaGlobal | OpenRA.Mods.Common\Scripting\Global\MediaGlobal.cs | 13 | `PlaySpeechNotification`, `PlaySoundNotification`, `PlaySound`, `PlayMusic`, `SetBackgroundMusic`, `StopMusic`, `PlayMovieFullscreen`, `PlayMovieInRadar`, `DisplayMessage`, `DisplayMessageToPlayer`, `DisplaySystemMessage`, `Debug`, `FloatingText` |
| UtilsGlobal | OpenRA.Mods.Common\Scripting\Global\UtilsGlobal.cs | 12 | `Do`, `Any`, `All`, `Where`, `Take`, `Skip`, `Concat`, `Random`, `Shuffle`, `ExpandFootprint`, `RandomInteger`, `FormatTime` |
| ScriptPropertyGroupAttribute | OpenRA.Game\Scripting\ScriptContext.cs | 10 | `ScriptPropertyGroupAttribute`, `ScriptActorProperties`, `ScriptPlayerProperties`, `ScriptGlobalAttribute`, `FatalError`, `RegisterMapActor`, `WorldLoaded`, `Tick`, `Dispose`, `CreateTable` |
| CPosGlobal | OpenRA.Mods.Common\Scripting\Global\CoordinateGlobals.cs | 7 | `New`, `NewWithLayer`, `New`, `New`, `New`, `New`, `FromCells` |
| ScriptMemberWrapper | OpenRA.Game\Scripting\ScriptMemberWrapper.cs | 4 | `Get`, `Set`, `WrappableMembers`, `RequiredTraitNames` |
| ActorGlobal | OpenRA.Mods.Common\Scripting\Global\ActorGlobal.cs | 4 | `Create`, `BuildTime`, `CruiseAltitude`, `Cost` |
| ScriptMemberExts | OpenRA.Game\Scripting\ScriptMemberExts.cs | 3 | `LuaDocString`, `LuaDocString`, `LuaDocString` |
| LuaValueExts | OpenRA.Game\Scripting\ScriptTypes.cs | 3 | `WrappedClrType`, `TryGetClrValue`, `ToLuaValue` |
| ColorGlobal | OpenRA.Mods.Common\Scripting\Global\ColorGlobal.cs | 3 | `New`, `FromRGB`, `FromHex` |
| DateGlobal | OpenRA.Mods.Common\Scripting\Global\DateTimeGlobal.cs | 2 | `Seconds`, `Minutes` |
| PlayerGlobal | OpenRA.Mods.Common\Scripting\Global\PlayerGlobal.cs | 2 | `GetPlayer`, `GetPlayers` |
| ReinforcementsGlobal | OpenRA.Mods.Common\Scripting\Global\ReinforcementsGlobal.cs | 2 | `Reinforce`, `ReinforceWithTransport` |
| UserInterfaceGlobal | OpenRA.Mods.Common\Scripting\Global\UserInterfaceGlobal.cs | 2 | `SetMissionText`, `GetFluentMessage` |
| ScriptActorInterface | OpenRA.Game\Scripting\ScriptActorInterface.cs | 1 | `OnActorDestroyed` |
| ScriptObjectWrapper | OpenRA.Game\Scripting\ScriptObjectWrapper.cs | 1 | `ContainsKey` |
| AngleGlobal | OpenRA.Mods.Common\Scripting\Global\AngleGlobal.cs | 1 | `New` |
| BeaconGlobal | OpenRA.Mods.Common\Scripting\Global\BeaconGlobal.cs | 1 | `New` |
| LightingGlobal | OpenRA.Mods.Common\Scripting\Global\LightingGlobal.cs | 1 | `Flash` |
| RadarGlobal | OpenRA.Mods.Common\Scripting\Global\RadarGlobal.cs | 1 | `Ping` |

### 1.2 actor（81 方法）

| 对象 | 文件 | 方法数 | 方法 |
|---|---|---|---|
| BaseActorProperties | OpenRA.Mods.Common\Scripting\Properties\GeneralProperties.cs | 10 | `HasProperty`, `Flash`, `Teleport`, `CallFunc`, `Wait`, `Destroy`, `Stop`, `AddTag`, `RemoveTag`, `HasTag` |
| MissionObjectiveProperties | OpenRA.Mods.Common\Scripting\Properties\MissionObjectiveProperties.cs | 10 | `AddObjective`, `AddPrimaryObjective`, `AddSecondaryObjective`, `MarkCompletedObjective`, `MarkFailedObjective`, `IsObjectiveCompleted`, `IsObjectiveFailed`, `GetObjectiveDescription`, `GetObjectiveType`, `HasNoRequiredUnits` |
| CombatProperties | OpenRA.Mods.Common\Scripting\Properties\CombatProperties.cs | 6 | `Hunt`, `AttackMove`, `Patrol`, `PatrolUntil`, `Attack`, `CanTarget` |
| MobileProperties | OpenRA.Mods.Common\Scripting\Properties\MobileProperties.cs | 5 | `Move`, `ScriptedMove`, `MoveIntoWorld`, `Scatter`, `EnterTransport` |
| PlayerProperties | OpenRA.Mods.Common\Scripting\Properties\PlayerProperties.cs | 5 | `GetActors`, `GetGroundAttackers`, `GetActorsByType`, `GetActorsByTypes`, `HasPrerequisites` |
| ProductionProperties | OpenRA.Mods.Common\Scripting\Properties\ProductionProperties.cs | 5 | `Produce`, `Build`, `IsProducing`, `Build`, `IsProducing` |
| AircraftProperties | OpenRA.Mods.Common\Scripting\Properties\AircraftProperties.cs | 4 | `Move`, `ReturnToBase`, `Land`, `Resupply` |
| AmmoPoolProperties | OpenRA.Mods.Common\Scripting\Properties\AmmoPoolProperties.cs | 3 | `AmmoCount`, `MaximumAmmoCount`, `Reload` |
| ConditionProperties | OpenRA.Mods.Common\Scripting\Properties\ConditionProperties.cs | 3 | `GrantCondition`, `RevokeCondition`, `AcceptsCondition` |
| PlayerConditionProperties | OpenRA.Mods.Common\Scripting\Properties\PlayerConditionProperties.cs | 3 | `GrantCondition`, `RevokeCondition`, `AcceptsCondition` |
| TransportProperties | OpenRA.Mods.Common\Scripting\Properties\TransportProperties.cs | 3 | `LoadPassenger`, `UnloadPassenger`, `UnloadPassengers` |
| CaptureProperties | OpenRA.Mods.Common\Scripting\Properties\CaptureProperties.cs | 2 | `Capture`, `CanCapture` |
| CarryallProperties | OpenRA.Mods.Common\Scripting\Properties\CarryallProperties.cs | 2 | `PickupCarryable`, `DeliverCarryable` |
| DeliversCashProperties | OpenRA.Mods.Common\Scripting\Properties\DeliveryProperties.cs | 2 | `DeliverCash`, `DeliverExperience` |
| GainsExperienceProperties | OpenRA.Mods.Common\Scripting\Properties\GainsExperienceProperties.cs | 2 | `GiveExperience`, `GiveLevels` |
| RepairableBuildingProperties | OpenRA.Mods.Common\Scripting\Properties\RepairableBuildingProperties.cs | 2 | `StartBuildingRepairs`, `StopBuildingRepairs` |
| AirstrikeProperties | OpenRA.Mods.Common\Scripting\Properties\AirstrikeProperties.cs | 1 | `TargetAirstrike` |
| DemolitionProperties | OpenRA.Mods.Common\Scripting\Properties\DemolitionProperties.cs | 1 | `Demolish` |
| DiplomacyProperties | OpenRA.Mods.Common\Scripting\Properties\DiplomacyProperties.cs | 1 | `IsAlliedWith` |
| GuardProperties | OpenRA.Mods.Common\Scripting\Properties\GuardProperties.cs | 1 | `Guard` |
| HarvesterProperties | OpenRA.Mods.Common\Scripting\Properties\HarvesterProperties.cs | 1 | `FindResources` |
| HealthProperties | OpenRA.Mods.Common\Scripting\Properties\HealthProperties.cs | 1 | `Kill` |
| InstantlyRepairsProperties | OpenRA.Mods.Common\Scripting\Properties\InstantlyRepairsProperties.cs | 1 | `InstantlyRepair` |
| NukeProperties | OpenRA.Mods.Common\Scripting\Properties\NukeProperties.cs | 1 | `ActivateNukePower` |
| ParadropProperties | OpenRA.Mods.Common\Scripting\Properties\ParadropProperties.cs | 1 | `Paradrop` |
| ParatroopersProperties | OpenRA.Mods.Common\Scripting\Properties\ParatroopersProperties.cs | 1 | `TargetParatroopers` |
| PlayerPowerProperties | OpenRA.Mods.Common\Scripting\Properties\PowerProperties.cs | 1 | `TriggerPowerOutage` |
| ScaredCatProperties | OpenRA.Mods.Common\Scripting\Properties\ScaredCatProperties.cs | 1 | `Panic` |
| SellableProperties | OpenRA.Mods.Common\Scripting\Properties\SellableProperties.cs | 1 | `Sell` |
| TransformProperties | OpenRA.Mods.Common\Scripting\Properties\TransformProperties.cs | 1 | `Deploy` |

### 1.3 cnc（5 方法）

| 对象 | 文件 | 方法数 | 方法 |
|---|---|---|---|
| DisguiseProperties | OpenRA.Mods.Cnc\Scripting\Properties\DisguiseProperties.cs | 2 | `DisguiseAs`, `DisguiseAsType` |
| ChronosphereProperties | OpenRA.Mods.Cnc\Scripting\Properties\ChronosphereProperties.cs | 1 | `Chronoshift` |
| InfiltrateProperties | OpenRA.Mods.Cnc\Scripting\Properties\InfiltrateProperties.cs | 1 | `Infiltrate` |
| IonCannonProperties | OpenRA.Mods.Cnc\Scripting\Properties\IonCannonProperties.cs | 1 | `ActivateIonCannon` |

## 2. 指令(order) API

### 2.1 引擎/模组能处理的指令（玩家点一下就会产生）

| 指令 | 处理者 |
|---|---|
| `ActivateCondition` | GrantChargedConditionOnToggleInfo |
| `AfterDeployTransform` | TransformsIntoTransformsInfo |
| `AssaultMove` | AttackMoveInfo |
| `AttackMove` | AttackMoveInfo |
| `BeginMinefield` | MinelayerInfo |
| `C4` | DemolitionInfo |
| `CancelProduction` | ProductionQueueInfo |
| `CaptureActor` | CapturesInfo |
| `DeliverCash` | DeliversCashInfo |
| `DeliverExperience` | DeliversExperienceInfo |
| `DeliverUnit` | CarryallInfo |
| `DeployTransform` | TransformsInfo, TransformsIntoTransformsInfo |
| `Detonate` | MadTankInfo |
| `DetonateAttack` | MadTankInfo |
| `Disguise` | DisguiseTooltipInfo |
| `Dock` | DockClientManagerInfo, TransformsIntoDockClientInfo |
| `Enter` | AircraftInfo, TransformsIntoAircraftInfo |
| `EnterTransport` | PassengerInfo, TransformsIntoPassengerInfo |
| `EnterTunnel` | EntersTunnelsInfo, TransformsIntoEntersTunnelsInfo |
| `ForceDock` | DockClientManagerInfo, TransformsIntoDockClientInfo |
| `ForceEnter` | AircraftInfo |
| `GrantConditionOnDeploy` | GrantConditionOnDeployInfo |
| `Guard` | GuardInfo |
| `Harvest` | HarvesterInfo |
| `Infiltrate` | InfiltratesInfo |
| `InstantRepair` | InstantlyRepairsInfo |
| `Land` | AircraftInfo |
| `Move` | AircraftInfo, MobileInfo, TransformsIntoAircraftInfo, TransformsIntoMobileInfo |
| `PauseProduction` | ProductionQueueInfo |
| `PickupUnit` | CarryallInfo |
| `PlaceBeacon` | PlaceBeaconInfo |
| `PlaceMine` | MinelayerInfo |
| `PlaceMinefield` | MinelayerInfo |
| `PortableChronoTeleport` | PortableChronoInfo |
| `Repair` | RepairableInfo, TransformsIntoRepairableInfo |
| `RepairBridge` | RepairsBridgesInfo |
| `RepairBuilding` | AllyRepairInfo |
| `RepairNear` | RepairableNearInfo |
| `ReturnToBase` | AircraftInfo |
| `Scatter` | AircraftInfo, MobileInfo |
| `Sell` | SellableInfo |
| `SetUnitStance` | AutoTargetInfo |
| `StartProduction` | ProductionQueueInfo |
| `Stop` | AircraftInfo, AttackBaseInfo, MobileInfo, RallyPointInfo, TransformsIntoMobileInfo |
| `Surrender` | MissionObjective |
| `Unload` | CargoInfo, CarryallInfo |

### 2.2 代码里会被发出的指令（含 AI/Bot）

| 指令 | 发出者 |
|---|---|
| `ActivateCondition` | GrantChargedConditionOnToggleInfo |
| `AfterDeployTransform` | TransformsIntoTransformsInfo |
| `Attack` | AirStateBase |
| `AttackMove` | GroundStateBase, UnitsForProtectionIdleState |
| `BeginMinefield` | MinelayerInfo |
| `CancelProduction` | OrderFieldsExts |
| `CaptureActor` | CaptureManagerBotModuleInfo |
| `Chat` | OrderFieldsExts |
| `Command` | OrderFieldsExts |
| `CreateGroup` | UnitOrderGenerator |
| `DeployTransform` | McvExpansionManagerBotModuleInfo, McvManagerBotModuleInfo, TransformsInfo |
| `Detonate` | MadTankInfo |
| `DisableChatEntry` | PlayerMessageTracker |
| `Dock` | HarvesterBotModuleInfo |
| `EndKickVote` | VoteKickTracker |
| `GrantConditionOnDeploy` | GrantConditionOnDeployInfo |
| `HandshakeRequest` | Server |
| `HandshakeResponse` | Server, UnitOrders |
| `Harvest` | HarvesterBotModuleInfo |
| `Move` | AIUtils, AirStateBase, McvExpansionManagerBotModuleInfo, McvManagerBotModuleInfo, MinelayerBotModuleInfo, MobileInfo |
| `PauseProduction` | OrderFieldsExts |
| `PlaceBeacon` | BeaconOrderGenerator |
| `PlaceMine` | MinelayerInfo |
| `PlaceMinefield` | MinelayerBotModuleInfo, MinelayerInfo |
| `PortableChronoTeleport` | PortableChronoInfo |
| `PurchaseOrder` | PurchaseWidgetLogic |
| `RepairBuilding` | BuildingRepairBotModuleInfo, RepairOrderGenerator |
| `ReturnOrder` | ProductionIcon |
| `ReturnToBase` | AirStateBase, AircraftInfo |
| `Scatter` | CommandBarLogic |
| `Sell` | BaseBuilderBotModuleInfo |
| `SetRallyPoint` | BaseBuilderBotModuleInfo |
| `SetUnitStance` | StanceSelectorLogic |
| `StartKickVote` | VoteKickTracker |
| `StartProduction` | OrderFieldsExts |
| `Stop` | AttackOrderPowerInfo, CommandBarLogic, GroundStateBase |
| `Surrender` | IngameMenuLogic, PlayerCommandsInfo |
| `Unload` | CargoInfo, CarryallInfo |

## 3. 聊天命令 API（玩家在聊天框输入）

| 文件 | 类 | 命令 |
|---|---|---|
| OpenRA.Mods.Common\Commands\DebugVisualizationCommands.cs | DebugVisualizationCommandsInfo | `/DevActorTags`, `/DevCombatGeometry`, `/DevDepthBuffer`, `/DevRenderGeometry`, `/DevScreenMap`, `/actor-tags`, `/combat-geometry`, `/depth-buffer`, `/render-geometry`, `/screen-map` |
| OpenRA.Mods.Common\Commands\DevCommands.cs | DevCommandsInfo | `/all`, `/build-anywhere`, `/clear-shroud`, `/crash`, `/dispose`, `/enable-tech`, `/fast-charge`, `/give-cash`, `/give-cash-all`, `/grow-resources`, `/heal`, `/instant-build`, `/kill`, `/player-experience`, `/reset-shroud`, `/unlimited-power`, `/visibility` |
| OpenRA.Mods.Common\Commands\HelpCommand.cs | HelpCommandInfo | `/help` |
| OpenRA.Mods.Common\Commands\PlayerCommands.cs | PlayerCommandsInfo | `/pause`, `/surrender` |

## 4. Bot/AI 模块（自动化控制的既有范式）

| 模块 | 说明 | 会发出的指令 |
|---|---|---|
| AirStateBase |  | `Attack`, `Move`, `ReturnToBase` |
| AirStateBase |  | `Attack`, `Move`, `ReturnToBase` |
| AttackOrFleeFuzzy |  |  |
| AttackOrFleeFuzzy |  |  |
| BaseBuilderBotModuleInfo | Manages AI base construction. | `Sell`, `SetRallyPoint` |
| BaseBuilderBotModuleInfo | Manages AI base construction. | `Sell`, `SetRallyPoint` |
| BaseBuilderQueueManager |  |  |
| BaseBuilderQueueManager |  |  |
| BuildingRepairBotModuleInfo | Manages AI repairing base buildings. | `RepairBuilding` |
| BuildingRepairBotModuleInfo | Manages AI repairing base buildings. | `RepairBuilding` |
| CaptureManagerBotModuleInfo | Manages AI capturing logic. | `CaptureActor` |
| CaptureManagerBotModuleInfo | Manages AI capturing logic. | `CaptureActor` |
| GroundStateBase |  | `AttackMove`, `Stop` |
| GroundStateBase |  | `AttackMove`, `Stop` |
| HarvesterBotModuleInfo | Put this on the Player actor. Manages bot harvesters to ensu | `Dock`, `Harvest` |
| HarvesterBotModuleInfo | Put this on the Player actor. Manages bot harvesters to ensu | `Dock`, `Harvest` |
| McvExpansionManagerBotModuleInfo | Manages AI MCVs and expansion. | `DeployTransform`, `Move` |
| McvExpansionManagerBotModuleInfo | Manages AI MCVs and expansion. | `DeployTransform`, `Move` |
| McvManagerBotModuleInfo | Manages AI MCVs. | `DeployTransform`, `Move` |
| McvManagerBotModuleInfo | Manages AI MCVs. | `DeployTransform`, `Move` |
| MinelayerBotModuleInfo | Manages AI minelayer unit related with  | `Move`, `PlaceMinefield` |
| MinelayerBotModuleInfo | Manages AI minelayer unit related with  | `Move`, `PlaceMinefield` |
| PowerDownBotModuleInfo | Manages AI powerdown. |  |
| PowerDownBotModuleInfo | Manages AI powerdown. |  |
| ResourceMapBotModuleInfo | Harvestable and valuable resource types. |  |
| ResourceMapBotModuleInfo | Harvestable and valuable resource types. |  |
| Squad |  |  |
| Squad |  |  |
| SquadManagerBotModuleInfo | Manages AI squads. |  |
| SquadManagerBotModuleInfo | Manages AI squads. |  |
| StateBase |  | `Move` |
| StateBase |  | `Move` |
| StateMachine |  |  |
| StateMachine |  |  |
| SupportPowerBotModuleInfo | Manages bot support power handling. |  |
| SupportPowerBotModuleInfo | Manages bot support power handling. |  |
| SupportPowerDecision | Adds metadata for the AI bots. |  |
| SupportPowerDecision | Adds metadata for the AI bots. |  |
| UnitBuilderBotModuleInfo | Controls AI unit production. |  |
| UnitBuilderBotModuleInfo | Controls AI unit production. |  |
| UnitsForProtectionIdleState |  | `AttackMove` |
| UnitsForProtectionIdleState |  | `AttackMove` |

## 5. Trait 统计（模组层 API 面）

| 程序集 | TraitInfo 数 |
|---|---|
| OpenRA.Mods.Common | 508 |
| OpenRA.Mods.Cnc | 72 |
| OpenRA.Mods.D2k | 14 |
