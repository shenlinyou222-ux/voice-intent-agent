# -*- coding: utf-8 -*-
"""AoE4 语音控制的【执行侧】：意图 → 写命令文件 → 游戏内 SCAR 脚本执行。

与 voice_agent/actions.py 同构：每个动作 (arg, cfg, dry=False) -> (ok, msg, info)。
链路：语音 → ASR → 意图(规则直通) → aoe4_* 动作 → Scratch\\vb_cmd.txt
      → 游戏内 VoiceBridge_Poll 读到 → UnitEntry_DeploySquads / LocalCommand_EntityBuildSquad
"""
from __future__ import annotations

try:
    from . import bridge, game_profile
except ImportError:                       # 直接跑单文件时
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from aoe4 import bridge, game_profile


def profile_of(cfg=None) -> game_profile.Aoe4Profile:
    pid = None
    try:
        pid = (cfg.behavior or {}).get("aoe4_profile") if cfg else None
    except Exception:
        pid = None
    return game_profile.load(pid)


def _send(cmd: str, dry: bool, what: str) -> tuple[bool, str, dict]:
    if dry:
        return True, f"[dry] {what}（会写: {cmd}）", {"cmd": cmd, "dry": True}
    if not bridge.is_game_running():
        return False, f"{what}失败：游戏没在跑（RelicCardinal.exe）", None
    ok, msg = bridge.send(cmd)
    if not ok:
        return False, f"{what}失败：{msg}", None
    return True, f"{what}：{msg}", {"cmd": cmd, "path": str(bridge.cmd_path())}


def _send_many(cmd_one: str, n: int, dry: bool, what: str) -> tuple[bool, str, dict]:
    """★ 一次调用 = 一个单位；数量 N = 调用 N 次（各占一个槽位文件）。"""
    n = max(1, min(9, int(n)))
    if dry:
        return True, f"[dry] {what} × {n}（每条: {cmd_one}）", {"cmd": cmd_one, "n": n, "dry": True}
    if not bridge.is_game_running():
        return False, f"{what}失败：游戏没在跑（RelicCardinal.exe）", None
    # ★ 一次性写 N 个槽（bridge.send_many 内部写完后统一清其它槽）。
    #   不能循环调 bridge.send —— 每次 send 都会清槽，会把刚写的前几条清掉 ✗
    ok, msg = bridge.send_many(cmd_one, n)
    if not ok:
        return False, f"{what}失败：{msg}", None
    return True, f"{what} × {n}（{msg}）", {"cmd": cmd_one, "n": n, "path": str(bridge.cmd_path())}


def act_aoe4_spawn_villager(arg, cfg, dry=False):
    n = int(arg) if str(arg or "").strip().isdigit() else 1
    return _send_many("spawn villager 1", n, dry, "生成农民")


def act_aoe4_train_villager(arg, cfg, dry=False):
    n = int(arg) if str(arg or "").strip().isdigit() else 1
    return _send_many("train villager 1", n, dry, "训练农民")


def act_aoe4_ping(arg, cfg, dry=False):
    return _send("ping", dry, "探活")


def act_aoe4_key(arg, cfg, dry=False):
    """按键通道：按一下游戏里绑定的键 → 游戏内回调执行（UI_AddCommandBinding）。"""
    combo = None
    try:
        combo = (cfg.behavior or {}).get("aoe4_key") if cfg else None
    except Exception:
        combo = None
    combo = combo or (arg or "").strip() or None
    if not combo:
        return False, "没配置按键（behavior.aoe4_key）", None
    if dry:
        return True, f"[dry] 会按 {combo}", {"key": combo}
    if not bridge.is_game_running():
        return False, "按键失败：游戏没在跑", None
    ok, msg = bridge.press_key(combo)
    return ok, msg, {"key": combo}


def act_aoe4_state(arg, cfg, dry=False):
    st = bridge.last_state()
    if st:
        return True, f"游戏侧状态：{st}", {"state": st}
    return True, "游戏侧还没有回执（脚本每秒写一次 state）", None


# ══════════════════════════════════════════════════════════════════
# v17 部队编队（对应 SCAR 侧 mark / guard / guardstop）
#   mark      = 标记当前屏幕中心（游戏侧读 Misc_GetScreenCenterPosition）
#   guard N   = N 队攻击移动到标记点，到位后散开游走、遇敌向基地后撤
# ══════════════════════════════════════════════════════════════════

def act_aoe4_mark(arg, cfg, dry=False):
    return _send("mark", dry, "标记此处")


def act_aoe4_guard(arg, cfg, dry=False):
    n = int(arg) if str(arg or "").strip().isdigit() else 1
    n = max(1, min(9, n))
    return _send(f"doctrine guard {n}", dry, f"{n} 队过去警戒")


def act_aoe4_doctrine(arg, cfg, dry=False):
    """arg = "<战术名>:<队号>"（如 scout:1）→ 写 "doctrine scout 1"。

    战术名来自 SCAR 侧的 VB_DOCTRINES 表（事先编排好的动作树模板）：
      guard 警戒 / scout 探查 / harass 骚扰 / support 火力支援
    """
    name, _, num = str(arg or "guard:1").partition(":")
    name = (name or "guard").strip().lower()
    n = int(num) if num.strip().isdigit() else 1
    n = max(1, min(9, n))
    label = {"guard": "警戒", "scout": "探查", "harass": "骚扰",
             "support": "火力支援"}.get(name, name)
    return _send(f"doctrine {name} {n}", dry, f"{n} 队过去{label}")


def act_aoe4_alloc(arg, cfg, dry=False):
    """arg = "food,wood,gold,stone"（如 "7,0,2,3"）→ 写 "alloc 7 0 2 3"

    对应 SCAR 的 VoiceBridge_Alloc：数一遍当前工人 → 把超额的调去缺人的资源
    """
    parts = [p.strip() for p in str(arg or "").split(",")]
    nums = []
    for i in range(4):
        try:
            nums.append(max(0, min(99, int(parts[i]))))
        except Exception:
            nums.append(0)
    cmd = "alloc " + " ".join(str(x) for x in nums)
    label = f"肉{nums[0]} 木{nums[1]} 金{nums[2]} 石{nums[3]}"
    return _send(cmd, dry, f"调配工人到 {label}")


def act_aoe4_tech(arg, cfg, dry=False):
    """arg = attrib 里的科技短名（如 upgrade_melee_armor_i）→ 写 "tech <短名>"

    对应 SCAR 侧 VoiceBridge_ResearchTech：
      BP_GetUpgradeBlueprint(短名) → LocalCommand_PlayerUpgrade(pid, ubp, false, false)
    ★ 不需要点开建筑面板 ✓（消灭子界面）
    """
    name = str(arg or "").strip()
    if not name:
        return False, "没给科技短名", None
    return _send(f"tech {name}", dry, f"研究科技 {name}")


def act_aoe4_guardstop(arg, cfg, dry=False):
    return _send("guardstop", dry, "解除警戒")


GAME_ACTIONS = {
    "aoe4_spawn_villager": act_aoe4_spawn_villager,
    "aoe4_train_villager": act_aoe4_train_villager,
    "aoe4_ping": act_aoe4_ping,
    "aoe4_state": act_aoe4_state,
    "aoe4_key": act_aoe4_key,
    "aoe4_mark": act_aoe4_mark,
    "aoe4_guard": act_aoe4_guard,
    "aoe4_doctrine": act_aoe4_doctrine,
    "aoe4_guardstop": act_aoe4_guardstop,
    "aoe4_tech": act_aoe4_tech,
    "aoe4_alloc": act_aoe4_alloc,
}


if __name__ == "__main__":
    for name, fn in GAME_ACTIONS.items():
        print(f"  {name}: {fn('2', None, dry=True)}")
