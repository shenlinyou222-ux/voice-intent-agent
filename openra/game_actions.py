# -*- coding: utf-8 -*-
"""OpenRA 语音控制的【执行侧】—— 与 voice_agent/actions.py 同构的动作函数。

每个函数签名 (arg, cfg, dry=False) -> (ok, msg, info)，注册进 actions.ACTIONS 后
就可以被 voice_agent 的意图层直接调度。

链路：
    语音 → ASR(版本化热词) → 意图(规则直通 + 版本化 embedding 索引)
        → game_* 动作（本文件）→ openra/bridge_client → 游戏内 VoiceBridge trait
        → world.IssueOrder(...)  ← 与玩家点击完全同一条路径

安全：
    · 只在 dry_run 时完全不连游戏
    · 桥连不上就返回 ok=False 并说明原因，绝不静默失败
"""
from __future__ import annotations

import json
import math
import time
try:                                    # 作为包导入（voice_agent 里 import openra.game_actions）
    from . import game_profile
    from .bridge_client import BridgeError, VoiceBridgeClient
except ImportError:                     # 直接 python openra/game_actions.py 跑自测
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from openra import game_profile
    from openra.bridge_client import BridgeError, VoiceBridgeClient

_CLIENT: VoiceBridgeClient | None = None
_CLIENT_KEY: tuple | None = None


# ══════════════════════════════════════════════════════════════
#  连接管理
# ══════════════════════════════════════════════════════════════
def profile_of(cfg=None) -> game_profile.GameProfile:
    pid = None
    try:
        pid = (cfg.behavior or {}).get("openra_profile") if cfg else None
    except Exception:
        pid = None
    return game_profile.load(pid)


def bridge_settings(prof: game_profile.GameProfile, cfg=None) -> dict:
    b = dict(prof.bridge)
    try:
        beh = cfg.behavior or {}
        b["host"] = beh.get("openra_host", b.get("host", "127.0.0.1"))
        b["port"] = int(beh.get("openra_port", b.get("port", 38900)))
        b["token"] = beh.get("openra_token", b.get("token"))
        b["timeout"] = float(beh.get("openra_timeout", b.get("timeout", 10.0)))
    except Exception:
        pass
    # 游戏窗口失焦/最小化时 OpenRA 逻辑 tick 会变稀，超时太短会偶发失败
    b["timeout"] = max(float(b.get("timeout", 10.0)), 8.0)
    return b


def client(cfg=None, prof: game_profile.GameProfile | None = None) -> VoiceBridgeClient:
    """取（或重建）到游戏桥的连接。"""
    global _CLIENT, _CLIENT_KEY
    prof = prof or profile_of(cfg)
    b = bridge_settings(prof, cfg)
    key = (b["host"], int(b["port"]), b.get("token"), b.get("timeout"))
    if _CLIENT is not None and _CLIENT_KEY == key and _CLIENT.connected:
        return _CLIENT
    _CLIENT = VoiceBridgeClient(b["host"], int(b["port"]), b.get("token"), float(b.get("timeout", 3.0)))
    _CLIENT_KEY = key
    return _CLIENT


def probe(cfg=None) -> dict:
    prof = profile_of(cfg)
    b = bridge_settings(prof, cfg)
    try:
        c = client(cfg, prof)
        info = c.hello()
        return {"up": True, "info": info, "profile": prof.version_report()}
    except Exception as e:
        return {"up": False, "err": str(e), "profile": prof.version_report(),
                "bridge": {"host": b["host"], "port": b["port"]}}


# ══════════════════════════════════════════════════════════════
#  小工具
# ══════════════════════════════════════════════════════════════
def _coords(arg: str) -> tuple[int, int] | None:
    if not arg:
        return None
    parts = str(arg).replace("，", ",").split(",")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _coords_flex(arg, cfg=None) -> tuple[int, int] | None:
    """坐标容错：参数可能是 '60,40'（规则直通切好的），
    也可能是整句「移动到六十 四十」（模型路把原句当参数传过来）。
    """
    c = _coords(arg)
    if c:
        return c
    try:
        prof = profile_of(cfg)
        return prof.coords(str(arg or ""), prof.game.get("coord_max", 128))
    except Exception:
        return None


def _centroid(units: list[dict]) -> tuple[int, int] | None:
    if not units:
        return None
    return (round(sum(u["x"] for u in units) / len(units)),
            round(sum(u["y"] for u in units) / len(units)))


def _nearest_enemy(state: dict) -> dict | None:
    mine = [u for u in state.get("units", []) if not u.get("building")]
    enemies = state.get("enemies", [])
    if not enemies:
        return None
    c = _centroid(mine) or (0, 0)
    return min(enemies, key=lambda e: (e["x"] - c[0]) ** 2 + (e["y"] - c[1]) ** 2)


def _here(state: dict) -> tuple[int, int] | None:
    """「这里/这边」没有屏幕坐标时的兜底：当前选中部队的重心。"""
    sel = set(state.get("selection") or [])
    units = [u for u in state.get("units", []) if (not sel or u["id"] in sel)]
    return _centroid(units) or _centroid(state.get("units", []))


def _guard(arg, cfg, dry, need_coords=True):
    """把 arg 解析成 (x, y)；「nearest/here/空」时用状态里的兜底。"""
    c = _coords(arg)
    if c:
        return c, None
    if dry:
        return None, "dry-run：需要坐标（或先选中部队）"
    st = client(cfg).state()
    if str(arg).lower().startswith("near") or need_coords is False:
        e = _nearest_enemy(st)
        if e:
            return (e["x"], e["y"]), None
    h = _here(st)
    if h:
        return h, None
    return None, "拿不到坐标：请说「移动到 x y」或先选中部队"


# ══════════════════════════════════════════════════════════════
#  动作
# ══════════════════════════════════════════════════════════════
def act_game_select(arg, cfg, dry=False):
    prof = profile_of(cfg)
    filt = arg or "all"
    if filt not in ("all", "idle", "mobile", "building", "harvester", "armed") \
            and not filt.startswith(("type:", "queue:", "near")):
        filt = prof.filter_for(filt)
    if dry:
        return True, f"[dry] 选中 {filt}", None
    d = client(cfg).select(mode="set", filter=filt)
    return True, f"选中 {d.get('count', 0)} 个单位（{filt}）", d


def act_game_move(arg, cfg, dry=False):
    c = _coords_flex(arg, cfg)
    if not c:
        return False, f"移动需要坐标（收到 {arg!r}）", None
    if dry:
        return True, f"[dry] 移动到 {c[0]},{c[1]}", None
    d = client(cfg).move(*c)
    return True, f"移动 {d.get('issued', 0)} 个单位到 {c[0]},{c[1]}", d


def act_game_attack(arg, cfg, dry=False):
    a = (arg or "").strip().lower()
    if a and a not in ("nearest", "here") and _coords(arg):
        c = _coords(arg)
        if dry:
            return True, f"[dry] 攻击 {c[0]},{c[1]}", None
        d = client(cfg).attack(x=c[0], y=c[1])
        return True, f"攻击 {c[0]},{c[1]}（{d.get('issued', 0)} 个单位）", d

    if dry:
        return True, "[dry] 攻击最近的敌人", None
    st = client(cfg).state()
    e = _nearest_enemy(st)
    if not e:
        return False, "视野里没有敌人", None
    d = client(cfg).attack(x=e["x"], y=e["y"])
    return True, f"攻击最近的敌人 {e['type']}@{e['x']},{e['y']}（{d.get('issued', 0)} 个单位）", d


def act_game_attackmove(arg, cfg, dry=False):
    c = _coords_flex(arg, cfg)
    if not c:
        if dry:
            return True, "[dry] 攻击前进（无坐标）", None
        c, err = _guard(arg, cfg, dry, need_coords=False)
        if not c:
            return False, err, None
    if dry:
        return True, f"[dry] 攻击前进到 {c[0]},{c[1]}", None
    d = client(cfg).attackmove(*c)
    return True, f"攻击前进到 {c[0]},{c[1]}（{d.get('issued', 0)} 个单位）", d


def act_game_stop(arg, cfg, dry=False):
    if dry:
        return True, "[dry] 停止", None
    d = client(cfg).stop()
    return True, f"{d.get('issued', 0)} 个单位停止", d


def act_game_scatter(arg, cfg, dry=False):
    if dry:
        return True, "[dry] 散开", None
    d = client(cfg).scatter()
    return True, f"{d.get('issued', 0)} 个单位散开", d


def act_game_guard(arg, cfg, dry=False):
    c = _coords_flex(arg, cfg)
    if not c and not dry:
        c, err = _guard(arg, cfg, dry)
        if not c:
            return False, err, None
    if dry:
        return True, "[dry] 守卫", None
    d = client(cfg).guard(*c)
    return True, f"守卫 {c[0]},{c[1]}（{d.get('issued', 0)} 个单位）", d


def act_game_stance(arg, cfg, dry=False):
    prof = profile_of(cfg)
    stance = prof.stance(arg or "") or (arg or "").strip()
    if not stance:
        return False, "没听清姿态（停火/还击/防御/主动攻击）", None
    if dry:
        return True, f"[dry] 姿态 {stance}", None
    d = client(cfg).stance(stance)
    return True, f"姿态切换为 {stance}（{d.get('issued', 0)} 个单位）", d


def act_game_harvest(arg, cfg, dry=False):
    if dry:
        return True, "[dry] 采矿", None
    d = client(cfg).harvest()
    return True, f"{d.get('issued', 0)} 辆矿车去采矿", d


def act_game_deploy(arg, cfg, dry=False):
    """展开基地车。

    ⭐ 智能版：先问游戏「这里能展开吗」，不能就先挪到它给的可展开格子再展开。
       为什么必须这样：语音用户只会说「展开」，不会报坐标；而地图上到处是
       不能放置的地形（实测在 A Nuclear Winter 中央和边缘都放过不下）。
    """
    if dry:
        return True, "[dry] 展开", None

    b = client(cfg)
    info = b.deployable()
    if info.get("can"):
        d = b.deploy()
        return True, f"原地展开 {info.get('type')} → {info.get('into')}", d

    alts = info.get("alternatives") or []
    if not alts:
        return False, f"这附近没有能展开的位置（{info.get('reason') or '地形不允许'}）", info

    x, y = alts[0]
    b.move(x, y)
    for _ in range(30):
        time.sleep(0.5)
        st = b.state()
        unit = next((u for u in st.get("units", []) if u["id"] == info["actor"]), None)
        if unit and (unit["x"], unit["y"]) == (x, y) and not unit.get("activity"):
            break

    again = b.deployable()
    if not again.get("can"):
        return False, f"挪到 {x},{y} 后仍然放不下", again

    d = b.deploy()
    return True, f"先移动到 {x},{y}，再展开 {info.get('type')} → {info.get('into')}", d


def act_game_repair(arg, cfg, dry=False):
    if dry:
        return True, "[dry] 修理", None
    d = client(cfg).repair()
    return True, f"修理 {d.get('issued', 0)} 个建筑", d


def act_game_sell(arg, cfg, dry=False):
    if dry:
        return True, "[dry] 出售", None
    d = client(cfg).sell()
    return True, f"出售 {d.get('issued', 0)} 个建筑", d


def act_game_capture(arg, cfg, dry=False):
    if dry:
        return True, "[dry] 占领", None
    st = client(cfg).state()
    e = _nearest_enemy(st)
    if not e:
        return False, "视野里没有可占领的目标", None
    d = client(cfg).capture(id=e["id"])
    return True, f"占领 {e['type']}@{e['x']},{e['y']}", d


def act_game_rally(arg, cfg, dry=False):
    c = _coords_flex(arg, cfg)
    if not c and not dry:
        c, err = _guard(arg, cfg, dry)
        if not c:
            return False, err, None
    if dry:
        return True, "[dry] 集结点", None
    d = client(cfg).rally(*c)
    return True, f"集结点设为 {c[0]},{c[1]}", d


def act_game_build(arg, cfg, dry=False):
    prof = profile_of(cfg)
    raw = (arg or "").strip()
    if not raw:
        return False, "没听清要造什么", None
    count = 1
    item_text = raw
    if "x" in raw.lower():
        left, _, right = raw.lower().rpartition("x")
        item_text = left or raw
        if right.isdigit():
            count = max(1, min(50, int(right)))
    item = prof.item(item_text)
    if not item:
        return False, f"不认识这个单位/建筑：{item_text}", None
    # ⭐ 模型路会把整句当参数传过来（「两个矿车。」）→ 数量从句子里抠
    explicit = "x" in raw.lower() and raw.lower().rpartition("x")[2].isdigit()
    if not explicit:
        count = prof.count(raw, count)
    if dry:
        return True, f"[dry] 生产 {item['name']}({item['id']}) x{count}", None
    d = client(cfg).build(item["id"], count)
    return True, f"生产 {item['name']} x{count}", {"item": item, **d}


def act_game_place(arg, cfg, dry=False):
    """放置已经造好的建筑（红警里建筑造完必须放到地图上）。

    没给坐标就用「自己建筑的重心 + 3 格」当落点，桥那边会自动在附近找能放的位置。
    """
    if dry:
        return True, "[dry] 放置建筑", None

    b = client(cfg)
    st = b.state()
    ready = [i["item"] for q in st.get("queues", []) for i in q["items"] if i.get("done")]
    if not ready:
        return False, "没有待放置的建筑（先说「建造电厂」之类）", None

    item = ready[0]
    c = _coords(arg)
    if not c:
        buildings = [u for u in st.get("units", []) if u.get("building")]
        if buildings:
            c = (round(sum(u["x"] for u in buildings) / len(buildings)) + 3,
                 round(sum(u["y"] for u in buildings) / len(buildings)) + 3)
        else:
            c = (st["map"]["width"] // 2, st["map"]["height"] // 2)

    d = b.place(item, c[0], c[1])
    return True, f"放置 {item} 到 {d.get('x')},{d.get('y')}", d


def act_game_group(arg, cfg, dry=False):
    """语音编队：把指定单位编成 1~9 队（红警的控制组）。

    arg 可能是：
      · 规则直通给的 JSON：{"n":1,"mode":"set","parts":[{"filter":"type:2tnk","count":3},…]}
      · 模型路给的整句：「三辆坦克二步兵编为一队」→ 现场解析
    """
    prof = profile_of(cfg)
    spec = None
    raw = str(arg or "").strip()
    if raw.startswith("{"):
        try:
            spec = json.loads(raw)
        except Exception:
            spec = None
    if spec is None:
        spec = prof.group_spec(raw) or {}

    n = int(spec.get("n", 1))
    mode = spec.get("mode", "set")
    parts = spec.get("parts") or []
    if dry:
        desc = "、".join(f"{p.get('name', p['filter'])}×{p.get('count') or '全部'}" for p in parts) or "当前选中"
        return True, f"[dry] 编 {n} 队：{desc}", None

    d = client(cfg).group(n=n, mode=mode, parts=parts)
    desc = "、".join(f"{p.get('name', p['filter'])}×{p.get('count') or '全部'}" for p in parts)
    return True, f"{n} 队编好了：{d.get('count', 0)} 个单位" + (f"（{desc}）" if desc else ""), d


def act_game_explore(arg, cfg, dry=False):
    """自动探索：挑一个跑得最快的（或选中的）单位，沿未探索区域串出来的路线跑一遍。"""
    if dry:
        return True, "[dry] 自动探索", None
    d = client(cfg).explore()
    wps = d.get("waypoints") or []
    return True, f"{d.get('type')} 去侦察：{len(wps)} 个路点", d


def act_game_cancel(arg, cfg, dry=False):
    prof = profile_of(cfg)
    item = prof.item(arg or "") if arg else None
    if not item:
        return False, f"要取消什么？（收到 {arg!r}）", None
    if dry:
        return True, f"[dry] 取消 {item['name']}", None
    d = client(cfg).cancel(item["id"], 1)
    return True, f"取消生产 {item['name']}", d


def act_game_state(arg, cfg, dry=False):
    if dry:
        return True, "[dry] 查询战况", None
    st = client(cfg).state()
    power = st.get("power", {})
    queues = st.get("queues", [])
    qtxt = "、".join(
        f"{q['type']}:" + "/".join(i["item"] for i in q["items"][:4]) for q in queues if q["items"])
    msg = (f"现金 {st.get('cash', 0)}，电力 {power.get('provided', 0)}/{power.get('drained', 0)}"
           f"（余 {power.get('excess', 0)}），我方单位 {len(st.get('units', []))} 个，"
           f"视野内敌人 {len(st.get('enemies', []))} 个"
           + (f"，生产中 {qtxt}" if qtxt else ""))
    return True, msg, st


def act_game_ping(arg, cfg, dry=False):
    c = _coords_flex(arg, cfg)
    if not c and not dry:
        c, err = _guard(arg, cfg, dry)
        if not c:
            return False, err, None
    if dry:
        return True, "[dry] 标记", None
    d = client(cfg).ping(*c, color="Red")
    return True, f"标记 {c[0]},{c[1]}", d


def act_game_chat(arg, cfg, dry=False):
    text = (arg or "").strip()
    if not text:
        return False, "要说什么？", None
    if dry:
        return True, f"[dry] 喊话 {text}", None
    d = client(cfg).chat(text)
    return True, f"已喊话：{text}", d


# ══════════════════════════════════════════════════════════════
#  注册表 + 自测
# ══════════════════════════════════════════════════════════════
GAME_ACTIONS = {
    "game_select": act_game_select,
    "game_move": act_game_move,
    "game_attack": act_game_attack,
    "game_attackmove": act_game_attackmove,
    "game_stop": act_game_stop,
    "game_scatter": act_game_scatter,
    "game_guard": act_game_guard,
    "game_stance": act_game_stance,
    "game_harvest": act_game_harvest,
    "game_deploy": act_game_deploy,
    "game_repair": act_game_repair,
    "game_sell": act_game_sell,
    "game_capture": act_game_capture,
    "game_rally": act_game_rally,
    "game_build": act_game_build,
    "game_place": act_game_place,
    "game_cancel": act_game_cancel,
    "game_group": act_game_group,
    "game_explore": act_game_explore,
    "game_state": act_game_state,
    "game_ping": act_game_ping,
    "game_chat": act_game_chat,
}


class _NullCfg:
    """没有 voice_agent Config 时的最小替身（自测用）。"""
    behavior = {}


def selftest(dry=True):
    """不需要游戏也能跑：走规则直通 → 动作（dry-run）全链路。"""
    prof = profile_of()
    cfg = _NullCfg()
    probes = [
        "选中所有坦克", "移动到 60 40", "攻击最近的敌人", "攻击前进到 70 70",
        "停止", "散开", "姿态防御", "矿车去采矿", "基地车展开", "造两个矿车",
        "取消建造坦克", "修理建筑", "卖掉电厂", "占领敌方建筑", "设置集结点 40 40",
        "标记 50 50", "喊话进攻",
    ]
    ok = 0
    print(f"profile {prof.id}  hotwords={prof.hotwords_version}  index={prof.index_version}")
    for text in probes:
        parsed = prof.parse(text)
        if not parsed:
            print(f"  ✗ {text:18s} 规则直通没命中")
            continue
        intent, arg = parsed
        fn = GAME_ACTIONS.get(intent)
        if not fn:
            print(f"  ✗ {text:18s} {intent} 没有动作实现")
            continue
        good, msg, _info = fn(arg, cfg, dry)
        ok += bool(good)
        print(f"  {'✓' if good else '✗'} {text:18s} -> {intent:16s} {msg}")
    print(f"\n{ok}/{len(probes)} 通过（dry_run={dry}）")
    return ok, len(probes)


if __name__ == "__main__":
    import sys

    selftest(dry="--live" not in sys.argv)
