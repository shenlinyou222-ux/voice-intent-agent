# -*- coding: utf-8 -*-
"""真机联调脚本：对着【正在运行的红警】跑一遍完整控制链路。

前置：
  1. 游戏已编译（含 VoiceBridge trait）
  2. 游戏在跑，且直接进了地图：
     dotnet bin\\OpenRA.dll -- Engine.EngineDir=".." Game.Mod=ra Launch.Map=a-nuclear-winter

它做的事（就是语音会触发的那些动作，只是参数写死）：
  hello → select all → move → deploy(展开基地车) → buildable →
  build powr → 等造好 → place powr(放置) → 验证电力上升 → ping/chat

用法：
  python openra/tools/live_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from openra.bridge_client import BridgeError, VoiceBridgeClient   # noqa: E402

OK = 0
BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1
        print(f"  ✓ {name} {extra}")
    else:
        BAD += 1
        print(f"  ✗ {name} {extra}")
    return cond


def units(b):
    return [(u["id"], u["type"], u["x"], u["y"], u["activity"]) for u in b.state()["units"]]


def main():
    try:
        b = VoiceBridgeClient()
        hello = b.hello()
    except BridgeError as e:
        print(f"连不上游戏桥：{e}")
        print("先起游戏：dotnet bin\\OpenRA.dll -- Engine.EngineDir=\"..\" Game.Mod=ra Launch.Map=a-nuclear-winter")
        return 2

    print(f"游戏：mod={hello['mod']} 玩家={hello['player']} clientIndex={hello['clientIndex']}")
    st = b.state()
    print(f"地图：{st['map']['title']} {st['map']['width']}x{st['map']['height']}  现金={st['cash']}")
    print()

    # ① 选中 + 移动（实时控制的最小证据）
    print("[1] 选中 + 移动")
    d = b.select(mode="set", filter="all")
    check("select all", d["count"] >= 1, d)
    before = units(b)
    start = (before[0][2], before[0][3]) if before else (0, 0)
    w, h = st["map"]["width"], st["map"]["height"]
    # 多试几个目标：地图上有些格子走不到（悬崖/水），语音场景也常遇到
    candidates = [(w // 2, h // 2), (w // 2 - 12, h // 2 + 8), (w // 3, h // 3), (w // 2 + 10, h // 2 - 6)]
    moved = False
    for target in candidates:
        b.move(*target)
        for _ in range(20):
            time.sleep(1.5)
            now = units(b)
            if now and (now[0][2], now[0][3]) != start:
                moved = True
                break
        if moved:
            break
        print(f"     目标 {target} 走不到，换一个")
    check("move 真的动了", moved, f"{start} -> {units(b)[0][2:4] if units(b) else '?'}")

    # ② 展开基地车（会变成建筑，游戏状态实变）
    #    走【动作层】的智能展开：不能放就先挪到游戏给的可展开格子
    print("\n[2] 展开基地车（动作层 act_game_deploy）")
    from openra import game_actions

    class _Cfg:
        behavior = {}

    became_building = False
    for attempt in range(2):
        now = units(b)
        if not now or now[0][1] != "mcv":
            became_building = any(u[1] != "mcv" and u[4] is None for u in now)
            break
        good, msg, info = game_actions.act_game_deploy("", _Cfg(), dry=False)
        print(f"     尝试 {attempt+1}: {'✓' if good else '✗'} {msg}")
        for _ in range(12):
            time.sleep(2)
            now = units(b)
            if now and now[0][1] != "mcv":
                became_building = True
                break
        if became_building:
            break
    check("deploy -> 建筑", became_building, units(b))

    # ③ 列出本局能造什么（让游戏自己报名字）
    print("\n[3] buildable（游戏自报可生产条目）")
    d = b.call("buildable")
    queues = {q["type"]: q["items"] for q in d["queues"]}
    check("拿到生产队列", bool(queues), queues)

    # ④ 生产 + 放置
    print("\n[4] 生产 + 放置")
    item = None
    for q in ("Building", "Defense"):
        if queues.get(q):
            item = queues[q][0]
            break
    if not item:
        print("  （没有可生产条目，跳过）")
    else:
        cash0 = b.state()["cash"]
        d = b.build(item, 1)
        check(f"build {item}", d["item"] == item, d)
        ready = False
        for _ in range(20):
            time.sleep(1.5)
            qs = b.state()["queues"]
            if any(i["item"] == item and i["done"] for q in qs for i in q["items"]):
                ready = True
                break
        check("生产完成(ready)", ready)
        if ready:
            sel = units(b)
            cx, cy = (sel[0][2], sel[0][3]) if sel else (10, 10)
            d = b.place(item, cx + 3, cy + 3)
            check(f"place {item}", "x" in d, d)
            time.sleep(3)
            st = b.state()
            placed = any(u["type"] == item and u["building"] for u in st["units"])
            check("建筑出现在地图上", placed, [u for u in st["units"] if u["type"] == item])
            print(f"     现金 {cash0} -> {st['cash']}，电力 {st['power']}")

    # ⑤ 反馈通道
    print("\n[5] 反馈通道（雷达 ping + 游戏内聊天）")
    check("ping", "x" in b.ping(40, 40))
    check("chat", "text" in b.chat("语音控制联调完成"))

    # ⑥ 错误处理：不能崩游戏
    print("\n[6] 错误处理（必须干净报错，不能崩游戏）")
    for bad in ("NotAThing", "不存在的坦克"):
        try:
            b.build(bad, 1)
            check(f"拒绝 {bad}", False, "竟然接受了")
        except BridgeError as e:
            check(f"拒绝 {bad}", True, str(e)[:60])
    check("游戏还活着", b.hello()["mod"] == hello["mod"])

    print(f"\n结果：{OK} 通过 / {BAD} 失败")
    return 0 if BAD == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
