# -*- coding: utf-8 -*-
"""真机验证新命令：编队(group) + 自动探索(explore)。

· 编队：把当前所有建筑编成 1 队，再按队选中 → 验证控制组真的建起来了
· 探索：等出现一个能动的单位（精炼厂造好会带一辆矿车），让它去探未探索区域，
        然后看它是不是真的沿路点跑起来了

用法: python openra/tools/live_test_newcmds.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from openra.bridge_client import BridgeError, VoiceBridgeClient  # noqa: E402

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1
        print(f"  ✓ {name} {extra}")
    else:
        BAD += 1
        print(f"  ✗ {name} {extra}")
    return cond


def mobile_units(b):
    return [u for u in b.state()["units"] if u.get("mobile")]


def main():
    b = VoiceBridgeClient(timeout=15)
    st = b.state()
    print(f"局面：现金 {st['cash']} 单位 {[(u['type'], u['x'], u['y']) for u in st['units']]}\n")

    # ── 1. 编队（用建筑验，建筑也能编队）──
    print("[1] 编队 group")
    try:
        d = b.group(1, mode="set", filter="building")
        check("把所有建筑编成 1 队", d.get("count", 0) >= 1, d)
        d2 = b.group(1, mode="select")
        check("按 1 队选中", d2.get("count", 0) >= 1, d2)
        d3 = b.group(2, mode="set", parts=[{"filter": "building", "count": 1}])
        check("parts 形式编 2 队", d3.get("count", 0) == 1, d3)
    except BridgeError as e:
        check("编队", False, str(e))

    # ── 2. 等一个能动的单位，然后自动探索 ──
    print("\n[2] 自动探索 explore（等一个能动的单位，最多 3 分钟）")
    scout = None
    for i in range(36):
        mus = mobile_units(b)
        if mus:
            scout = mus[0]
            print(f"     找到侦察兵：{scout['type']} @({scout['x']},{scout['y']})  [等待 {i*5}s]")
            break
        time.sleep(5)

    if not scout:
        print("     （3 分钟内没有能动的单位 —— 精炼厂/兵营还没好，跳过 explore 实测）")
    else:
        before = (scout["x"], scout["y"])
        try:
            d = b.explore(id=scout["id"], count=5)
            wps = d.get("waypoints") or []
            check("explore 给出路点", len(wps) >= 1, f"{len(wps)} 个：{wps[:5]}")
        except BridgeError as e:
            check("explore", False, str(e))
            wps = []
        if wps:
            moved = False
            for _ in range(20):
                time.sleep(3)
                now = [u for u in mobile_units(b) if u["id"] == scout["id"]]
                if now and (now[0]["x"], now[0]["y"]) != before:
                    moved = True
                    print(f"     位置：{before} -> ({now[0]['x']},{now[0]['y']})  活动={now[0]['activity']}")
                    break
            check("侦察兵真的动起来了", moved)

    print(f"\n结果：{OK} 通过 / {BAD} 失败")
    return 0 if BAD == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
