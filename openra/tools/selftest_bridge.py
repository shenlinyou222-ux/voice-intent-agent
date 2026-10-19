# -*- coding: utf-8 -*-
"""用【假游戏】验证桥协议与客户端/动作层（不需要真的开游戏）。

假游戏 = 一个 TCP 服务，按 openra/bridge_client.py 的协议回答：
    {"id":..,"cmd":"hello"}  -> {"ok":true,"data":{...}}
    {"id":..,"cmd":"state"}  -> 战况快照
    ...
它同时记录收到的每条指令，测试最后打印「指令轨迹」，用来核对
语音 → 意图 → 动作 → 指令 的映射是不是我们想要的。

用法：
  python openra/tools/selftest_bridge.py
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))       # voice_agent/ 目录

from openra.bridge_client import BridgeError, VoiceBridgeClient   # noqa: E402
from openra import game_actions                                    # noqa: E402
from openra import game_profile                                    # noqa: E402

STATE = {
    "mod": "ra", "tick": 4242, "player": "Commander",
    "cash": 5000, "resources": 0, "capacity": 0,
    "power": {"provided": 100, "drained": 60, "excess": 40},
    "map": {"title": "Testing Grounds", "width": 128, "height": 128},
    "selection": [11, 12],
    "units": [
        {"id": 11, "type": "2TNK", "x": 20, "y": 20, "hp": 100, "idle": True, "building": False},
        {"id": 12, "type": "2TNK", "x": 22, "y": 20, "hp": 100, "idle": True, "building": False},
        {"id": 13, "type": "HARV", "x": 30, "y": 30, "hp": 100, "idle": True, "building": False},
        {"id": 14, "type": "POWR", "x": 10, "y": 10, "hp": 80, "idle": True, "building": True},
    ],
    "enemies": [
        {"id": 91, "type": "3TNK", "x": 60, "y": 60, "hp": 100, "building": False},
        {"id": 92, "type": "GUN", "x": 80, "y": 20, "hp": 100, "building": True},
    ],
    "queues": [{"type": "Vehicle", "items": [{"item": "2TNK", "done": False, "paused": False,
                                              "remaining": 120, "total": 300}]},
               {"type": "Building", "items": [{"item": "powr", "done": True, "paused": False,
                                               "remaining": 0, "total": 180}]}],
}


class FakeGame(threading.Thread):
    """假游戏桥：语义与 C# 端 VoiceBridge 一致。"""

    def __init__(self):
        super().__init__(daemon=True)
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.trace: list[dict] = []
        self.running = True
        self.drop_next = False

    def run(self):
        while self.running:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        f = conn.makefile("rwb")
        try:
            while self.running:
                line = f.readline()
                if not line:
                    return
                if self.drop_next:
                    self.drop_next = False
                    return
                req = json.loads(line.decode("utf-8"))
                self.trace.append(req)
                resp = self._handle(req)
                f.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
                f.flush()
        except Exception:
            return
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _handle(self, req):
        rid, cmd = req.get("id"), req.get("cmd")
        args = req.get("args") or {}
        if cmd == "hello":
            return {"id": rid, "ok": True, "data": {"bridge": "fake-game", "version": 1, "mod": "ra"}}
        if cmd == "state":
            return {"id": rid, "ok": True, "data": STATE}
        if cmd == "buildable":
            return {"id": rid, "ok": True, "data": {
                "queues": [{"type": "Building", "items": ["powr"]}, {"type": "Defense", "items": ["fenc"]}],
                "actorCount": 307}}
        if cmd == "deployable":
            return {"id": rid, "ok": True, "data": {
                "actor": 11, "type": "mcv", "into": "fact", "can": True, "cell": [20, 20], "alternatives": []}}
        if cmd == "place":
            return {"id": rid, "ok": True, "data": {
                "item": args.get("item"), "x": args.get("x"), "y": args.get("y")}}
        if cmd == "move":
            return {"id": rid, "ok": True, "data": {"issued": 2, "x": args.get("x"), "y": args.get("y")}}
        if cmd == "select":
            return {"id": rid, "ok": True, "data": {"count": 2, "selection": [11, 12], "mode": args.get("mode")}}
        if cmd in ("attack", "attackmove", "guard", "rally", "ping", "chat", "build", "cancel",
                   "pause", "stop", "scatter", "harvest", "deploy", "sell", "repair", "capture",
                   "enter", "stance", "raw"):
            return {"id": rid, "ok": True, "data": {"issued": 2, "cmd": cmd, "args": args}}
        return {"id": rid, "ok": False, "err": f"unknown cmd '{cmd}'"}


class Cfg:
    def __init__(self, port):
        self.behavior = {"openra_port": port, "openra_host": "127.0.0.1"}


def main():
    game = FakeGame()
    game.start()
    time.sleep(0.2)
    cfg = Cfg(game.port)
    ok = bad = 0

    def check(name, cond, extra=""):
        nonlocal ok, bad
        if cond:
            ok += 1
            print(f"  ✓ {name} {extra}")
        else:
            bad += 1
            print(f"  ✗ {name} {extra}")

    print(f"假游戏桥: 127.0.0.1:{game.port}\n")

    # ── 1. 协议层 ──
    print("[1] 协议层")
    c = VoiceBridgeClient("127.0.0.1", game.port)
    hello = c.hello()
    check("hello", hello.get("bridge") == "fake-game", hello)
    st = c.state()
    check("state", st["cash"] == 5000 and len(st["units"]) == 4, f"cash={st['cash']} units={len(st['units'])}")
    d = c.move(60, 40)
    check("move", d.get("issued") == 2, d)
    try:
        c.call("nonsense")
        check("未知指令应报错", False)
    except BridgeError as e:
        check("未知指令应报错", "unknown cmd" in str(e), str(e))

    # ── 2. 断线重连 ──
    print("\n[2] 断线重连")
    game.drop_next = True
    try:
        d = c.move(10, 10)      # 服务端会掐掉这条连接，客户端应重连后重发
        check("重连后重发", d.get("issued") == 2, d)
    except BridgeError as e:
        check("重连后重发", False, str(e))

    # ── 3. 动作层（真实走 socket，不是 dry-run）──
    print("\n[3] 动作层（真连假游戏）")
    game.trace.clear()
    cases = [
        ("game_select", "queue:Vehicle"),
        ("game_move", "60,40"),
        ("game_attack", "nearest"),
        ("game_attackmove", "70,70"),
        ("game_stop", ""),
        ("game_scatter", ""),
        ("game_stance", "Defend"),
        ("game_harvest", ""),
        ("game_deploy", ""),
        ("game_build", "HARVx2"),
        ("game_place", "auto"),
        ("game_deploy", ""),
        ("game_cancel", "2TNK"),
        ("game_repair", ""),
        ("game_sell", ""),
        ("game_rally", "40,40"),
        ("game_ping", "50,50"),
        ("game_chat", "进攻"),
    ]
    for intent, arg in cases:
        fn = game_actions.GAME_ACTIONS[intent]
        good, msg, _ = fn(arg, cfg, dry=False)
        check(f"{intent}({arg})", good, msg)

    print("\n[4] 规则直通 → 意图 → 指令 轨迹")
    prof = game_profile.load()
    for text in ["选中所有坦克", "移动到 60 40", "攻击最近的敌人", "造两个矿车", "姿态防御"]:
        parsed = prof.parse(text)
        if parsed:
            fn = game_actions.GAME_ACTIONS[parsed[0]]
            good, msg, _ = fn(parsed[1], cfg, dry=False)
            print(f"  {text:14s} -> {parsed[0]:16s} {msg}")

    print(f"\n指令轨迹（假游戏收到 {len(game.trace)} 条）：")
    for t in game.trace:
        a = t.get("args") or {}
        brief = ",".join(f"{k}={v}" for k, v in a.items() if k != "queued") or "-"
        print(f"  #{t.get('id'):<3} {t.get('cmd'):<12} {brief}")

    game.running = False
    try:
        game.sock.close()
    except Exception:
        pass

    print(f"\n结果：{ok} 通过 / {bad} 失败")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
