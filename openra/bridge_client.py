# -*- coding: utf-8 -*-
"""OpenRA 语音控制桥 —— 客户端（纯 stdlib）。

游戏内 `VoiceBridge` trait 在 127.0.0.1:<port> 上开了一个 NDJSON（一行一个 JSON）的 TCP 口。
这个模块把它包成 Python 侧的对象：

    from openra.bridge_client import VoiceBridgeClient
    b = VoiceBridgeClient()          # 默认 127.0.0.1:38900
    print(b.hello())                 # {"bridge": "...", "mod": "ra", ...}
    st = b.state()                   # 战况快照（钱/电/单位/敌人/生产队列）
    b.select(filter="mobile", mode="set")
    b.move(60, 40)
    b.build("2tnk", 2)

协议（v1）
----------
请求   {"id": <int>, "cmd": "<name>", "token": <str?>, "args": {...}}
响应   {"id": <int>, "ok": true,  "data": {...}}
       {"id": <int>, "ok": false, "err": "<原因>"}

约定：一个连接 = 一个游戏内客户端；失败自动重连一次。
"""
from __future__ import annotations

import json
import socket
import threading
import time

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 38900
PROTOCOL_VERSION = 1


class BridgeError(RuntimeError):
    """桥返回 ok=false，或者根本连不上。"""


class VoiceBridgeClient:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 token: str | None = None, timeout: float = 8.0, autoconnect: bool = True):
        self.host = host
        self.port = int(port)
        self.token = token
        self.timeout = float(timeout)
        self._sock: socket.socket | None = None
        self._reader = None
        # RLock：失败重连时会在持锁状态下递归调用 call()，普通 Lock 会自锁死
        self._lock = threading.RLock()
        self._seq = 0
        self.last_error: str | None = None
        if autoconnect:
            try:
                self.connect()
            except BridgeError as e:
                self.last_error = str(e)

    # ── 连接管理 ─────────────────────────────────────────────
    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> "VoiceBridgeClient":
        self.close()
        try:
            s = socket.create_connection((self.host, self.port), timeout=self.timeout)
        except OSError as e:
            raise BridgeError(f"连不上游戏桥 {self.host}:{self.port}（{e}）——游戏里 VoiceBridge 起来了吗？") from e
        s.settimeout(self.timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = s
        self._reader = s.makefile("rb")
        self.last_error = None
        return self

    def close(self):
        for obj in (self._reader, self._sock):
            try:
                if obj is not None:
                    obj.close()
            except Exception:
                pass
        self._sock = None
        self._reader = None

    def __enter__(self):
        if not self.connected:
            self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    # ── 请求 ─────────────────────────────────────────────────
    def call(self, cmd: str, _retry: bool = True, **args) -> dict:
        with self._lock:
            if not self.connected:
                try:
                    self.connect()
                except BridgeError:
                    if _retry:
                        raise
            self._seq += 1
            payload = {"id": self._seq, "cmd": cmd}
            if self.token:
                payload["token"] = self.token
            if args:
                payload["args"] = args

            try:
                self._sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                line = self._reader.readline()
            except (OSError, AttributeError) as e:
                self.close()
                if _retry:
                    return self.call(cmd, _retry=False, **args)
                raise BridgeError(f"桥连接断了：{e}") from e

            if not line:
                self.close()
                if _retry:
                    return self.call(cmd, _retry=False, **args)
                raise BridgeError("桥关闭了连接")

            try:
                resp = json.loads(line.decode("utf-8"))
            except ValueError as e:
                raise BridgeError(f"桥返回的不是 JSON：{line[:200]!r}") from e

        if not resp.get("ok"):
            raise BridgeError(resp.get("err") or "桥返回失败")
        return resp.get("data") or {}

    # ── 便捷方法（与 VoiceBridge trait 的 cmd 一一对应）──────
    def hello(self):
        return self.call("hello")

    def state(self):
        return self.call("state")

    def select(self, mode="set", filter="all", ids=None, x=None, y=None, r=None):
        args = {"mode": mode}
        if ids:
            args["ids"] = list(ids)
        else:
            args["filter"] = filter
            if x is not None:
                args["x"] = x
            if y is not None:
                args["y"] = y
            if r is not None:
                args["r"] = r
        return self.call("select", **args)

    def move(self, x, y, queued=False):
        return self.call("move", x=int(x), y=int(y), queued=bool(queued))

    def attack(self, x=None, y=None, id=None, queued=False):
        args = {"queued": bool(queued)}
        if id is not None:
            args["id"] = int(id)
        else:
            args["x"], args["y"] = int(x), int(y)
        return self.call("attack", **args)

    def attackmove(self, x, y, queued=False):
        return self.call("attackmove", x=int(x), y=int(y), queued=bool(queued))

    def stop(self):
        return self.call("stop")

    def scatter(self):
        return self.call("scatter")

    def guard(self, x, y, queued=False):
        return self.call("guard", x=int(x), y=int(y), queued=bool(queued))

    def stance(self, stance):
        return self.call("stance", stance=stance)

    def harvest(self):
        return self.call("harvest")

    def deploy(self):
        return self.call("deploy")

    def sell(self):
        return self.call("sell")

    def repair(self):
        return self.call("repair")

    def enter(self, id=None, x=None, y=None, order=None):
        args = {}
        if id is not None:
            args["id"] = int(id)
        else:
            args["x"], args["y"] = int(x), int(y)
        if order:
            args["order"] = order
        return self.call("enter", **args)

    def capture(self, id=None, x=None, y=None):
        args = {"id": int(id)} if id is not None else {"x": int(x), "y": int(y)}
        return self.call("capture", **args)

    def build(self, item, count=1):
        return self.call("build", item=item, count=int(count))

    def cancel(self, item, count=1):
        return self.call("cancel", item=item, count=int(count))

    def pause(self, item, pause=True):
        return self.call("pause", item=item, pause=bool(pause))

    def rally(self, x, y):
        return self.call("rally", x=int(x), y=int(y))

    def buildable(self):
        """本局当前可生产的条目（由游戏自己报，避免名字猜错）。"""
        return self.call("buildable")

    def deployable(self):
        """当前选中的单位能不能在原地展开；不能的话给出附近可以展开的格子。"""
        return self.call("deployable")

    def place(self, item, x, y, variant=0):
        return self.call("place", item=item, x=int(x), y=int(y), variant=int(variant))

    def group(self, n, mode="set", parts=None, filter=None, ids=None):
        """编队（红警控制组）：n=1~9；parts=[{filter,count}] 表达「三辆坦克+两步兵」。"""
        args = {"n": int(n), "mode": mode}
        if parts:
            args["parts"] = parts
        elif filter:
            args["filter"] = filter
        elif ids:
            args["ids"] = list(ids)
        return self.call("group", **args)

    def explore(self, id=None, points=None, count=None):
        """自动探索：不给路点就在未探索区域自动串一条路线。"""
        args = {}
        if id is not None:
            args["id"] = int(id)
        if points:
            args["points"] = [list(p) for p in points]
        if count:
            args["count"] = int(count)
        return self.call("explore", **args)

    def ping(self, x, y, color=None):
        args = {"x": int(x), "y": int(y)}
        if color:
            args["color"] = color
        return self.call("ping", **args)

    def chat(self, text):
        return self.call("chat", text=str(text))

    def raw(self, order, ids=None, id=None, x=None, y=None, target_string=None,
            extra_data=None, queued=False):
        args = {"order": order, "queued": bool(queued)}
        if ids:
            args["ids"] = list(ids)
        if id is not None:
            args["id"] = int(id)
        if x is not None:
            args["x"] = int(x)
        if y is not None:
            args["y"] = int(y)
        if target_string is not None:
            args["target_string"] = target_string
        if extra_data is not None:
            args["extra_data"] = int(extra_data)
        return self.call("raw", **args)


def probe(host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=1.5) -> dict:
    """探活：返回 {up: bool, info|err}。给 --list / 诊断用。"""
    t0 = time.perf_counter()
    try:
        with VoiceBridgeClient(host, port, timeout=timeout) as c:
            info = c.hello()
        info["rtt_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return {"up": True, "info": info}
    except Exception as e:
        return {"up": False, "err": str(e)}


if __name__ == "__main__":
    import sys

    host = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOST
    port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT
    r = probe(host, port)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    sys.exit(0 if r["up"] else 1)
