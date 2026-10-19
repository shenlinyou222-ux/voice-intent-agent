# -*- coding: utf-8 -*-
"""AoE4 语音桥：把命令写进游戏可写的 Scratch 目录，游戏内 SCAR 脚本轮询读取。

通道依据（都是实测/日志证据）：
  · 游戏日志：Using [C:\\Users\\user\\Documents\\My Games\\Age of Empires IV\\] as base writable folder
              Using [...] as base transient writable folder
  · exe 里注册的别名：save / screen / playback / cache / scratch / screenshots
  · 用户目录里确实存在 Scratch\\ 与 Cache\\ 文件夹
  · 游戏内脚本读 scratch:vb_cmd.txt（见 mod 的 voicecontroller.scar）

命令文本（脚本侧解析）：
    spawn villager [n]     直接生成 n 个农民
    train villager [n]     城镇中心训练 n 个农民
    ping                   探活
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

USER_DIR = Path(os.path.expanduser("~")) / "Documents" / "My Games" / "Age of Empires IV"
SCRATCH = USER_DIR / "Scratch"
CACHE = USER_DIR / "Cache"

CMD_NAME = "vb_cmd.txt"
STATE_NAME = "vb_cmd.txt.state"

# 多槽轮转：每条命令换一个文件名，规避"引擎按路径缓存文件内容"（实测第一条命令生效、
# 后续写同一路径不生效）。内容尾部带 #序号，脚本侧按序号去重，重复口令也能再次执行。
SLOTS = 12


def _seed_seq() -> int:
    """★★ 血债：_SEQ 必须【跨进程重启】单调递增 ✗→✓

    原来 _SEQ = 0 写死，每次 agent 重启就归零 ✗ →
      重启后写的是【同样的文件 + 同样的 #序号】→
      引擎按路径缓存过这些文件的读取 → SCAR 脚本看到序号没变 →
      判定"不是新命令" → **不执行**（实测：18:38 之后命令游戏侧全无反应 ✗）
    修法：时间戳播种 + 取现有槽位里的最大序号 → 永远比历史大 ✓
    """
    base = int(time.time() * 10)
    mx = 0
    try:
        for p in USER_DIR.glob("vb_cmd_*.txt"):
            m = re.search(r"#(\d+)\s*$",
                          p.read_text(encoding="utf-8", errors="ignore").strip())
            if m:
                mx = max(mx, int(m.group(1)))
    except Exception:
        pass
    return max(base, mx + 1)


_SEQ = _seed_seq()


def cmd_path() -> Path:
    """主通道文件（兼容/调试用）。"""
    return USER_DIR / CMD_NAME


def slot_path(seq: int) -> Path:
    return USER_DIR / f"vb_cmd_{(seq % SLOTS) + 1}.txt"


def _clear_stale(keep: set[Path]) -> None:
    """把没在本次命令里的槽位清空 —— 防止【测试数据污染】（旧命令被下一局读到）。

    游戏脚本每秒轮询所有槽位：空文件被跳过；同一槽位只有内容变化才会执行，
    而每条命令都带唯一 #序号，所以清空不会漏执行。
    """
    for i in range(1, SLOTS + 1):
        p = USER_DIR / f"vb_cmd_{i}.txt"
        if p in keep:
            continue
        try:
            if p.exists() and p.read_text(encoding="utf-8", errors="replace").strip():
                p.write_text("", encoding="utf-8")
        except Exception:
            pass
    # 主通道也清空（脚本同样会读它）
    try:
        cmd_path().write_text("", encoding="utf-8")
    except Exception:
        pass


def send_many(cmd_one: str, n: int) -> tuple[bool, str]:
    """写 N 条【各自一个单位】的命令，并清掉其它槽位（防污染）。"""
    global _SEQ
    n = max(1, min(9, int(n)))
    cmd_one = (cmd_one or "").strip()
    if not cmd_one:
        return False, "空命令"
    keep: set[Path] = set()
    try:
        for _ in range(n):
            _SEQ += 1
            p = slot_path(_SEQ)
            p.write_text(f"{cmd_one} #{_SEQ}\n", encoding="utf-8")
            keep.add(p)
        _clear_stale(keep)
    except Exception as e:
        return False, f"写命令失败: {type(e).__name__}: {e}"
    return True, f"已发送 {n} 条: {cmd_one}"


def state_path() -> Path:
    return SCRATCH / STATE_NAME


def send(cmd: str, wait: float = 0.0) -> tuple[bool, str]:
    """兼容入口：写一条命令（并清掉其它槽位，防测试数据污染）。"""
    ok, msg = send_many(cmd, 1)
    if ok and wait:
        time.sleep(wait)
    return ok, msg


def last_state() -> str:
    try:
        p = state_path()
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        pass
    return ""


_GAME_CACHE = [0.0, False]      # [时间戳, 结果]
_GAME_TTL = 10.0                # ★ 缓存 10 秒：tasklist 子进程要 50~70ms，
                                #   每次动作都跑一次是纯浪费（实测动作段 72ms 就是它）

def is_game_running() -> bool:
    """游戏是否在跑（进程名）。带 10 秒缓存 —— 别再每次动作都 spawn tasklist。"""
    now = time.time()
    if now - _GAME_CACHE[0] < _GAME_TTL:
        return _GAME_CACHE[1]
    try:
        import subprocess
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq RelicCardinal.exe"],
                             capture_output=True, text=True, timeout=5).stdout
        _GAME_CACHE[1] = "RelicCardinal" in out
    except Exception:
        pass
    _GAME_CACHE[0] = now
    return _GAME_CACHE[1]


# ── 按键通道 ──────────────────────────────────────────────────────────
# 游戏内脚本用 UI_AddCommandBinding("hud_game", "toggle_diplomacy_panel", "VoiceBridge_OnKey")
# 把这个键绑到我们的回调（官方 diplomacy.scar 就是这么绑的）。
# 语音侧只要"按一下这个键"，游戏内回调就执行 → 指挥单位。
VK = {
    "ctrl": 0x11, "alt": 0x12, "shift": 0x10,
    "enter": 0x0D, "esc": 0x1B, "tab": 0x09, "space": 0x20,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73, "f5": 0x74, "f6": 0x75,
    "f7": 0x76, "f8": 0x77, "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    ",": 0xBC, ".": 0xBE, "/": 0xBF, ";": 0xBA, "'": 0xDE, "[": 0xDB, "]": 0xDD,
    "\\": 0xDC, "-": 0xBD, "=": 0xBB, "`": 0xC0,
}
for _c in "abcdefghijklmnopqrstuvwxyz":
    VK[_c] = ord(_c.upper())
for _d in "0123456789":
    VK[_d] = ord(_d)


def press_key(combo: str) -> tuple[bool, str]:
    """按一下某个键（可组合，如 "ctrl+h"）。只发键盘事件，不碰鼠标。"""
    import ctypes
    import time as _t
    u = ctypes.windll.user32
    parts = [p.strip().lower() for p in (combo or "").split("+") if p.strip()]
    if not parts:
        return False, "没指定键"
    vks = []
    for p in parts:
        if p not in VK:
            return False, f"不认识的键: {p}"
        vks.append(VK[p])
    try:
        for v in vks:
            u.keybd_event(v, 0, 0, 0)
            _t.sleep(0.03)
        for v in reversed(vks):
            u.keybd_event(v, 0, 2, 0)
            _t.sleep(0.03)
    except Exception as e:
        return False, f"按键失败: {type(e).__name__}: {e}"
    return True, f"已按键: {combo}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(send(" ".join(sys.argv[1:])))
    else:
        print(f"命令文件: {cmd_path()}")
        print(f"状态文件: {state_path()}")
        print(f"游戏在跑: {is_game_running()}")
        print(f"上次状态: {last_state() or '(空)'}")
