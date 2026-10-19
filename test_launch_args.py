# -*- coding: utf-8 -*-
"""启动类动作的拼装回归：args 支持 + .cmd 必须经 cmd.exe /c。

为什么要钉：dsh.cmd 是 .cmd，Windows 上 CreateProcess 不能直接起 .cmd
（会报"不是有效的 Win32 应用程序"）；而"启动 dsh 会话"必须能带参数
（dsh web / dsh headless "任务" / --session-id 等）。
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import actions                                # noqa: E402
from actions import Config, act_launch_app    # noqa: E402

HERE = Path(__file__).parent
CAP = []


class FakePopen:
    def __init__(self, cmd, **kw):
        CAP.append(cmd)


actions.subprocess.Popen = FakePopen

DSH = (r"C:\Users\user\AppData\Local\Programs\DeepSeek Harness\resources"
       r"\runtime\cli\bin\dsh.cmd")
NOTEPAD = r"C:\WINDOWS\system32\notepad.exe"

cfg_raw = json.loads((HERE / "intent_config.json").read_text(encoding="utf-8"))
cfg_raw["apps"] += [
    {"name": "DSH会话", "aliases": ["dsh会话", "dsh"],
     "path": DSH, "args": ["web", "--no-open"]},
    {"name": "DSH任务", "aliases": ["dsh任务"],
     "path": DSH, "args": ["headless", "跑一下测试"]},
]
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                 encoding="utf-8") as f:
    json.dump(cfg_raw, f, ensure_ascii=False)
    tmp = f.name
CFG = Config(tmp)

CASES = [
    ("dsh会话", ["cmd.exe", "/c", "call", DSH, "web", "--no-open"]),
    ("dsh任务", ["cmd.exe", "/c", "call", DSH, "headless", "跑一下测试"]),
    ("记事本", [NOTEPAD]),
]


def main():
    bad = 0
    for arg, want in CASES:
        CAP.clear()
        ok, msg, info = act_launch_app(arg, CFG, dry=False)
        got = CAP[0] if CAP else None
        good = (got == want)
        if not good:
            bad += 1
        print(f"{'✅' if good else '❌'} {arg:10s} → {got}")
        if not good:
            print(f"     期望 {want}")
    n = len(CASES)
    print(f"\n{n - bad}/{n} 通过")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
