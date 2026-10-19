# -*- coding: utf-8 -*-
"""ops 动作层集成自测（全部 dry-run，不执行任何真实命令）。"""
import sys
from pathlib import Path

VA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VA))

from ops import game_actions as ga
from ops import game_profile as gp
from ops import render, window

prof = gp.load("ops-v1")


class FakeCfg:
    """最小 cfg 桩：只带 raw（urls）和 behavior。"""
    def __init__(self, prof):
        self.raw = prof.raw
        self.behavior = {}
        self.path = None


cfg = FakeCfg(prof)

print("=== ① GAME_ACTIONS 注册表 ===")
print(f"  共 {len(ga.GAME_ACTIONS)} 个动作")
for k in sorted(ga.GAME_ACTIONS):
    print(f"    {k}")

print()
print("=== ② dry-run 每个动作（不执行真实命令）===")
ok = bad = 0
for name, fn in sorted(ga.GAME_ACTIONS.items()):
    # 给需要参数的意图补一个假参数
    arg = ""
    for i in prof.intents:
        if i["id"] == name and i.get("needs_arg"):
            arg = "ttft" if name == "ops_metric" else "nginx"
            break
    try:
        r = fn(arg, cfg, dry=True)
        good = isinstance(r, tuple) and len(r) >= 2
        mark = "✓" if good else "✗"
        if good: ok += 1
        else: bad += 1
        msg = str(r[1])[:76] if good else str(r)
        print(f"  {mark} {name:<22} {msg}")
    except Exception as e:
        bad += 1
        print(f"  ✗ {name:<22} EXC {type(e).__name__}: {e}")

print()
print(f"  通过 {ok} / {ok+bad}")

print()
print("=== ③ 安全机制 ===")
print(f"  黑名单 {len(prof.raw['safety']['blacklist'])} 词：{', '.join(prof.raw['safety']['blacklist'][:8])} …")
print(f"  需确认 {prof.raw['safety']['confirm_required']}")
for t in ("删除所有pod", "rm -rf /tmp", "清空日志", "查磁盘"):
    print(f"    {t:<14} 黑名单命中 = {prof.is_blacklisted(t)}")

print()
print("=== ④ 两步确认状态机 ===")
r1 = ga.act_ops_restart_service("nginx", cfg, dry=False)
print(f"  第一步：{r1[1]}")
print(f"  待确认状态：{ga._PENDING}")
r2 = ga.act_ops_cancel("", cfg, dry=False)
print(f"  取消：{r2[1]}   状态：{ga._PENDING}")

print()
print("=== ⑤ 渲染层（isatty / NO_COLOR）===")
import os
print(f"  color_on() = {render.color_on()}   (stdout 被重定向时为 False)")
render.render_kv("TTFT 测试", [("p50", "128 ms"), ("p90", "210 ms"), ("p99", "340 ms")])
render.render_log("error: connection refused\nwarn: retry 3\nCrashLoopBackOff\nRunning fine")

print()
print("=== ⑥ 窗口层可用性 ===")
print(f"  wt.exe 可用 = {window.available()}")
ok2, msg2 = window.show_pod_why("nginx-xxx", dry=True) if False else (True, "（未实际拉窗口）")
r = window.open_in_window("kubectl get pods", "test", hl="yaml", dry=True)
print(f"  dry: {r[1]}")
