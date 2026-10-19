# -*- coding: utf-8 -*-
"""ops 域【真实执行】冒烟测试 —— 只跑只读命令，不碰任何变更动作。"""
import sys
from pathlib import Path

VA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VA))

from ops import game_actions as ga
from ops import render, window


class FakeCfg:
    def __init__(self, prof_raw):
        self.raw = prof_raw
        self.behavior = {}
        self.path = None


from ops import game_profile as gp
prof = gp.load("ops-v1")
cfg = FakeCfg(prof.raw)

render.title("真实执行冒烟（只读，dry=False）")

CASES = [
    ("ops_disk",  "", ga.act_ops_disk),
    ("ops_mem",   "", ga.act_ops_mem),
    ("ops_cpu",   "", ga.act_ops_cpu),
    ("ops_gpu",   "", ga.act_ops_gpu),
    ("ops_k8s_pods",  "", ga.act_ops_k8s_pods),
    ("ops_k8s_nodes", "", ga.act_ops_k8s_nodes),
    ("ops_metric", "ttft", ga.act_ops_metric),
    ("ops_metric", "kvcache", ga.act_ops_metric),
    ("ops_metric", "ttft:embed", ga.act_ops_metric),
]

for name, arg, fn in CASES:
    print()
    render.title(f"{name} {arg}".strip())
    try:
        ok, msg, info = fn(arg, cfg, dry=False)
        mark = "✓" if ok else "✗"
        print(f"  {mark} ok={ok}  msg={str(msg)[:150]}")
    except Exception as e:
        print(f"  ✗ EXC {type(e).__name__}: {e}")

print()
render.title("安全边界复查（不该执行）")
for name, arg, fn in [
    ("ops_restart_service", "nginx", ga.act_ops_restart_service),
    ("ops_confirm", "", ga.act_ops_confirm),
]:
    ok, msg, info = fn(arg, cfg, dry=False)
    print(f"  {name}({arg!r}) → ok={ok}  {msg}")

print()
render.title("AI 升级（dry，不真起会话）")
ok, msg, info = ga.act_ops_ask_ai("看看这个 yaml", cfg, dry=True)
print(f"  {ok}  {msg}")
