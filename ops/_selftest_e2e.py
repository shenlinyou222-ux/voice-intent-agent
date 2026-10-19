# -*- coding: utf-8 -*-
"""端到端：两段式对话（简化版 —— 不做指代消解，交给 AI 自己跑命令）。

    ① 「查集群」              → 规则直通 → 拿到 pod 列表（缓存进 _LAST_OUT）
    ② 「看下第3个error的pod」  → ★ 直接升级 AI（不做引用解析）
    ③ 「让ds看看第1个error的」 → ★ 直接升级 AI
    ④ 「随便说句话」           → 拒识
"""
import sys
from pathlib import Path

VA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VA))

from ops import game_actions as ga, game_profile as gp


class Cfg:
    def __init__(self, raw):
        self.raw = raw
        self.behavior = {}
        self.path = None


prof = gp.load("ops-v1")
cfg = Cfg(prof.raw)

print("=" * 70)
print("第 ① 步：说「查集群」  → 规则直通 0ms")
print("=" * 70)
i, a, tier = gp.route("查集群", prof)
print(f"  路由：tier={tier}  intent={i}")
ok, msg, info = ga.GAME_ACTIONS[i](a, cfg, dry=False)
print(f"  执行：ok={ok}  {len(ga._LAST_OUT['out'].splitlines())} 行输出已缓存（只作线索）")
print()

print("=" * 70)
print("第 ② 步：说「看下第3个error的pod」  → 直接升级 AI")
print("=" * 70)
i2, a2, tier2 = gp.route("看下第3个error的pod", prof)
print(f"  路由：tier={tier2}  intent={i2}  arg={a2!r}")
ok2, msg2, info2 = ga.GAME_ACTIONS[i2](a2, cfg, dry=True)
print(f"  {msg2}")
print()

print("=" * 70)
print("第 ③ 步：看会拼进 DSH 任务的完整文本")
print("=" * 70)
b = cfg.behavior
ws = str(b.get("dsh_workspace", r"C:\Users\user\Desktop\ds harness"))
head = (
    "我在做运维排查，请帮我分析。\n"
    f"工作区：{ws}\n"
    "环境：Windows + WSL(Ubuntu-24.04)。命令都在 WSL 里跑，"
    "kubectl / nvidia-smi / systemctl / journalctl / ss / df 都可用；"
    "k3s 集群是本机的（单节点 desktop-a00thlv）。\n"
    "★ 如果需要集群或系统现状，**请自己重新执行命令拿最新数据**"
    "（例如 `kubectl get pods -A -o wide`、`kubectl describe pod <name>`、"
    "`kubectl logs <name> --tail=100 -p`），**不要依赖我下面贴的输出** —— "
    "那是我之前跑的，可能已经过期。\n"
    "★ Prometheus 在 http://127.0.0.1:30090（NodePort，**必须从 WSL 里 curl**，"
    "Windows 上访问 localhost 会连接被拒）。\n"
)
ctx = (f"\n【线索：我最近跑过的命令（可能已过期，请以你自己跑的结果为准）】\n"
       f"意图：{ga._LAST_OUT.get('intent')}\n"
       f"命令：{ga._LAST_OUT.get('cmd')}\n"
       f"```\n{ga._LAST_OUT['out'][:400]}\n```\n")
for ln in (head + "\n我的诉求：看下第3个error的pod\n" + ctx).splitlines():
    print("  │ " + ln)
print()

print("=" * 70)
print("第 ④ 步：边界 —— 不该触发的")
print("=" * 70)
for t in ("随便说句话", "你好", "今天天气怎么样"):
    i4, a4, tier4 = gp.route(t, prof)
    print(f"  {t:<16} → tier={tier4}  intent={i4}")
print()
print("  回归：规则层不该被抢的")
for t, want in (("查集群", "ops_k8s_pods"), ("查磁盘", "ops_disk"),
                ("这个pod怎么挂了", "ops_k8s_describe"), ("重启nginx", "ops_restart_service")):
    i5, a5, tier5 = gp.route(t, prof)
    mark = "✓" if i5 == want else "✗"
    print(f"  {mark} {t:<16} → {i5}  (期望 {want})  tier={tier5}")
