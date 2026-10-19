# -*- coding: utf-8 -*-
"""ops 域【第②档 embedding 路由】实测。

关键：这些说法**故意不在闭集/短语表里** —— 只有 embedding kNN 能兜住。
如果它们命中了 rule 档，说明闭集写得太宽；命中 embed 档才是我们要的。
"""
import sys
from pathlib import Path

VA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VA))

from ops import game_profile as gp

prof = gp.load("ops-v1")
print(f"profile = {prof.profile_id}   index_version = {prof.raw.get('index_version')}")
print(f"  index_dim={prof.raw.get('index_dim')} index_samples={prof.raw.get('index_samples')}")
print()

# 加载索引（走生产同源的 EmbedIntent）
idx_path = VA / "ops" / "profiles" / "ops-v1" / "index.json"
embed_cls = None
try:
    from intent_embed import EmbedIntent
    embed_cls = EmbedIntent(
        url="http://172.20.71.101:30802/v1/embeddings",
        index=str(idx_path),
        thr=0.45, margin=0.0,
    )
    print(f"  ✅ EmbedIntent 加载成功：{len(embed_cls.labels)} 示例 / {len(embed_cls.intents)} 类 / thr={embed_cls.thr}")
except Exception as e:
    print(f"  ❌ EmbedIntent 加载失败：{type(e).__name__}: {e}")
    raise SystemExit(1)
print()

# 故意用【不在闭集里】的说法 —— 只有 embedding 能兜
CASES = [
    ("硬盘空间够吗",         "ops_disk"),
    ("存储还够用吗",         "ops_disk"),
    ("内存还有多少",         "ops_mem"),
    ("内存是不是不够了",      "ops_mem"),
    ("机器忙不忙",           "ops_cpu"),
    ("显卡现在什么情况",      "ops_gpu"),
    ("显存紧张吗",           "ops_gpu"),
    ("容器都正常吗",         "ops_k8s_pods"),
    ("有没有挂掉的容器",      "ops_k8s_pods"),
    ("最近日志有啥",         "ops_logs"),
    ("谁在占内存",           "ops_proc"),
    ("哪些端口开着",         "ops_ports"),
    ("首token要等多久",      "ops_metric"),
    ("生成快不快",           "ops_metric"),
    ("缓存用了多少",         "ops_metric"),
    ("把监控打开",           "ops_open"),
    ("重新启动一下nginx",     "ops_restart_service"),
]

print("=== 第②档 embedding 路由 ===")
ok = bad = 0
for text, want in CASES:
    intent, arg, tier = gp.route(text, prof, embed_cls=embed_cls)
    good = (intent == want)
    mark = "✓" if good else "✗"
    if good: ok += 1
    else: bad += 1
    print(f"  {mark} {text:<20} → tier={str(tier):<6} intent={str(intent):<22} 期望 {want}")

print()
print(f"通过 {ok} / {ok+bad}")

print()
print("=== 回归：规则层仍优先（不该被 embedding 抢）===")
for text, want_tier in [("查磁盘", "rule"), ("查集群", "rule"), ("重启nginx", "rule"),
                        ("让ds看看这个yaml", "ai"), ("删除所有pod", None)]:
    intent, arg, tier = gp.route(text, prof, embed_cls=embed_cls)
    mark = "✓" if tier == want_tier else "✗"
    print(f"  {mark} {text:<20} → tier={tier}  intent={intent}  arg={arg!r}  （期望 tier={want_tier}）")
