# -*- coding: utf-8 -*-
"""ops 域三档路由的功能自测（不联网、不执行命令）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # voice_agent/
sys.path.insert(0, str(Path(__file__).resolve().parent))          # ops/

from ops import game_profile as gp

prof = gp.load("ops-v1")
print(f"profile = {prof.profile_id}")
print(f"  意图 {len(prof.intents)} | items {len(prof.items())} | 闭集候选 {len(prof._cands)} | 热词 {len(prof.hotwords())}")
print()

CASES = [
    # (说法, 期望 tier, 期望意图)
    ("查磁盘",              "rule", "ops_disk"),
    ("磁盘还剩多少",         "rule", "ops_disk"),
    ("空间够不够",           "rule", "ops_disk"),
    ("查内存",              "rule", "ops_mem"),
    ("查显卡",              "rule", "ops_gpu"),
    ("显存用了多少",         "rule", "ops_gpu"),
    ("查集群",              "rule", "ops_k8s_pods"),
    ("查所有pod",           "rule", "ops_k8s_pods"),
    ("查节点",              "rule", "ops_k8s_nodes"),
    ("查TTFT",             "rule", "ops_metric"),
    ("查TPOT",             "rule", "ops_metric"),
    ("查KV缓存",            "rule", "ops_metric"),
    ("查march7的TTFT",      "rule", "ops_metric"),
    ("打开面板",            "rule", "ops_open"),
    ("重启nginx",           "rule", "ops_restart_service"),
    ("确认",               "rule", "ops_confirm"),
    ("取消",               "rule", "ops_cancel"),
    # 拼音滑窗（同音字）—— ★ 不靠手写表，靠拼音键
    ("察磁盘",              "rule", "ops_disk"),
    ("查现卡",              "rule", "ops_gpu"),      # 现/显 同音
    ("查机群",              "rule", "ops_k8s_pods"), # 机/集 近音
    ("差磁盘",              "rule", "ops_disk"),     # 差/查 同音
    # ★ 误伤回归：不做逐字替换，这句必须还能命中短语表
    ("谁在占内存",           "rule", "ops_proc"),
    # AI 升级（显式触发词）
    ("让ds看看这个yaml",     "ai",   "ops_ask_ai"),
    ("帮我排查一下",          "ai",   "ops_ask_ai"),
    # ★ 分层原则：规则给事实、AI 给分析
    #   「这个 pod 怎么挂了」→ 规则层 describe 直接给 Events（比 AI 快且确定）
    #   「让 ds 看看这个 pod 怎么挂了」→ 显式求助 → 升级 AI
    ("这个pod怎么挂了",       "rule", "ops_k8s_describe"),
    ("让ds看看这个pod怎么挂了", "ai",  "ops_ask_ai"),
    # 危险动词 → 拒识
    ("删除所有pod",          None,   None),
    ("清空日志",             None,   None),
    ("rm -rf /",           None,   None),
    # 无关话 → 拒识
    ("今天天气怎么样",         None,   None),
    ("你好",                None,   None),
]

ok = bad = 0
for text, want_tier, want_intent in CASES:
    intent, arg, tier = gp.route(text, prof, embed_cls=None)   # 不联网，只测 ①③④
    good = (tier == want_tier) and (intent == want_intent)
    mark = "✓" if good else "✗"
    if good: ok += 1
    else: bad += 1
    extra = f"  arg={arg!r}" if arg else ""
    print(f"  {mark} {text:<20} → tier={str(tier):<5} intent={str(intent):<20}{extra}")
    if not good:
        print(f"      期望 tier={want_tier} intent={want_intent}")

print()
print(f"通过 {ok} / {ok+bad}")
if bad:
    print(f"⚠️ {bad} 条不符")

# ══════════════════════════════════════════════════════════════════
#  ★ 拼音滑窗的边界测试（直接测 _find）
# ══════════════════════════════════════════════════════════════════
print()
print("=== 拼音滑窗边界（_find）===")
from ops.game_profile import _find, _py_key, _PY_MIN, _has_pypinyin

print(f"  pypinyin 可用 = {_has_pypinyin()}   拼音门槛 _PY_MIN = {_PY_MIN}")

BOUND = [
    # (text, cand, 期望下标, 说明)
    ("察磁盘",     "查磁盘",  0,  "3 字同音 → 拼音命中"),
    ("差磁盘",     "查磁盘",  0,  "3 字同音 → 拼音命中"),
    ("查机群",     "查集群",  0,  "近音 → 拼音命中"),
    ("谁在占内存",  "谁在占内存", 0, "字面命中"),
    ("茶",         "查",     -1, "★ 单字候选【不做】拼音（门槛拦截）"),
    ("差错",       "查",     -1, "★ 单字候选不做拼音（差/查 同音但被门槛拦）"),
    ("察磁盘",     "磁盘",    1, "★ 2 字候选【可以】拼音命中（察/磁 音不同，靠滑窗定位）"),
    ("查磁盘",     "磁盘",    1,  "子串命中（非开头）"),
    ("完全无关的话", "查磁盘",  -1, "无关 → 不命中"),
]
bok = bbad = 0
for text, cand, want, note in BOUND:
    got = _find(text, cand)
    good = (got == want)
    if good: bok += 1
    else: bbad += 1
    print(f"  {'✓' if good else '✗'} _find({text!r}, {cand!r}) = {got}  （期望 {want}） {note}")
print(f"  通过 {bok} / {bok+bbad}")
if _has_pypinyin():
    print(f"  拼音键示例：查磁盘={_py_key('查磁盘')}  察磁盘={_py_key('察磁盘')}  差磁盘={_py_key('差磁盘')}")
else:
    print("  ⚠️ 当前 python 没装 pypinyin —— 拼音用例会退化（生产用 E:\\voice_agent_build 的 python）")
