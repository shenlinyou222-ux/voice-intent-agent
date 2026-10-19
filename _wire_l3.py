import json
from pathlib import Path
root = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
a = root / "actions.py"; t = a.read_text(encoding="utf-8")
if "_resolve_closed_pre_link" not in t:
    t += '''

# ══════════════════════════════════════════════════════════════════════
# L3 实体链接兜底（包一层 resolve_closed）
#   为什么放在这里：resolve_closed 是"参数 -> 闭集实体"的最后一关，
#   它返回 None 就意味着要拒识/乱开。实测血债：「直呼」「记事板」
#   「第四个」「知乎网」「Deep Seek」这类 ASR 近音/近形/经验误听，
#   靠原来的编辑距离会**乱吸**（豆瓣->豆包）或**吸不到**（知乎网）。
#   现在：原来命中就用原来的 ✓；没命中才走四路融合的实体链接，
#   分数 >= entity_link_thr（默认 0.75）才接受，否则仍然拒识。
_RESOLVE_CLOSED_ORIG = resolve_closed


def resolve_closed(intent, arg, cfg, *a, **kw):
    r = _RESOLVE_CLOSED_ORIG(intent, arg, cfg, *a, **kw)
    if r:
        return r
    try:
        from entity_link import link
        table = {"launch_app": cfg.apps, "open_url": cfg.urls,
                 "open_file": cfg.files}.get(intent)
        if not table:
            return None
        pairs, back = [], {}
        for e in table:
            for nm in [e.get("name", "")] + [str(x) for x in (e.get("aliases") or [])]:
                if nm:
                    pairs.append((nm, e.get("name")))
                    back[nm] = e
        name, score, how = link(
            str(arg or ""), pairs,
            thr=float(cfg.behavior.get("entity_link_thr") or 0.75))
        if name and name in back:
            try:
                from dlog import log
                log(f"   🔗 实体链接：{arg!r} -> {name}（{how} {score:.2f}）", "INT")
            except Exception:
                pass
            return back[name]
    except Exception:
        pass
    return None
'''
    a.write_text(t, encoding="utf-8"); print("  ✅ resolve_closed 已加实体链接兜底")
p = root / "intent_config.json"; d = json.loads(p.read_text(encoding="utf-8")); b = d["behavior"]
b["entity_link_thr"] = 0.75
b["_comment_entity_link"] = ("L3 实体链接（entity_link.py 四路融合：归一化精确/拼音相同/"
    "拼音差1(名字>=3字)/编辑距离(>=3字或后缀字)/经验误听表/可选语义余弦）。"
    "接在 resolve_closed 兜底：原来命中优先，未命中才走链接，分数<阈值仍拒识。")
p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("  entity_link_thr =", b["entity_link_thr"])
