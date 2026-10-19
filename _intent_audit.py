# -*- coding: utf-8 -*-
"""意图识别体检：拿 utt/index.jsonl 里【真实麦克风】的存档，看它在你的声音上错在哪。

为什么必须看这个而不是 val 语料：val 是 TTS 干净音频，端到端 97.5%；
真实麦克风有识别错字 + 说话方式差异，才是意图识别真正的战场。

判据（用生产同款闭集/触发词，不另起一套）：
  · 漏判：文本里【同时】有触发词和闭集名（= 规则本来能精确解），却被判 unknown
          —— 这是最该修的一类，因为它不是"模型不确定"，是规则没接住。
  · 误执行：accepted=True 但置信度低（<0.75）—— 可能把噪声/非命令当成命令执行了。
  · 拒识：accepted=False 且 intent != unknown（动作执行失败/门禁拦下）。
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from actions import Config  # noqa: E402

TOPN = int(sys.argv[1]) if len(sys.argv) > 1 else 25


def load_rows():
    rows = []
    for line in (HERE / "utt" / "index.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("src") == "mic":
            rows.append(r)
    return rows


def main():
    cfg = Config(None)
    # 触发词 + 闭集名（含别名）—— 与生产 plausible()/规则层同一份来源
    triggers = set()
    for v in cfg.triggers.values():
        triggers.update(v)
    names = set()
    for group in (cfg.apps, cfg.urls, cfg.files):
        for e in group:
            names.add(e["name"])
            names.update(e.get("aliases", []))
    tl_trig = sorted(triggers, key=len, reverse=True)

    rows = load_rows()
    print(f"\n  ══ 真实麦克风存档 {len(rows)} 条 ══")
    days = Counter(r["ts"][:10] for r in rows)
    print("  按天：" + "  ".join(f"{d}:{n}" for d, n in sorted(days.items())))
    by_int = Counter(r.get("intent", "?") for r in rows)
    print(f"\n  意图分布：")
    for k, n in by_int.most_common():
        acc = sum(1 for r in rows if r.get("intent") == k and r.get("accepted"))
        print(f"     {k:<14} {n:>4} 条   其中 accepted={acc}（{acc/max(1,n)*100:.0f}%）")

    # ── 漏判：有触发词 + 有闭集名，却判 unknown ──
    miss = []
    for r in rows:
        t = (r.get("text") or "").strip()
        if not t or r.get("intent") != "unknown":
            continue
        hit_trig = [x for x in tl_trig if x and x in t]
        hit_name = [x for x in names if x and x.lower() in t.lower()]
        if hit_trig and hit_name:
            miss.append((r, hit_trig[:2], hit_name[:2]))
    print(f"\n  ══ ⚠️ 漏判（有触发词 {len(triggers)} 个 / 闭集名 {len(names)} 个，本该规则直通，却判 unknown）"
          f"：{len(miss)} 条 ══")
    for r, tr, nm in miss[-TOPN:]:
        print(f"     {r['ts'][5:]}  conf={r.get('conf', 0):.2f}  「{r['text'][:30]}」"
              f"   触发={tr} 闭集={nm}")

    # ── 低置信却执行了 ──
    low = [r for r in rows if r.get("accepted") and float(r.get("conf") or 0) < 0.75]
    print(f"\n  ══ ⚠️ 低置信但执行了（conf<0.75）：{len(low)} 条 ══")
    for r in low[-TOPN:]:
        print(f"     {r['ts'][5:]}  {r.get('intent')}  conf={r.get('conf'):.2f}"
              f"  arg={r.get('arg')!r}  「{r.get('text','')[:28]}」")

    # ── 拒识/失败（intent 不是 unknown 但没执行）──
    rej = [r for r in rows if not r.get("accepted") and r.get("intent") != "unknown"]
    print(f"\n  ══ 拒识/执行失败（判出意图但没执行）：{len(rej)} 条 ══")
    c = Counter(r.get("intent") for r in rej)
    print("     按意图：" + "  ".join(f"{k}:{v}" for k, v in c.most_common()))
    for r in rej[-min(12, TOPN):]:
        print(f"     {r['ts'][5:]}  {r.get('intent')}  conf={r.get('conf'):.2f}  "
              f"arg={r.get('arg')!r}  「{r.get('text','')[:28]}」")

    # ── 执行了的都是什么（看规则层接住了多少）──
    ex = [r for r in rows if r.get("accepted")]
    print(f"\n  ══ ✅ 真执行了 {len(ex)} 条，按意图 ══")
    for k, n in Counter(r.get("intent") for r in ex).most_common():
        print(f"     {k:<14} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
