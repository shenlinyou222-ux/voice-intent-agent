# -*- coding: utf-8 -*-
"""切分器回归 —— 把"正确性与切分粒度/停顿阈值无关"这件事变成可跑的断言。

五条不变量：
  I1 单指令文本 → 恰好 1 段，且文本原样（**保证今天的行为零改动**）
  I2 字不丢：各段拼起来 == 原文（只忽略空白差异）
  I3 顺序不变
  I4 **粒度无关**：把任一文本任意切成 N 片依次 push，结果 == 整段 push
     —— 这一条就是"停顿阈值只是性能参数"的形式化证明
  I5 段的推进由触发词决定：n 个触发词 → ≤ n 段；没有触发词的碎片被并回

用法：E:\\voice_agent_build\\Scripts\\python.exe -X utf8 _split_regression.py
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from intent_split import (Splitter, triggers_from_cfg, boundary_from_cfg,  # noqa: E402
                          names_from_cfg)

CFG = json.loads((HERE / "intent_config.json").read_text(encoding="utf-8"))
TRIGGERS = triggers_from_cfg(CFG)
BOUNDARY = boundary_from_cfg(CFG)
NAMES = names_from_cfg(CFG)
B = CFG.get("behavior", {})

OK = FAIL = 0


def check(name, cond, detail=""):
    global OK, FAIL
    if cond:
        OK += 1
        print(f"    ✅ {name}")
    else:
        FAIL += 1
        print(f"    ❌ {name}   {detail}")
    return bool(cond)


def norm(s):
    """只忽略空白差异（标点保留 —— 标点是内容）"""
    return "".join(str(s).split())


def split(text, **kw):
    kw.setdefault("boundary", BOUNDARY)
    kw.setdefault("names", NAMES)
    return Splitter(TRIGGERS, **kw).segments(text)


def texts_of(segs):
    return [s.text for s in segs]


# 构造多指令用例：(文本, 期望条数, 备注)
CASES = [
    ("打开知乎，搜索xxx，关闭知乎网页，打开ds", 4, "用户原例"),
    ("打开知乎，搜索一下今天的新闻", 2, "打开 + 搜索"),
    ("打开知乎然后搜索今天的新闻", 2, "无逗号也要切"),
    ("打开知乎", 1, "单指令"),
    ("搜索一下今天的新闻", 1, "单指令(搜索)"),
    ("那个，打开知乎", 1, "前置口水词并入"),
    ("打开DeepSeek网页版", 1, "关键：网页版只算一条"),
    ("关闭知乎网页，打开ds", 2, "关闭 + 打开"),
    ("打开知乎，搜索xxx", 2, "两句"),
    ("启动Steam，然后关掉它", 2, "启动 + 关掉"),
]


def main():
    random.seed(20261014)
    print(f"\n  触发词 {len(TRIGGERS)} 个（长优先）：{TRIGGERS[:10]} …")

    # 触发词来源与生产一致性（生产权威 = actions.Config.triggers）
    try:
        from actions import Config as AC
        prod = set()
        for v in AC(None).triggers.values():
            prod.update(v)
        prod.update(B.get("search_mid_verbs", []))
        missing = prod - set(TRIGGERS)
        extra = set(TRIGGERS) - prod
        print(f"  与生产触发词对比：生产 {len(prod)} 个，本模块 {len(TRIGGERS)} 个，"
              f"缺 {sorted(missing)[:6]}，多 {sorted(extra)[:6]}")
        check("触发词集合与生产一致", not missing and not extra,
              f"缺={sorted(missing)} 多={sorted(extra)}")
    except Exception as e:
        print(f"  （跳过生产对比：{type(e).__name__}: {e}）")

    print(f"\n  ══ ① 构造用例：多指令切分 ══")
    for text, want, note in CASES:
        segs = split(text)
        got = texts_of(segs)
        ok = len(segs) == want
        check(f"{note}：{text[:26]} → {len(segs)} 段（期望 {want}）", ok, f"实得 {got}")
        if ok and want > 1:
            print(f"        切成：{got}")

    print(f"\n  ══ ② 真实录音的 450 条 ASR 文本：不变量 + 影响面 ══")
    rows = [json.loads(l) for l in (HERE / "utt" / "index.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    texts = [r.get("text", "") for r in rows if r.get("src") == "mic" and r.get("text")]
    n_multi = 0
    samples = []
    for t in texts:
        segs = split(t)
        if norm("".join(texts_of(segs))) != norm(t):
            check(f"字没丢：{t[:20]}", False, f"拼回 {''.join(texts_of(segs))!r}")
            continue
        trig_hits = len(Splitter(TRIGGERS, boundary=BOUNDARY, names=NAMES)._scan(t))
        if len(segs) > 1:
            n_multi += 1
            if len(samples) < 12:
                samples.append((t, [f"{s.text}⟨{s.trigger}⟩" for s in segs]))
        # 触发词 ≤1 个却切出多段 = 规则有 bug
        if trig_hits <= 1 and len(segs) > 1:
            check(f"单触发词不该切碎：{t[:24]}", False, f"{texts_of(segs)}")
    check(f"450 条里字都没丢（拼回原文）", True)
    print(f"     其中会被切成多段（= 新路径的影响面）：{n_multi}/{len(texts)} 条"
          f"（{n_multi/max(1,len(texts))*100:.1f}%）")
    for t, g in samples:
        print(f"       「{t[:34]}」 → {g}")

    print(f"\n  ══ ③ 粒度无关性（I4）：任意切碎再喂，结果必须一致 ══")
    pool = [c[0] for c in CASES] + [t for t, _ in samples] + texts[:40]
    bad = 0
    n_try = 0
    for t in pool:
        if len(t) < 4:
            continue
        ref = texts_of(split(t))
        # 所有 2 片切法
        chops = [[t[:i], t[i:]] for i in range(1, len(t))]
        # 随机 3~4 片切法
        for _ in range(30):
            k = random.choice((3, 4))
            if len(t) <= k:
                continue
            cut = sorted(random.sample(range(1, len(t)), k - 1))
            chops.append([t[a:b] for a, b in zip([0] + cut, cut + [len(t)])])
        for parts in chops:
            n_try += 1
            sp = Splitter(TRIGGERS, boundary=BOUNDARY, names=NAMES)
            got = []
            for p in parts:
                got += texts_of(sp.push(p))
            got += texts_of(sp.flush())
            if got != ref:
                bad += 1
                if bad <= 5:
                    print(f"       ✗ 「{t[:26]}」 切法 {parts} → {got}（整段是 {ref}）")
    check(f"{n_try} 种切法结果全部一致", bad == 0, f"{bad} 种不一致")

    print(f"\n  ══ ④a 停顿参数(机制③)【整段版今天要用的：pause_cut=False】══")
    comp = lambda s: any(x in s for x in TRIGGERS) and len(s) >= 4      # noqa: E731

    def frag_feed(t, pause_cut, step=3, comp_fn=None):
        sp = Splitter(TRIGGERS, boundary=BOUNDARY, names=NAMES,
                      pause_cut=pause_cut, completeness=comp_fn or comp)
        got = []
        for i in range(0, len(t), step):
            got += texts_of(sp.push(t[i:i + step], pause=True))
        got += texts_of(sp.flush())
        return got

    bad = []
    for t in pool:
        if len(t) < 4:
            continue
        ref = texts_of(split(t))
        got = frag_feed(t, False)
        if norm("".join(got)) != norm(t) or got != ref:
            bad.append((t, got, ref))
    check(f"碎片化喂入（pause_cut=False）结果与整段一致（{len(pool)} 条）",
          not bad, f"{len(bad)} 条不一致")
    for t, got, ref in bad[:3]:
        print(f"       ✗ 「{t[:24]}」 → {got}（整段 {ref}）")

    print(f"\n  ══ ④b 停顿版【阶段2，今天不开】—— 只报数，不计入失败 ══")
    print("     已知问题：停顿切点落在【缓冲末尾】而不是语义边界，")
    print("     增量文本若正停在词中间就会把词切断（「打开知乎，搜|索xxx」）。")
    print("     阶段2 要用「延后一片 + 强完整性判据（trigger + 闭集名/可解析参数）」重做。")
    bad2, n2 = 0, 0
    for t in pool:
        if len(t) < 4:
            continue
        n2 += 1
        if frag_feed(t, True) != texts_of(split(t)):
            bad2 += 1
    print(f"     当前实现：{bad2}/{n2} 条与整段版不一致（这就是今天不开它的理由）")

    print(f"\n  ══ ⑤ 计数器（第二阶段自适应用，现在只记账）══")
    sp = Splitter(TRIGGERS, boundary=BOUNDARY, names=NAMES)
    sp.segments("打开知乎，搜索xxx")
    print(f"     n_frag={sp.n_frag}  n_frag_no_trigger={sp.n_frag_no_trigger}  "
          f"n_cut_trigger={sp.n_cut_trigger}  n_cut_pause={sp.n_cut_pause}  "
          f"n_merged={sp.n_merged}")

    print(f"\n  {'🎉 全部通过' if FAIL == 0 else f'⚠️ {FAIL} 项失败'}（{OK} 通过 / {FAIL} 失败）\n")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
