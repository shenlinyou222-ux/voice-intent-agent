# -*- coding: utf-8 -*-
"""端到端测试：模型 + 分句 + 动作（全部 dry-run，不会真的启动程序）

跑三组：
  ① 规则分句自测（不需要模型）—— 覆盖所有意图形态
  ② 模型意图识别 —— 在 val 集上算精度 + 混淆
  ③ 端到端 —— 模型判意图 + 规则切参数 + 动作（dry-run），算最终执行正确率
"""
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

from actions import Config, parse, resolve, dispatch
from model_runner import IntentModel, read_wav

CFG_PATH = HERE / "intent_config.json"
JOBS = HERE / "intent_dataset.jsonl"
WAV = HERE / "wavs"
OUT = HERE / "test_report.json"

cfg = Config(CFG_PATH)
fails = []


# ══════════════════════════════════════════════════════════
def t1_rules():
    print("=" * 80)
    print("① 规则分句自测（纯规则，不需要模型）")
    print("=" * 80)
    cases = [
        ("启动记事本", "launch_app", "记事本"),
        ("打开任务管理器", "launch_app", "任务管理器"),
        ("运行Blender", "launch_app", "Blender"),
        ("打开B站", "open_url", "B站"),
        ("打开B站网页", "open_url", "B站"),
        ("上知乎", "open_url", "知乎"),
        ("打开CUDA编程指南文件", "open_file", "CUDA编程指南"),
        ("打开并行规约", "open_file", "并行规约"),
        ("打开github.com", "open_web", "github.com"),
        ("打开一个不存在的东西", "open_web", "一个不存在的东西"),
        ("搜索python教程", "web_search", "python教程"),
        ("搜一下resnet50", "web_search", "resnet50"),
        ("搜索文件季度报告", "file_search", "季度报告"),
        ("找文件实验数据", "file_search", "实验数据"),
        ("精确查找resnet50", "exact_search", "resnet50"),
        ("精确搜索bert模型", "exact_search", "bert模型"),
        ("今天天气怎么样", "unknown", "今天天气怎么样"),
        ("打开", "unknown", "打开"),
        ("搜索", "unknown", "搜索"),
    ]
    ok = 0
    for text, want_i, want_a in cases:
        it, arg = parse(text, cfg)
        good = (it == want_i and arg == want_a)
        ok += int(good)
        mark = "✅" if good else "❌"
        if not good:
            fails.append(f"规则: {text!r} 得到 ({it},{arg!r})，期望 ({want_i},{want_a!r})")
        print(f"  {mark} {text:26s} → {it:14s} arg={arg!r}")
    print(f"\n  规则分句 {ok}/{len(cases)} 通过")
    return ok / len(cases)


# ══════════════════════════════════════════════════════════
def t2_model():
    print()
    print("=" * 80)
    print("② 模型意图识别（val 集）")
    print("=" * 80)
    model = IntentModel(model_dir=HERE / "model", device="cpu")
    jobs = [json.loads(l) for l in io.open(JOBS, encoding="utf-8") if l.strip()]
    val = [j for j in jobs if j["split"] == "val"]
    print(f"  val {len(val)} 条，{len(model.INTENTS)} 类")

    per = defaultdict(lambda: [0, 0])
    conf = Counter()
    texts_ok = 0
    for n, j in enumerate(val):
        wav = read_wav(str(WAV / f"{j['id']}.wav"))
        r = model.predict(wav)
        per[j["intent"]][0] += 1
        per[j["intent"]][1] += int(r["intent"] == j["intent"])
        if r["intent"] != j["intent"]:
            conf[(j["intent"], r["intent"])] += 1
        texts_ok += int(r["text"] == j["text"])
        if (n + 1) % 200 == 0:
            print(f"    {n+1}/{len(val)}", flush=True)

    tot = sum(v[0] for v in per.values())
    ok = sum(v[1] for v in per.values())
    print()
    for it in sorted(per):
        a, b = per[it]
        print(f"    {it:14s} {b:>4}/{a:<4} = {b/a*100:6.2f}%")
    print(f"\n  总体 {ok}/{tot} = {ok/tot*100:.2f}%")
    print(f"  CTC 转写完全正确 {texts_ok}/{len(val)} = {texts_ok/len(val)*100:.2f}%")
    if conf:
        print(f"\n  混淆 top 8:")
        for (a, b), c in conf.most_common(8):
            print(f"    {a:14s} → {b:14s} {c}")
    return model, ok / tot, per


# ══════════════════════════════════════════════════════════
def t3_e2e(model):
    print()
    print("=" * 80)
    print("③ 端到端：模型判意图 + 规则切参数 + 动作（dry-run）")
    print("=" * 80)
    jobs = [json.loads(l) for l in io.open(JOBS, encoding="utf-8") if l.strip()]
    val = [j for j in jobs if j["split"] == "val"]

    ok = 0
    bad = []
    for n, j in enumerate(val):
        wav = read_wav(str(WAV / f"{j['id']}.wav"))
        r = model.predict(wav)
        intent, text = r["intent"], r["text"]

        # 门控：精确查找必须含"精确"
        kw = None
        for it in cfg.raw["intents"]:
            if it["id"] == intent:
                kw = it.get("gate_keyword")
        if kw and kw not in text:
            intent = "unknown"

        rule_intent, rule_arg = parse(text, cfg)
        # ⭐ 实际用的是 resolve：规则拿不到就按模型给的意图去闭集 snap
        arg = resolve(intent, text, cfg)
        # 期望：模型意图 == 数据标签，且动作能执行成功（或该拒识）
        want = j["intent"]
        if want == "unknown":
            good = (intent == "unknown")
        else:
            good = (intent == want)
            if good:
                aok, msg, _ = dispatch(intent, arg, cfg, dry=True)
                good = aok
        ok += int(good)
        if not good and len(bad) < 15:
            bad.append((j["text"], text, want, intent, arg))
        if (n + 1) % 200 == 0:
            print(f"    {n+1}/{len(val)}", flush=True)

    print(f"\n  端到端执行正确 {ok}/{len(val)} = {ok/len(val)*100:.2f}%")
    if bad:
        print(f"\n  错例（前 15 条）:")
        for ref, hyp, want, got, arg in bad:
            print(f"    说 {ref}")
            print(f"    转 {hyp}   期望 {want}  得到 {got}  arg={arg!r}")
    return ok / len(val)


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    r1 = t1_rules()
    model, r2, per = t2_model()
    r3 = t3_e2e(model)

    print()
    print("=" * 80)
    print("  汇总")
    print("=" * 80)
    print(f"  ① 规则分句        {r1*100:6.2f}%")
    print(f"  ② 模型意图识别    {r2*100:6.2f}%")
    print(f"  ③ 端到端执行      {r3*100:6.2f}%")
    if fails:
        print(f"\n  ⚠️ 规则层失败 {len(fails)} 项:")
        for f in fails[:10]:
            print(f"    {f}")
    json.dump({"rules": r1, "model": r2, "e2e": r3,
               "per_intent": {k: v[1] / v[0] for k, v in per.items()}},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n-> {OUT}")
    sys.exit(0 if (r1 > 0.95 and r2 > 0.9 and r3 > 0.9) else 1)
