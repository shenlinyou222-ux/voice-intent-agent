# -*- coding: utf-8 -*-
"""建索引 + 用 split 字段做离线验证（含被砍掉的意图）。"""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
sys.path.insert(0, str(HERE))
from intent_embed import EmbedIntent

DROP = ["web_search"]          # ⭐ 用户决定：这类意图过时，砍掉
rows = [json.loads(l) for l in (HERE / "intent_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
rows = [r for r in rows if r["intent"] not in DROP]
tr = [r for r in rows if r.get("split") == "train"]
va = [r for r in rows if r.get("split") != "train"]
print("砍掉 %s 后：train=%d val=%d  剩余意图=%s" % (DROP, len(tr), len(va), sorted({r["intent"] for r in rows})))

ei = EmbedIntent(thr=0.30, margin=0.0)
t0 = time.time()
ei.build([(r["text"], r["intent"]) for r in tr])
print("建索引耗时 %.1fs  向量数=%d  dim=%d" % (time.time() - t0, len(ei.labels), ei.vecs.shape[1]))
p = ei.save(str(HERE / "intent_index.json"))
print("索引已存:", Path(p).name, "%.2f MB" % (Path(p).stat().st_size / 1e6))

# 验证
pred, lat = [], []
for r in va:
    d = ei.predict(r["text"]); pred.append(d["intent"]); lat.append(d["ms"])
acc = sum(1 for p_, r in zip(pred, va) if p_ == r["intent"]) / len(va)
print("\n  === 离线验证（val %d 条，kNN+阈值，零训练）===" % len(va))
print("  准确率 = %.4f     单句延迟 中位 %.0fms (含 HTTP)" % (acc, float(np.median(lat))))
for lab in sorted({r["intent"] for r in rows}):
    idx = [i for i, r in enumerate(va) if r["intent"] == lab]
    if idx: print("    %-14s n=%-4d recall=%.2f" % (lab, len(idx), sum(pred[i] == lab for i in idx) / len(idx)))
bad = [(va[i]["text"][:32], va[i]["intent"], pred[i]) for i in range(len(va)) if pred[i] != va[i]["intent"]][:10]
print("\n  错例（前 10）:")
for t, g, p_ in bad: print("    真=%-13s 判=%-13s 「%s」" % (g, p_, t))
