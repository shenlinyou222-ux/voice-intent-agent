# -*- coding: utf-8 -*-
"""三条路 A/B/C 对比：砍掉音频塔之前，先确认文本路真的不差。

同一个 val 集（intent_dataset.jsonl 的 split==val，609 条真实语料 wav + 文本）：
  A 纯文本路：B(文本) → head_text(768→8)          ← 砍掉音频塔后要走的路
  B 现状（音频+文本）：concat(c1 音频, c2 文本) → head(1536→8)
  C 纯音频路：只用 c1（音频塔 + ctc 软查表）→ 1536 头的音频半边（不可用，仅参考）
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model_runner import IntentModel, read_wav     # noqa: E402

HERE = Path(__file__).parent
MD = HERE / "model"


def main():
    rows = [json.loads(l) for l in
            open(HERE / "intent_dataset.jsonl", encoding="utf-8")]
    val = [r for r in rows if r.get("split") == "val"]
    val = [(r, HERE / "wavs" / (r["id"] + ".wav")) for r in val]
    val = [(r, p) for r, p in val if p.exists()]
    print(f"val 集：{len(val)} 条（有 wav）")

    print("\n加载 int8 模型（含 A/B/ctc，用于对比）...")
    m = IntentModel(model_dir=str(MD), device="cpu", int8=True, verbose=False)
    INT = m.INTENTS
    idx = {n: i for i, n in enumerate(INT)}

    # 文本头
    ht = torch.load(str(MD / "head_text.pt"), map_location="cpu",
                    weights_only=False)
    ht_w, ht_b = ht["w"], ht["b"]
    print(f"head_text.pt: val_acc(训练时, 含增强) = {ht.get('val_acc', 0)*100:.2f}%")

    @torch.no_grad()
    def text_feat(text):
        ids = m.tok(text, truncation=True, max_length=64)["input_ids"]
        return m._b_cls(m.E_ALL[ids])

    @torch.no_grad()
    def text_pred(text):
        f = text_feat(text)
        p = torch.softmax(ht_w @ f + ht_b, -1)
        return INT[int(p.argmax())], float(p.max())

    res = {"A 纯文本路": [0, 0.0], "B 现状(音频+文本)": [0, 0.0],
           "C 纯音频(CTC 文本外挂关掉)": [0, 0.0]}
    errs = {"A": [], "B": [], "C": []}
    t0 = time.time()
    for k, (r, p) in enumerate(val):
        truth = r["intent"]
        wav = read_wav(str(p))
        # A
        pi, pc = text_pred(r["text"])
        res["A 纯文本路"][0] += int(pi == truth)
        res["A 纯文本路"][1] += pc
        if pi != truth:
            errs["A"].append((r["text"], truth, pi))
        # B / C：走真模型（text_override = 真文本 / None）
        rb = m.predict(wav, text_override=r["text"])
        res["B 现状(音频+文本)"][0] += int(rb["intent"] == truth)
        res["B 现状(音频+文本)"][1] += rb["conf"]
        if rb["intent"] != truth:
            errs["B"].append((r["text"], truth, rb["intent"]))
        rc = m.predict(wav)                       # 不挂文本 → CTC 自解码 + 音频
        res["C 纯音频(CTC 文本外挂关掉)"][0] += int(rc["intent"] == truth)
        res["C 纯音频(CTC 文本外挂关掉)"][1] += rc["conf"]
        if rc["intent"] != truth:
            errs["C"].append((r["text"], truth, rc["intent"]))
        if (k + 1) % 100 == 0:
            print(f"  ... {k+1}/{len(val)}  {(time.time()-t0):.0f}s")

    n = len(val)
    print(f"\n{'='*74}\n结果（{n} 条 val，{time.time()-t0:.0f}s）\n{'='*74}")
    print(f"{'路线':<28}{'准确率':>10}{'平均置信':>10}")
    for k, (c, s) in res.items():
        print(f"{k:<28}{c/n*100:>9.2f}%{s/n:>10.3f}")

    for tag in ("A", "B", "C"):
        e = errs[tag]
        print(f"\n{tag} 错例 {len(e)} 条（前 8）:")
        for text, truth, got in e[:8]:
            print(f"   真={truth:14s} 判={got:14s} 「{text}」")

    print("\n按意图（A 纯文本路）:")
    per = {}
    for r, p in val:
        pi, _ = text_pred(r["text"])
        a, b = per.setdefault(r["intent"], [0, 0])
        per[r["intent"]] = [a + int(pi == r["intent"]), b + 1]
    for k, (c, t) in sorted(per.items(), key=lambda kv: -kv[1][1]):
        print(f"   {k:16s} {c:4d}/{t:<4d} {c/t*100:6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
