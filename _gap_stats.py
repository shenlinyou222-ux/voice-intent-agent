# -*- coding: utf-8 -*-
"""纯停顿切分到底会不会切错？——拿 428 条真实麦克风录音量"句内停顿"分布。

方法：对每条已存档的真实语句（它本身就是 VAD 切出来的一段），
      用和 trim_edges 同一套阈值算出 20ms 包络，找【内部】静音段
      （头尾静音不算，那是预滚动/尾巴），看 ≥250/400/600ms 的内部停顿各占多少比例。

为什么这个比例直接决定"纯停顿切分"行不行：
  · 内部停顿 ≥ 阈值  →  纯停顿切分会把这一句【从中间切开】= 一条指令变两条残句
  · 本项目的实测血债：utt/162207「就好像可以了」+ 162212「感觉好像可以了吧」
    就是同一句话被切两半、还重复打了一遍

注：存档段里不可能有 ≥704ms 的停顿（那样早被 VAD 切走了），
    所以这里量到的正是"如果改用更短阈值（300~600ms）会误切多少"。
"""
import json
import sys
from pathlib import Path

import numpy as np
import wave

HERE = Path(__file__).parent
FRAME = 320          # 20ms
THRESHOLDS = (80, 100, 150, 200, 250, 300, 400, 600)


def load(p):
    with wave.open(str(p), "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return x, sr


def internal_gaps_ms(x, ratio=6.0, floor=0.0008):
    """返回内部静音段时长列表（ms）"""
    n = len(x) // FRAME
    if n < 4:
        return []
    e = np.sqrt((x[:n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1))
    amb = float(np.percentile(e, 10))
    thr = max(amb * ratio, floor)
    v = e >= thr
    idx = np.where(v)[0]
    if len(idx) == 0:
        return []
    a, b = int(idx[0]), int(idx[-1])          # 只看首末发声之间
    gaps, run = [], 0
    for k in range(a, b + 1):
        if not v[k]:
            run += 1
        else:
            if run:
                gaps.append(run * 20)
            run = 0
    return gaps


rows = [json.loads(l) for l in (HERE / "utt" / "index.jsonl").read_text(
    encoding="utf-8").splitlines() if l.strip()]
rows = [r for r in rows if r.get("src") == "mic"]

stats = {t: 0 for t in THRESHOLDS}
per_file, n_ok = [], 0
for r in rows:
    p = HERE / "utt" / r["wav"]
    if not p.exists():
        continue
    try:
        x, sr = load(p)
    except Exception:
        continue
    if len(x) < 4000:
        continue
    n_ok += 1
    g = internal_gaps_ms(x)
    mx = max(g) if g else 0
    per_file.append((mx, r["wav"], r.get("text", ""), g))
    for t in THRESHOLDS:
        if any(q >= t for q in g):
            stats[t] += 1

print(f"\n  真实麦克风存档：{len(rows)} 条，可用 {n_ok} 条")
print(f"\n  ══ 句内停顿会让【纯停顿切分】误切的比例 ══")
for t in THRESHOLDS:
    print(f"     阈值 {t}ms：{stats[t]:>4} 条会被从中间切开 "
          f"= {stats[t]/max(1,n_ok)*100:5.1f}%")

gaps_all = [q for _, _, _, gs in per_file for q in gs]
if gaps_all:
    a = np.array(gaps_all)
    print(f"\n  内部停顿样本 {len(a)} 个：中位 {np.median(a):.0f}ms  "
          f"p90 {np.percentile(a,90):.0f}ms  max {a.max():.0f}ms")
    for lo, hi in ((100, 200), (200, 300), (300, 500), (500, 704)):
        m = ((a >= lo) & (a < hi)).sum()
        print(f"     {lo:>3}~{hi:<3}ms: {m:>4} 个（{m/len(a)*100:4.1f}%）")

print(f"\n  ══ 内部停顿最长的 12 条（一句话里其实停过很久的实证）══")
for mx, wav, text, g in sorted(per_file, reverse=True)[:12]:
    print(f"     {mx:>4}ms  停顿 {[int(q) for q in g[:6]]}  「{text[:26]}」")
print()
