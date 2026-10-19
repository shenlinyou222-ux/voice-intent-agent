# -*- coding: utf-8 -*-
"""看每个 utt 段的【整段包络形状】—— 判断"响头+轻身"是真起音还是被切/被 AGC 抬起来的。

输出：整段 60 格 sparkline（20ms/格，取值按段内峰值归一）＋ 头/身/尾三段 RMS 与峰值位置。
"""
import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
rows = [json.loads(l) for l in (HERE / "utt" / "index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
rows = [r for r in rows if r.get("src") == "mic"][-14:]

BARS = " .:-=+*#%@"


def spark(e):
    m = float(np.max(e)) or 1e-9
    return "".join(BARS[min(9, int(v / m * 9.999))] for v in e)


for r in rows:
    p = HERE / "utt" / r["wav"]
    if not p.exists():
        continue
    with wave.open(str(p), "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    h = int(sr * 0.02)
    n = len(x) // h
    e = np.sqrt((x[:n * h].reshape(n, h) ** 2).mean(axis=1))
    # 重采样到 60 格
    idx = np.linspace(0, n, 61).astype(int)
    e60 = np.array([e[a:b].max() if b > a else 0.0 for a, b in zip(idx[:-1], idx[1:])])
    a = e[:5].mean()
    b = e[15:min(n, 75)].mean()
    c = e[-5:].mean()
    print(f"\n  {r['wav'][:44]}  ({len(x)/sr:.2f}s)")
    print(f"    |{spark(e60)}|   (20ms/格, 按段内峰值归一; 峰值格={int(np.argmax(e))}/{n})")
    print(f"    头100ms {a:.5f}   身(0.3~1.5s) {b:.5f}   尾100ms {c:.5f}"
          f"   头/身={a/max(b,1e-9):5.2f}x   |x[0]|={abs(float(x[0])):.5f}"
          f"   文本={r.get('text','')!r}")
print()
