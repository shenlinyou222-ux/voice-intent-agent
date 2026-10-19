# -*- coding: utf-8 -*-
"""段首切断排查：看每个 utt 段【开头】的包络是"从静音渐起"还是"凭空就在高位"。

判读：
  · 段首 0~300ms 里存在一段 RMS < 段内RMS×0.15 的静音（预滚动留下的） -> 段头完整 ✓
  · 第 0 个样点起就是高位、且前 10ms 内没有爬升 -> 从说话中间切进来 ✗（吃掉前一两个字）
  · 起始处 |x[0]| 明显非零（>0.01）-> 波形被硬截断（有跳变）✗
"""
import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
rows = [json.loads(l) for l in (HERE / "utt" / "index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
rows = [r for r in rows if r.get("src") == "mic"][-12:]


def load(p):
    with wave.open(str(p), "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return x, sr


def env(x, sr, win_ms=10):
    h = max(1, int(sr * win_ms / 1000))
    n = len(x) // h
    return np.sqrt((x[:n * h].reshape(n, h) ** 2).mean(axis=1))


for r in rows:
    p = HERE / "utt" / r["wav"]
    if not p.exists():
        continue
    x, sr = load(p)
    if len(x) < 1600:
        continue
    e = env(x, sr)
    rms = float(np.sqrt((x ** 2).mean()))
    # 找第一个"起音点"：包络第一次超过 段内RMS×1.0 的窗
    idx = np.argmax(e >= rms) if (e >= rms).any() else -1
    # 段首静音长度：从 0 开始连续低于 rms*0.15 的窗数
    sil = 0
    for v in e:
        if v < rms * 0.15:
            sil += 1
        else:
            break
    head = " ".join(f"{v:.4f}" for v in e[:16])
    print(f"\n  {r['wav'][:40]}")
    print(f"    时长 {len(x)/sr:5.2f}s  段内RMS {rms:.5f}  首50ms {float(np.sqrt((x[:int(.05*sr)]**2).mean())):.5f}"
          f"  段首静音 {sil*10}ms  起音点(>=RMS) {idx*10 if idx>=0 else -1}ms  |x[0]|={abs(float(x[0])):.5f}"
          f"  前3样点={x[:3]}")
    print(f"    前160ms 包络(10ms/格): {head}")
print()
