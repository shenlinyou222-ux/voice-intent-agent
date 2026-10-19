# -*- coding: utf-8 -*-
"""把真实起音放大看：原始包络（20ms）与 Silero 概率（32ms）逐格对照。
   要回答的问题：Silero 的 prob 越过 0.2 的时刻，比"声音真的起来"晚了多少？
   —— 段起点固定在「起音块 − 19 块（608ms）」之前，所以只要这个"晚" < 600ms 就不会切。
"""
import sys
from pathlib import Path
import wave

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _cut_align import probs_of, onset_times, DIR  # noqa: E402
from vad_silero import WINDOW, SR  # noqa: E402

meta = __import__("json").loads((DIR / "meta.json").read_text(encoding="utf-8"))
with wave.open(str(DIR / "raw.wav"), "rb") as w:
    x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0

p = probs_of(x)
on = onset_times(p)
print(f"\n  {DIR/'raw.wav'}  {len(x)/SR:.1f}s   起音 {len(on)} 个: "
      f"{[round(v,2) for v in on]}")
amb = float(np.median([np.sqrt(np.mean(x[i:i+WINDOW]**2)) for i in range(0, len(x)-WINDOW, WINDOW)]))
print(f"  环境底（块 RMS 中位）{amb:.5f}")

BARS = " .:-=+*#%@"
for t_on in on:
    i0 = max(0, int((t_on - 1.0) * SR / WINDOW))
    i1 = int((t_on + 1.5) * SR / WINDOW)
    print(f"\n  ── 起音 {t_on:.2f}s 附近（每格 32ms；'#' 标 prob 首越 0.2 的块）──")
    for i in range(i0, min(i1, len(p))):
        t = i * WINDOW / SR
        r = float(np.sqrt(np.mean(x[i*WINDOW:(i+1)*WINDOW]**2)))
        mark = "#" if abs(t - t_on) < 0.017 else " "
        bar = BARS[min(9, int(r / max(amb * 6, 1e-9) * 9.999))]
        print(f"     {t:7.2f}s {mark} 原始RMS {r:.5f} {bar}   prob {p[i]:.3f} "
              f"{'█' * int(p[i] * 20)}")
print()
