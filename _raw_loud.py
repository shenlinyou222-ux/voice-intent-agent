# -*- coding: utf-8 -*-
"""原始录音里把最响的 100ms 窗找出来，逐个报：原始 RMS / 增益后 db / Silero prob
   —— 用来判断"这段到底是不是人声"，以及 VAD 为什么没触发。"""
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from vad_silero import PreGain, SileroVAD, WINDOW, SR  # noqa

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "_cutprobe"
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 12

with wave.open(str(DIR / "raw.wav"), "rb") as w:
    x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0

g = PreGain(target_rms=0.03, max_gain=8.0)
vad = SileroVAD()
probs, dbs, gains = [], [], []
for i in range(0, len(x) - WINDOW + 1, WINDOW):
    c = g.apply(x[i:i + WINDOW])
    probs.append(vad.prob(c))
    rms = float(np.sqrt(np.mean((c * 32767) ** 2)))
    dbs.append(20 * np.log10(rms + 1e-7) if rms > 0 else -np.inf)
    gains.append(g.last_g)

h = 1600                      # 100ms
n = len(x) // h
rms100 = np.array([float(np.sqrt((x[i*h:(i+1)*h] ** 2).mean())) for i in range(n)])
order = np.argsort(rms100)[::-1]
print(f"\n  {DIR/'raw.wav'}  {len(x)/SR:.2f}s  峰值 {np.abs(x).max():.5f}")
print(f"  原始 100ms 窗 RMS：中位 {np.median(rms100):.5f}  p95 {np.percentile(rms100,95):.5f}  "
      f"最大 {rms100.max():.5f}")

# 频谱质心：人声 200~2000Hz；敲桌子/电流声偏两端
def centroid(seg):
    S = np.abs(np.fft.rfft((seg - seg.mean()) * np.hanning(len(seg))))
    f = np.fft.rfftfreq(len(seg), 1.0 / SR)
    return float((f * S).sum() / (S.sum() + 1e-12))


print(f"\n  最响的 {TOPN} 个 100ms 窗：")
print(f"     {'时间':>7} {'原始RMS':>9} {'增益dB':>7} {'prob':>6} {'质心Hz':>7}  说明")
for k in order[:TOPN]:
    t = k * 0.1
    ci = min(int(t / (WINDOW / SR)), len(probs) - 1)
    seg = x[k*h:(k+1)*h]
    c = centroid(seg)
    note = []
    if rms100[k] < 0.001:
        note.append("≈环境底")
    if c < 300:
        note.append("低频（撞击/风声）")
    elif c > 3500:
        note.append("高频（齿音/电流）")
    else:
        note.append("像人声")
    if probs[ci] < 0.3:
        note.append(f"VAD 说不是人声({probs[ci]:.2f})")
    print(f"     {t:6.1f}s {rms100[k]:9.5f} {dbs[ci]:7.1f} {probs[ci]:6.3f} {c:7.0f}  "
          f"{' / '.join(note)}")
print(f"\n  增益轨迹：起 {gains[0]:.1f} → 中位 {np.median(gains):.1f} → 末 {gains[-1]:.1f}")
