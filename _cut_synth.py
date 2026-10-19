# -*- coding: utf-8 -*-
"""给 _cut_report.py 做冒烟测试：造一条"已知真相"的原始流
   ① 1.5s 环境噪声  →  ② 句子 A（含一个 0.5s 句内停顿，低于 704ms 判停门限）
   →  ③ 0.85s 停顿（高于门限，VAD 必然收段）→  ④ 句子 B（紧接着就说，制造"段头没预滚动"）
   →  ⑤ 1.5s 环境噪声
   期望：段 A 的段头有几百 ms 余量 ✓；段 B 的段头余量很小/为负 ✗（复现"切断"）
"""
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "_cutprobe_synth"
OUT.mkdir(exist_ok=True)
SR = 16000
rng = np.random.default_rng(7)


def load(p):
    with wave.open(str(p), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def to_level(x, rms):
    return x * (rms / max(float(np.sqrt((x ** 2).mean())), 1e-9))


noise = rng.normal(0, 1, int(1.5 * SR)).astype(np.float32)
noise = to_level(noise, 0.0010)                       # 环境底：原始 RMS 0.001（≈ -60dBFS）

a = load(HERE / "utt" / "152312_mic_dictation_目前状态呢.wav")
b = load(HERE / "utt" / "152959_mic_open_url_看一下是哪里有问题.wav")
a = to_level(a, 0.0025)                               # 原始语音电平 0.0025（≈ 本机实测）
b = to_level(b, 0.0025)
gap_in = np.zeros(int(0.5 * SR), np.float32)          # 句内 0.5s 停顿
pause = np.zeros(int(0.85 * SR), np.float32)          # 跨句 0.85s 停顿（>704ms -> 收段）
tail = to_level(rng.normal(0, 1, int(1.5 * SR)).astype(np.float32), 0.0010)

# 句子 A：前半 + 0.5s 停顿 + 后半
half = len(a) // 2
stream = np.concatenate([noise, a[:half], gap_in, a[half:], pause, b, tail])
with wave.open(str(OUT / "raw.wav"), "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((np.clip(stream, -1, 1) * 32767).astype("<i2").tobytes())
marks = {"noise_end": len(noise), "A_start": len(noise),
         "gap_in": [len(noise) + half, len(noise) + half + len(gap_in)],
         "pause": [len(noise) + len(a) + len(gap_in),
                   len(noise) + len(a) + len(gap_in) + len(pause)],
         "B_start": len(noise) + len(a) + len(gap_in) + len(pause)}
import json
(OUT / "truth.json").write_text(json.dumps(
    {k: (v if not isinstance(v, list) else v) for k, v in marks.items()},
    ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  造好 {OUT/'raw.wav'}  {len(stream)/SR:.2f}s")
for k, v in marks.items():
    if isinstance(v, list):
        print(f"    {k}: {v[0]/SR:.2f}s ~ {v[1]/SR:.2f}s")
    else:
        print(f"    {k}: {v/SR:.2f}s")
