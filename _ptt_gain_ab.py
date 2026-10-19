# -*- coding: utf-8 -*-
"""PTT 段该用哪个电平送 ASR？—— 拿 demo 存的同一批 wav 做 A/B（×1 vs ×8）。

背景：现状送 ASR 的是【VAD 引擎输出】（PreGain 常年 ×8）；
      PTT 路没有引擎增益，所以要确认"真原始电平"能不能照样转对。
"""
import sys
from pathlib import Path

import numpy as np
import wave

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _cut_align import asr  # noqa: E402

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "_ptt"
files = sorted(DIR.glob("*.wav"))
print(f"\n  {DIR}  共 {len(files)} 个")
print(f"  {'文件':<20} {'原始RMS':>9} {'×1 转写':<34} {'×8 转写'}")
for f in files:
    with wave.open(str(f), "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    rms = float(np.sqrt((x ** 2).mean()))
    out = {}
    for g in (1.0, 8.0):
        try:
            out[g] = asr(x, sr=sr, gain=g)
        except Exception as e:
            out[g] = f"❌{type(e).__name__}"
    print(f"  {f.name:<20} {rms:9.5f} {out[1.0]:<34} {out[8.0]}")
print()
