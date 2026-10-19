# -*- coding: utf-8 -*-
"""预滚动（pre-roll）实测：喂 [1s 静音][一段真语音][1s 静音]，看切出来的段
   起点到底落在语音开始前的多少毫秒 —— 这就是"会不会吃掉前一两个字"的直接答案。

   同时也测第二种情形（真实说话里更常见的）：
   [1s 静音][语音 A][0.2s 短停顿][语音 B][1s 静音]
   —— 若 VAD 在短停顿处误判结束，B 段会从停顿后不久就被切出来（起点前没有静音）。
"""
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vad_silero import SileroVADEngine, WINDOW, SR  # noqa

HERE = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")


def load(p):
    with wave.open(str(p), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def run(stream, tag, **kw):
    eng = SileroVADEngine(pre_gain=False, **kw)
    got = []
    for i in range(0, len(stream), 480):
        got += eng.push(stream[i:i + 480])
    print(f"\n  ── {tag} ──")
    print(f"     切出 {len(got)} 段: {[f'{len(g)/SR:.2f}s' for g in got]}")
    return got


def onset_of(x, thr):
    """第一个超过 thr 的样点位置"""
    idx = np.argmax(np.abs(x) > thr)
    return int(idx) if np.abs(x).max() > thr else -1


src = load(HERE / "utt" / "152315_mic_dictation_它这个转写是不是有点问题.wav")
body = src[:int(1.8 * SR)]
speech = body * (0.03 / max(float(np.sqrt((body ** 2).mean())), 1e-9))   # 归一到 0.03 RMS

sil1 = np.zeros(int(1.0 * SR), np.float32)
gap = np.zeros(int(0.2 * SR), np.float32)
tail = np.zeros(int(1.0 * SR), np.float32)

print(f"  语音素材 {len(speech)/SR:.2f}s  归一后 RMS "
      f"{float(np.sqrt((speech**2).mean())):.5f}")

# ① 单段：静音 + 语音 + 静音
stream = np.concatenate([sil1, speech, tail])
segs = run(stream, "① [1s 静音][语音 1.8s][1s 静音]   （理想：段应从语音前 ~640ms 起）")
sp_start = len(sil1)
for k, g in enumerate(segs):
    # 段在流里的位置：用能量对齐（段首若有静音，它的长度就是预滚动）
    lead = 0
    win = int(0.005 * SR)
    while (lead + win) < len(g) and float(np.abs(g[lead:lead + win]).max()) < 0.002:
        lead += win
    print(f"     段{k}: 总长 {len(g)/SR:.2f}s  段首静音(<-46dBFS) ≈ {lead/SR*1000:.0f}ms")

# ② 两段：短停顿（0.2s）—— 真实说话里最常见的"切断"场景
stream2 = np.concatenate([sil1, speech[:int(0.8 * SR)], gap, speech[int(0.8 * SR):], tail])
segs2 = run(stream2, "② [1s 静音][语音 A 0.8s][0.2s 停顿][语音 B][1s 静音]")
for k, g in enumerate(segs2):
    lead = 0
    win = int(0.005 * SR)
    while (lead + win) < len(g) and float(np.abs(g[lead:lead + win]).max()) < 0.002:
        lead += win
    print(f"     段{k}: 总长 {len(g)/SR:.2f}s  段首静音(<-46dBFS) ≈ {lead/SR*1000:.0f}ms")

# ③ 两段：长停顿（0.8s）
stream3 = np.concatenate([sil1, speech[:int(0.8 * SR)], np.zeros(int(0.8 * SR), np.float32),
                          speech[int(0.8 * SR):], tail])
segs3 = run(stream3, "③ [1s 静音][语音 A 0.8s][0.8s 停顿][语音 B][1s 静音]")
for k, g in enumerate(segs3):
    lead = 0
    win = int(0.005 * SR)
    while (lead + win) < len(g) and float(np.abs(g[lead:lead + win]).max()) < 0.002:
        lead += win
    print(f"     段{k}: 总长 {len(g)/SR:.2f}s  段首静音(<-46dBFS) ≈ {lead/SR*1000:.0f}ms")
print()
