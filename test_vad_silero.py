# -*- coding: utf-8 -*-
r"""Silero 标准链路验收：分段正确性 + 放大后 ASR 是否照常

用法: E:\voice_agent_build\Scripts\python.exe test_vad_silero.py
"""
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SR = 16000


def load(p):
    with wave.open(str(p), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()),
                             dtype="<i2").astype(np.float32) / 32768.0


def run(engine, stream, chunk=480):
    engine.reset()
    segs = []
    for i in range(0, len(stream), chunk):
        segs += engine.push(stream[i:i + chunk])
    t = engine.flush()
    if t is not None:
        segs.append(t)
    return segs


voice = load(sorted((HERE / "wavs").glob("train_launch_app_*.wav"))[0])
voice_mic = voice / (np.sqrt((voice ** 2).mean()) + 1e-12) * 0.0015
rng = np.random.default_rng(0)
noise_mic = rng.normal(0, 0.0005, 4 * SR).astype(np.float32)
sil = np.zeros(4 * SR, dtype=np.float32)
mix = voice_mic[:2 * SR] * 1.0 + rng.normal(0, 0.0004, 2 * SR).astype(np.float32)

from vad_silero import SileroVADEngine                 # noqa: E402

cases = [
    ("① 本机电平人声 + 静音  (期望 1 段)", np.concatenate([sil[:SR], voice_mic, sil])),
    ("② 纯噪声(本机噪声底)   (期望 0 段)", noise_mic),
    ("③ 数字静音            (期望 0 段)", sil),
    ("④ 正常电平人声         (期望 1 段)", np.concatenate([sil[:SR], voice, sil])),
    ("⑤ 本机电平 + 噪声混合   (期望 1 段)", np.concatenate([sil[:SR], mix, sil])),
    ("⑥ 两句相隔 1s          (期望 2 段)", np.concatenate(
        [sil[:SR], voice_mic, sil[:SR], voice_mic, sil])),
]

e = SileroVADEngine()
print(f"{'用例':40s} {'段数':>4s} {'段长(s)':>18s} {'AGC增益':>8s} {'末次prob':>9s}")
print("-" * 84)
ok_all = True
for tag, stream in cases:
    segs = run(e, stream)
    want = int(tag.split("期望 ")[1].split(" ")[0].rstrip("段"))
    good = len(segs) == want
    ok_all &= good
    print(f"{tag:40s} {len(segs):>4d} "
          f"{str([round(len(s)/SR,2) for s in segs]):>18s} "
          f"{e.gain_now:>8.1f} {e.last_prob:>9.3f}  {'✅' if good else '❌'}")

print(f"\n分段全部符合预期: {'✅' if ok_all else '❌'}")

print("\n--- 放大后 ASR 是否照常（GGUF Q4_K_M）---")
try:
    from asr_gguf import GGUFASR
    asr = GGUFASR()
    asr.load()
    for tag, x in (("原始电平(0.0015)", voice_mic), ("AGC后(~0.03)", np.clip(voice_mic * 30, -1, 1))):
        t = asr.transcribe(x)
        print(f"  {tag:18s} → 「{t}」  ({asr.last_ms:.0f}ms {asr.last_tokens}tok)")
except Exception as ex:
    print(f"  （ASR 验证跳过: {type(ex).__name__}: {ex}）")
