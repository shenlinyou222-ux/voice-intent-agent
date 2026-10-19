import sys, wave
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from vad_silero import VADFactory
f = sorted(Path("utt").glob("*mic*dictation*.wav"))[-1]
with wave.open(str(f), "rb") as w:
    x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
print("  录音:", f.name[:40], " 原始 RMS=%.5f 峰值=%.5f" % (
    float(np.sqrt((x**2).mean())), float(np.abs(x).max())))
eng = VADFactory.get_vad_engine(target_rms=0.03, max_gain=30.0, prob_threshold=0.4,
                                db_threshold=40, required_hits=3, required_misses=40)
print("  引擎接口:", [m for m in ("push", "process", "feed", "accept") if hasattr(eng, m)])
for k in (1.0, 3.0, 5.0, 10.0):
    y = np.clip(x * k, -1, 1)
    e2 = VADFactory.get_vad_engine(target_rms=0.03, max_gain=30.0, prob_threshold=0.4,
                                   db_threshold=40, required_hits=3, required_misses=40)
    segs = e2.push(y) if hasattr(e2, "push") else None
    prob = getattr(e2, "last_prob", None)
    peak = getattr(e2, "peak_prob", None)
    print("  ×%-4g 输入RMS=%.4f -> 段数=%s  prob=%s peak_prob=%s" % (
        k, float(np.sqrt((y**2).mean())), (len(segs) if segs is not None else "n/a"),
        prob, peak))
