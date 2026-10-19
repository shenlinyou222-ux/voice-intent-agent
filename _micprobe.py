import sys, time, wave
from pathlib import Path
import numpy as np
import sounddevice as sd
sys.path.insert(0, ".")
OUT = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\mic_probe")
OUT.mkdir(exist_ok=True)

def save(path, x, sr=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())

print("  设备:", sd.query_devices(sd.default.device[0])["name"])
print("  采 8 秒（说话）...")
x = sd.rec(int(8 * 16000), samplerate=16000, channels=1, dtype="float32"); sd.wait(); x = x.ravel()
save(OUT / "01_mic_raw.wav", x)
print("  ① 原始麦克风        : RMS %.5f 峰值 %.5f -> 01_mic_raw.wav" % (
    float(np.sqrt((x**2).mean())), float(np.abs(x).max())))

# ② 走我们的 VAD 引擎（和 run_mic 同一条），把切出来的段落盘
from vad_silero import VADFactory
eng = VADFactory.get_vad_engine(target_rms=0.03, max_gain=30.0, prob_threshold=0.4,
                                db_threshold=40, required_hits=2, required_misses=22)
segs = []
segs = eng.push(x) or []
print("  ② VAD 切出 %d 段" % len(segs))
for i, s in enumerate(segs):
    a = np.asarray(s, dtype=np.float32)
    save(OUT / ("02_seg%d.wav" % (i + 1)), a)
    print("     段%d: %.2fs  RMS %.5f 峰值 %.5f -> 02_seg%d.wav" % (
        i + 1, len(a) / 16000.0, float(np.sqrt((a**2).mean())), float(np.abs(a).max()), i + 1))
# ③ AGC 增强后（handle() 里那段逻辑的等价实现）
tgt, maxg = 0.06, 60.0
rms0 = float(np.sqrt((x ** 2).mean()))
g = min(maxg, tgt / rms0) if rms0 > 1e-7 else 1.0
y = np.clip(x * g, -1, 1).astype(np.float32)
save(OUT / "03_after_agc.wav", y)
print("  ③ AGC 增强后(×%.1f)   : RMS %.5f 峰值 %.5f -> 03_after_agc.wav" % (
    g, float(np.sqrt((y**2).mean())), float(np.abs(y).max())))
print("  目录:", OUT)
