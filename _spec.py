import time
import numpy as np
import sounddevice as sd
SR = 16000
def analyze(x, sr=SR):
    n = len(x)
    if n < 512:
        return None
    x = x - x.mean()
    X = np.abs(np.fft.rfft(x * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    tot = X.sum() + 1e-12
    lo = X[f < 100].sum() / tot              # 低频隆隆/直流
    mid = X[(f >= 300) & (f <= 3400)].sum() / tot   # 人声主能量区
    hi = X[f > 5000].sum() / tot             # 高频（采样率错位会异常高）
    cen = float((f * X).sum() / tot)         # 谱质心
    rms = float(np.sqrt((x ** 2).mean())); pk = float(np.abs(x).max())
    zcr = float((np.diff(np.sign(x)) != 0).mean())
    return rms, pk, lo, mid, hi, cen, zcr
print("  采 10 秒（默认设备）...")
x = sd.rec(int(10 * SR), samplerate=SR, channels=1, dtype="float32"); sd.wait(); x = x.ravel()
print("  每秒统计： RMS      峰值     低频<100Hz  人声带300-3.4k  高频>5k  谱质心Hz  过零率")
for s in range(10):
    seg = x[s * SR:(s + 1) * SR]
    r = analyze(seg)
    if r:
        print("    %2ds   %.5f  %.5f   %5.1f%%      %5.1f%%        %5.1f%%   %7.0f   %.3f" %
              (s + 1, r[0], r[1], r[2] * 100, r[3] * 100, r[4] * 100, r[5], r[6]))
r = analyze(x)
print("  全程: RMS=%.5f 峰值=%.5f 低频=%.1f%% 人声带=%.1f%% 高频=%.1f%% 质心=%.0fHz" %
      (r[0], r[1], r[2]*100, r[3]*100, r[4]*100, r[5]))
print("  判读: 人声带占比高(>50%%) + 质心 300~2000Hz => 像人声 ✓；低频/高频占比高或质心异常 => 不是人声 ✗")
print("  各设备的默认采样率（若设备强制 44100 而我们按 16000 读，会导致音调错位）:")
for i, d in enumerate(sd.query_devices()):
    if d.get("max_input_channels", 0) > 0 and i < 32:
        print("    [%2d] %-42s default_sr=%d" % (i, d["name"][:40], int(d["default_samplerate"])))
