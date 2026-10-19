import time, numpy as np, sounddevice as sd
print("  默认输入设备:", sd.query_devices(sd.default.device[0])["name"])
x = sd.rec(int(6*16000), samplerate=16000, channels=1, dtype="float32"); sd.wait()
x = x.ravel()
for s in range(6):
    seg = x[s*16000:(s+1)*16000]
    print("  第 %d 秒: RMS=%.5f 峰值=%.5f" % (s+1, float(np.sqrt((seg**2).mean())), float(np.abs(seg).max())))
print("  全程 RMS=%.5f 峰值=%.5f  -> %s" % (float(np.sqrt((x**2).mean())), float(np.abs(x).max()),
      "有信号 ✓" if float(np.abs(x).max()) > 0.02 else "⚠️ 几乎没有信号（设备静音/选错/被独占）"))
