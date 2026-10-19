import json, wave
from pathlib import Path
import numpy as np
HERE = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
rows = [json.loads(l) for l in (HERE/"utt"/"index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
rows = [r for r in rows if r.get("src") == "mic"][-10:]
def load(p):
    with wave.open(str(p), "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)/32768.0
    return x, sr
print("  %-34s %6s %8s %8s %6s %6s %7s %8s %8s" % (
    "录音", "时长", "RMS", "峰值", "削波%", "直流", "质心Hz", "首50ms", "尾50ms"))
print("  " + "-" * 116)
for r in rows:
    p = HERE/"utt"/r["wav"]
    if not p.exists(): continue
    x, sr = load(p)
    if len(x) < 512: continue
    n = len(x)
    rms = float(np.sqrt((x**2).mean())); pk = float(np.abs(x).max())
    clip = 100.0*float((np.abs(x) > 0.99).mean()); dc = float(x.mean())
    X = np.abs(np.fft.rfft((x-x.mean())*np.hanning(n))); f = np.fft.rfftfreq(n, 1.0/sr)
    cen = float((f*X).sum()/(X.sum()+1e-12))
    h = int(0.05*sr)
    e0 = float(np.sqrt((x[:h]**2).mean())); e1 = float(np.sqrt((x[-h:]**2).mean()))
    print("  %-34s %5.2fs %8.5f %8.5f %5.2f%% %7.4f %7.0f %8.5f %8.5f" % (
        r["wav"][:34], n/sr, rms, pk, clip, dc, cen, e0, e1))
print()
print("  判读规则：")
print("   · 首/尾 50ms 能量 接近段内 RMS -> 段可能是【从说话中间切开】的 ✗（切断问题）")
print("   · 削波% > 1% -> 增益把信号推爆了 ✗（增强问题）")
print("   · 质心 <300Hz 或 >4000Hz -> 不像人声 ✗")
print("   · RMS < 0.002 -> 只到噪声底 ✗")
