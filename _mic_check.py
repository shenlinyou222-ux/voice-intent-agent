import time
import numpy as np
import sounddevice as sd
print("  === 输入设备清单 ===")
try:
    devs = sd.query_devices()
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            mark = " <-- 默认" if i == sd.default.device[0] else ""
            print("    [%2d] %-46s ch=%d sr=%d%s" % (i, d["name"][:44], d["max_input_channels"],
                                                    int(d["default_samplerate"]), mark))
except Exception as e:
    print("    查询失败:", e)
print("  === 采 2 秒原始音频（默认设备 16k 单声道）===")
try:
    x = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    x = x.ravel()
    rms = float(np.sqrt((x ** 2).mean())); pk = float(np.abs(x).max())
    print("    RMS=%.5f  峰值=%.5f  直流偏置=%.6f  削波比例=%.2f%%" % (
        rms, pk, float(x.mean()), 100.0 * float((np.abs(x) > 0.99).mean())))
    # 用工程里的 Silero 引擎算概率
    import sys; sys.path.insert(0, ".")
    from vad_silero import VADFactory
    eng = VADFactory.get_vad_engine(target_rms=0.03, max_gain=30.0, prob_threshold=0.4,
                                    db_threshold=40, required_hits=3, required_misses=40)
    probs, lvls = [], []
    segs = eng.push(x) if hasattr(eng, "push") else []
    print("    引擎段数（2 秒内）:", len(segs) if segs is not None else "n/a")
    for i in range(0, len(x) - 512, 512):
        try:
            r = eng.process(x[i:i+512]) if hasattr(eng, "process") else None
            if isinstance(r, dict):
                probs.append(r.get("prob", 0)); lvls.append(r.get("level", 0))
        except Exception:
            break
    if probs:
        print("    帧数=%d  概率 max=%.3f 中位=%.3f   电平 max=%.4f" % (
            len(probs), max(probs), float(np.median(probs)), max(lvls) if lvls else 0))
    else:
        print("    （引擎接口不是 process/push，跳过逐帧概率）")
except Exception as e:
    print("    采集失败:", type(e).__name__, e)
