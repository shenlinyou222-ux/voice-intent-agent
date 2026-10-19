from pathlib import Path
p = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\vad_silero.py")
t = p.read_text(encoding="utf-8")
old = "            min_gain=1.0, max_gain=30.0, smooth=0.3, ceiling=0.95):"
old2 = "    def __init__(self, target_rms=0.03, window_s=10.0, pct=95.0,\n            min_gain=1.0, max_gain=30.0, smooth=0.3, ceiling=0.95):"
new2 = ("    def __init__(self, target_rms=0.03, window_s=10.0, pct=95.0,\n"
        "            # ⚠️⚠️ 血债：min_gain 原来是 1.0 = AGC【只能放大不能衰减】✗\n"
        "            #   实测：输入稍大（RMS 0.0675 = 目标的 2.2 倍）时，AGC 算出该衰减\n"
        "            #   却被夹在 1.0 -> 电平永远偏高 -> Silero 概率 ~0.006 -> 切不出段\n"
        "            #   -> 用户实测「无法转写」。正常 AGC 必须能双向调节，故下限放到 0.1。\n"
        "            min_gain=0.1, max_gain=30.0, smooth=0.3, ceiling=0.95):")
if old2 in t:
    t = t.replace(old2, new2, 1); print("  ✅ min_gain 1.0 -> 0.1（允许衰减）")
elif old in t:
    t = t.replace(old, "            min_gain=0.1, max_gain=30.0, smooth=0.3, ceiling=0.95):", 1)
    print("  ✅ min_gain 1.0 -> 0.1（单行锚点）")
else:
    print("  ⚠️ 锚点未匹配")
p.write_text(t, encoding="utf-8")
