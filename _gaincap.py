import json
from pathlib import Path
root = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
p = root / "intent_config.json"
d = json.loads(p.read_text(encoding="utf-8")); b = d["behavior"]
old = {k: b.get(k) for k in ("agc_max_gain", "vad_max_gain")}
b["agc_max_gain"] = 8.0    # 原 60：把噪声底放大 60 倍 -> 削波 -> Silero 判非语音 -> 丢段
b["vad_max_gain"] = 8.0    # 原 30：同上
b["_comment_gain_cap"] = ("丢音频真因：AGC 上限 60/30 会把低电平输入(含噪声底 0.0007)放大到削波"
    "(峰值 1.0)，Silero 于是判非语音 -> 段丢失/乱码。真语音本就有 0.02~0.08，"
    "只需 <8 倍增益；噪声 ×8 仍是噪声，不会被放大成削波。")
p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("  agc_max_gain / vad_max_gain:", old, "->", {"agc_max_gain": b["agc_max_gain"], "vad_max_gain": b["vad_max_gain"]})
