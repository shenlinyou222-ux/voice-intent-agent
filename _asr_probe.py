# -*- coding: utf-8 -*-
"""把 _cutprobe/raw.wav 送 app 在用的 ASR（llama-asr :30803）转写，
   用"ASR 认不认这是话"来判定 Silero 低概率是音频问题还是 VAD 问题。"""
import json
import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CFG = json.loads((HERE / "intent_config.json").read_text(encoding="utf-8"))["behavior"]
URL = CFG.get("asr_http_url", "http://127.0.0.1:30803/v1/audio/transcriptions")
MODEL = CFG.get("asr_http_model", "qwen3-asr")

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "_cutprobe"
with wave.open(str(DIR / "raw.wav"), "rb") as w:
    SR = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def post(seg, sr=SR):
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(seg, -1, 1) * 32767).astype("<i2").tobytes())
    body = buf.getvalue()
    bnd = "----cutprobe"
    parts = []
    parts.append(f"--{bnd}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{MODEL}\r\n".encode())
    parts.append(f"--{bnd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
                 f"Content-Type: audio/wav\r\n\r\n".encode() + body + b"\r\n")
    parts.append(f"--{bnd}--\r\n".encode())
    data = b"".join(parts)
    req = urllib.request.Request(URL, data=data,
                                 headers={"Content-Type": f"multipart/form-data; boundary={bnd}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


# 逐 5 秒切片转写：定位"哪一段里有话"
print(f"  {DIR/'raw.wav'}  {len(x)/SR:.2f}s   →  {URL}")
step = 5 * SR
for i in range(0, len(x), step):
    seg = x[i:i + step]
    if len(seg) < SR:
        break
    rms = float(np.sqrt((seg ** 2).mean()))
    try:
        out = post(seg)
        txt = (out.get("text") or "").strip()
    except Exception as e:
        txt = f"❌ {type(e).__name__}: {e}"
    print(f"     {i/SR:5.1f}~{(i+len(seg))/SR:5.1f}s  RMS {rms:.5f}  峰值 {np.abs(seg).max():.5f}  "
          f"→ {txt!r}")

# 整段也来一次（放大 8 倍，模拟生产链路里的 PreGain）
g = x * 8.0
try:
    out = post(g)
    print(f"\n  整段（×8 增益后）→ {(out.get('text') or '').strip()!r}")
except Exception as e:
    print(f"\n  整段失败: {type(e).__name__}: {e}")
