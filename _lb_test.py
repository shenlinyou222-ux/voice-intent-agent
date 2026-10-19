import io, json, urllib.request, wave
from pathlib import Path
import numpy as np
HERE = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
ids = json.loads((HERE / "trad_token_ids.json").read_text(encoding="utf-8"))
URL = "http://127.0.0.1:30803/v1/audio/transcriptions"
f = HERE / "utt" / "115750_mic_unknown_DeepSeek網頁版.wav"
if not f.exists():
    cand = [p for p in (HERE/"utt").glob("*.wav") if "網" in p.name or "页" in p.name]
    f = cand[0] if cand else sorted((HERE/"utt").glob("*.wav"))[-1]
print("  测试录音:", f.name)
with wave.open(str(f), "rb") as w:
    raw = w.readframes(w.getnframes())
x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
buf = io.BytesIO()
with wave.open(buf, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
def call(extra=None):
    B = "----x"; parts = []
    for k, v in (("model", "qwen3-asr"), ("language", "zh"), ("response_format", "json")):
        parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (B, k, v)).encode())
    for k, v in (extra or {}).items():
        parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (B, k, v)).encode())
    parts.append(("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
                  "Content-Type: audio/wav\r\n\r\n" % B).encode())
    parts.append(buf.getvalue()); parts.append(b"\r\n"); parts.append(("--%s--\r\n" % B).encode())
    body = b"".join(parts)
    req = urllib.request.Request(URL, data=body, method="POST",
                                 headers={"Content-Type": "multipart/form-data; boundary=" + B})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read()).get("text", "")[:120]
    except Exception as e:
        return "ERR " + str(e)[:120]
print("  ① 基线            :", call())
print("  ② 传 logit_bias   :", call({"logit_bias": json.dumps({str(i): -100.0 for i in ids})}))
print("  ③ 传 banned_tokens:", call({"banned_tokens": json.dumps(ids[:50])}))
