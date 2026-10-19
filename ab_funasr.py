# -*- coding: utf-8 -*-
"""A/B：FunASR SenseVoice-small(CPU) vs 我们现在的 Qwen3-ASR 1.7B Q4(GPU, llama-server)。

为什么做：中文/英文转写谁更准，官方榜单不能直接套到我们场景
（用户自己的麦克风、远场、中英混说、Q4 量化）—— 只能用自己的真实录音测。

用法: python ab_funasr.py [--n 8] [--lang zh]
"""
import argparse
import io
import json
import statistics
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OURS = "http://127.0.0.1:30803/v1/audio/transcriptions"


def wav_of(p):
    with wave.open(str(p), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def ours(p):
    """我们的后端：GPU 上的 Qwen3-ASR 1.7B Q4_K_M（llama-server，OpenAI 兼容）"""
    x = wav_of(p)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    B = "----ab"
    parts = []
    for k, v in (("model", "qwen3-asr"), ("language", "zh"),
                 ("response_format", "json")):
        parts.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                      % (B, k, v)).encode())
    parts.append(("--%s\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
                  "Content-Type: audio/wav\r\n\r\n" % B).encode())
    parts.append(buf.getvalue())
    parts.append(b"\r\n")
    parts.append(("--%s--\r\n" % B).encode())
    t0 = time.time()
    req = urllib.request.Request(OURS, data=b"".join(parts), method="POST",
                                 headers={"Content-Type": "multipart/form-data; boundary=" + B})
    with urllib.request.urlopen(req, timeout=180) as r:
        txt = json.loads(r.read()).get("text", "")
    return txt.split("<asr_text>", 1)[-1].strip(), (time.time() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--lang", default="zh")
    a = ap.parse_args()

    print("  加载 SenseVoice-small（CPU，首次会下载模型 ~230MB）...")
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    sv = AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True,
                   device="cpu", disable_update=True)
    print("  模型就绪 ✓")

    files = sorted((HERE / "utt").glob("*mic*.wav"))[-a.n:]
    print("\n  %-34s | %-30s | %s" % ("录音", "SenseVoice(small,CPU)", "我们(Qwen3-ASR 1.7B Q4,GPU)"))
    print("  " + "-" * 112)
    rows = []
    for f in files:
        t0 = time.time()
        try:
            r = sv.generate(input=str(f), cache={}, language=a.lang, use_itn=True,
                            batch_size_s=60, merge_vad=True, merge_length_s=15)
            sv_txt = rich_transcription_postprocess(r[0]["text"]) if r else ""
        except Exception as e:
            sv_txt = "ERR " + type(e).__name__
        sv_ms = (time.time() - t0) * 1000
        try:
            our_txt, our_ms = ours(f)
        except Exception as e:
            our_txt, our_ms = "ERR " + type(e).__name__, 0.0
        rows.append((f.name, sv_txt, sv_ms, our_txt, our_ms))
        print("  %-34s | %-30s | %s" % (f.name[4:38], sv_txt[:30], our_txt[:34]))
    print("  " + "-" * 112)
    sv_lat = statistics.median([r[2] for r in rows])
    our_lat = statistics.median([r[4] for r in rows])
    print("  延迟中位:  SenseVoice %.0fms(CPU)   |   我们 %.0fms(GPU)" % (sv_lat, our_lat))
    same = sum(1 for r in rows if r[1].replace("。", "") == r[3].replace("。", ""))
    print("  完全相同的句子: %d/%d" % (same, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
