# -*- coding: utf-8 -*-
"""A/B：FunASR SenseVoice-small(ONNX, CPU) vs 我们现在的 Qwen3-ASR 1.7B Q4(GPU)。

为什么不用 funasr 全家桶：pip 装 funasr 卡在网络上（torchaudio/modelscope 一堆依赖 ✗），
sherpa-onnx 是单个轮子 ✓ 且原生支持 SenseVoice ✓ —— 同一份官方 SenseVoice 模型。

模型：E:\\models\\sensevoice-onnx\\sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17\\
用法: python ab_sensevoice.py --n 8 [--lang zh|auto|en]
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
SV_DIR = Path(r"E:\models\sensevoice-onnx\sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")


def wav_of(p):
    with wave.open(str(p), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def ours(p):
    """我们的后端：GPU 上的 Qwen3-ASR 1.7B Q4_K_M（llama-server）"""
    x = wav_of(p)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    B = "----ab"
    parts = []
    for k, v in (("model", "qwen3-asr"), ("language", "zh"), ("response_format", "json")):
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
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    if not SV_DIR.exists():
        print("  ⚠️ 模型目录不存在:", SV_DIR)
        print("     先下载: https://github.com/k2-fsa/sherpa-onnx/releases/download/"
              "asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2")
        return 1
    model = SV_DIR / ("model.int8.onnx" if (SV_DIR / "model.int8.onnx").exists()
                      else "model.onnx")
    tokens = SV_DIR / "tokens.txt"
    print("  加载 SenseVoice(ONNX):", model.name, "| tokens:", tokens.name)
    import sherpa_onnx
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model), tokens=str(tokens), num_threads=a.threads,
        use_itn=True, language=a.lang, debug=False)
    print("  就绪 ✓")

    files = sorted((HERE / "utt").glob("*mic*.wav"))[-a.n:]
    print("\n  %-30s | %-30s | %s" % ("录音", "SenseVoice(small,CPU)", "我们(Qwen3-ASR 1.7B,GPU)"))
    print("  " + "-" * 108)
    rows = []
    for f in files:
        t0 = time.time()
        try:
            s = rec.create_stream()
            # ⚠️ 1.13.8 里没有 accept_wave_file，正确方法是 accept_waveform(sr, 样本)
            s.accept_waveform(16000, wav_of(f))
            rec.decode_stream(s)
            sv_txt = (s.result.text or "").strip()
        except Exception as e:
            sv_txt = "ERR " + type(e).__name__
        sv_ms = (time.time() - t0) * 1000
        try:
            our_txt, our_ms = ours(f)
        except Exception as e:
            our_txt, our_ms = "ERR " + type(e).__name__, 0.0
        rows.append((f.name, sv_txt, sv_ms, our_txt, our_ms))
        print("  %-30s | %-30s | %s" % (f.name[4:34], sv_txt[:30], our_txt[:32]))
    print("  " + "-" * 108)
    print("  延迟中位:  SenseVoice %.0fms(CPU)  |  我们 %.0fms(GPU)" % (
        statistics.median([r[2] for r in rows]), statistics.median([r[4] for r in rows])))
    print("  完全相同的句子: %d/%d" % (
        sum(1 for r in rows if r[1].replace("。", "") == r[3].replace("。", "")), len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
