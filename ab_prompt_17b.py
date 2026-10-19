# -*- coding: utf-8 -*-
"""进程内 1.7B：热词 prompt 有/无 的 A/B（同一份权重、同一批真实录音）。

动机：同一段音频，app 的 llama-server（无 prompt）出「其他都不需要去干了。」✓
      我的进程内（382 字热词 prompt）出「其他的都不需要去看了。」✗
      → 怀疑热词 prompt 把 1.7B 带偏。
"""
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, hotword_context          # noqa: E402
from asr_gguf import GGUFASR                         # noqa: E402

HERE = Path(__file__).parent
GD17 = Path(r"C:\Users\user\Desktop\ds harness\aigc_demo_framework"
            r"\models\qwen3-asr-gguf-1.7b")


def read(p):
    with wave.open(str(p), "rb") as w:
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return x.reshape(-1, ch).mean(1) if ch > 1 else x


def main():
    cfg = Config(str(HERE / "intent_config.json"))
    clips = [p for p in sorted((HERE / "utt").glob("*mic*.wav"))][-3:]
    ctx = hotword_context(cfg, extra=["启动", "打开", "搜索"], max_chars=400,
                          ascii_only=False)
    asr = GGUFASR(gguf=GD17 / "Qwen3-ASR-1.7B-Q4_K_M.gguf",
                  mmproj=GD17 / "mmproj-Qwen3-ASR-1.7B-Q4_K_M.gguf",
                  n_threads=8, n_ctx=4096, language="Chinese")
    asr.load()
    for tag, c in (("无 prompt（对标 app）", None),
                   ("有 prompt（现在线上）", ctx)):
        asr.context = c
        print(f"\n=== {tag} ===")
        for p in clips:
            t = asr.transcribe(read(p))
            print(f"  {p.name[:38]:40s} {t!r}  ({asr.last_ms:.0f}ms)")
    asr.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
