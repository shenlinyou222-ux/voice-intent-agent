# -*- coding: utf-8 -*-
"""0.6B vs 1.7B A/B：在【用户真实麦克风录音】上比转写。

为什么要用 utt/ 里的录音：TTS 语料上英文名问题只是"半个真相"，
用户抱怨的是他自己说话时的转写 —— utt/*.wav 就是那些真实片段（不信看 index.jsonl）。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, hotword_context      # noqa: E402
from asr_gguf import GGUFASR                     # noqa: E402

HERE = Path(__file__).parent
GD6 = Path(r"C:\Users\user\Desktop\ds harness\aigc_demo_framework\models\qwen3-asr-gguf")
GD17 = Path(r"C:\Users\user\Desktop\ds harness\aigc_demo_framework\models\qwen3-asr-gguf-1.7b")

CFGS = [
    ("0.6B", GD6 / "Qwen3-ASR-0.6B-Q4_K_M.gguf", GD6 / "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"),
    ("1.7B", GD17 / "Qwen3-ASR-1.7B-Q4_K_M.gguf", GD17 / "mmproj-Qwen3-ASR-1.7B-Q4_K_M.gguf"),
]


def read_wav(p):
    import wave
    with wave.open(str(p), "rb") as w:
        ch, sr = w.getnchannels(), w.getframerate()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        t = np.linspace(0, len(a) / sr, int(len(a) * 16000 / sr), endpoint=False)
        a = np.interp(t, np.arange(len(a)) / sr, a).astype(np.float32)
    return a


def main():
    cfg = Config(str(HERE / "intent_config.json"))
    en = json.loads((HERE / "model" / "meta.json").read_text(encoding="utf-8"))["en_words"]
    ctx = hotword_context(cfg, extra=en, ascii_only=bool(
        cfg.behavior.get("asr_hotword_ascii_only", True)),
        max_chars=int(cfg.behavior.get("asr_hotword_chars", 320)))

    # 真实麦克风录音（utt/ 下 src=mic 的）
    idx = HERE / "utt" / "index.jsonl"
    mic = []
    if idx.exists():
        for line in idx.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("src") == "mic":
                p = HERE / "utt" / r["wav"]
                if p.exists():
                    mic.append((p, r.get("text", ""), r.get("intent", "")))
    mic = mic[-14:]                                     # 最近 14 条
    print(f"真实麦克风录音 {len(mic)} 条（来自 utt/，含你说失败的那些）")
    for p, t, i in mic[:6]:
        print(f"   {p.name[:44]:46s} 0.6B 当时转的:「{t}」")

    results = {}
    for tag, gguf, mm in CFGS:
        if not (gguf.exists() and mm.exists()):
            print(f"\n{tag}: 缺文件，跳过")
            continue
        print(f"\n{'='*74}\n{tag}\n{'='*74}")
        asr = GGUFASR(gguf=gguf, mmproj=mm, n_threads=6,
                      language=cfg.behavior.get("asr_language", "Chinese"),
                      context=ctx,
                      n_ctx=int(cfg.behavior.get("asr_n_ctx", 512)))
        asr.load()
        rows, t0 = [], time.time()
        for p, old, _i in mic:
            w = read_wav(p)
            try:
                got = asr.transcribe(w)
            except Exception as e:
                got = f"<err {type(e).__name__}: {e}>"
            rows.append((p.name, len(w) / 16000, old, got, asr.last_ms))
            print(f"   {p.name[:40]:42s} {len(w)/16000:4.2f}s")
            print(f"      0.6B当时:「{old}」")
            print(f"      {tag}现在:「{got}」  {asr.last_ms:.0f}ms")
        results[tag] = rows
        asr.unload()
        del asr
        print(f"  {len(rows)} 条用时 {time.time()-t0:.0f}s")

    if len(results) == 2:
        a, b = results["0.6B"], results["1.7B"]
        diff = sum(1 for x, y in zip(a, b) if x[3] != y[3])
        print(f"\n{'='*74}")
        print(f"两条模型转写不同的条数：{diff}/{len(a)}")
        print("（相同说明这段音频上两个模型一样；不同处人工看哪个更像人话）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
