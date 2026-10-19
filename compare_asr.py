# -*- coding: utf-8 -*-
r"""转写对比：自训 ctc_head（现在 built 走的） vs 它自己的解码器（Qwen3-ASR-0.6B 全量）

用同一批带真值的 TTS wav（intent_tts.tsv 里有 id → 文本），逐条比"转写完全正确"。

用法: E:\voice_agent_build\Scripts\python.exe compare_asr.py [条数]
"""
import io
import os
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def load_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def norm(s):
    return "".join(ch for ch in str(s).strip() if not ch.isspace())


def main():
    # 真值表
    jobs = {}
    for line in io.open(HERE / "intent_tts.tsv", encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            jobs[parts[0]] = parts[1]
    # 均匀抽 N 条（覆盖不同意图族）
    ids = sorted(jobs)[::max(1, len(jobs) // N)][:N]
    print(f"抽样 {len(ids)} 条（共 {len(jobs)} 条真值）\n")

    # ① 现在的 CTC 路
    print("载入 voice_agent 的意图模型（A + ctc_head + dmeta-B）...")
    from model_runner import IntentModel
    m = IntentModel(model_dir=str(HERE / "model"), device="cpu")
    print(f"  {sum(p.numel() for p in m.parameters() if hasattr(m, 'parameters')) if False else ''}"
          f"INTENTS={m.INTENTS}\n")

    # ② 它自己的解码器
    from asr_own import OwnASR
    own = OwnASR()

    rows = []
    for i in ids:
        p = HERE / "wavs" / f"{i}.wav"
        if not p.exists():
            continue
        wav, sr = load_wav(p)
        truth = jobs[i]
        r = m.predict(wav)
        ctc_txt = r["text"]
        own_txt = own.transcribe(wav)
        rows.append((i, truth, ctc_txt, own_txt, own.last_ms))
        print(f"{i[:34]:36s} {len(wav)/16000:5.2f}s")
        print(f"    真值 : {truth}")
        print(f"    CTC  : {ctc_txt}   {'✅' if norm(ctc_txt) == norm(truth) else '❌'}")
        print(f"    解码器: {own_txt}   {'✅' if norm(own_txt) == norm(truth) else '❌'}"
              f"  ({own.last_ms:.0f} ms)")

    if rows:
        c = sum(1 for _, t, a, _, _ in rows if norm(a) == norm(t))
        o = sum(1 for _, t, _, b, _ in rows if norm(b) == norm(t))
        print(f"\n{'='*64}\n转写完全正确:  CTC {c}/{len(rows)}   自己的解码器 {o}/{len(rows)}"
              f"   平均 {np.mean([r[4] for r in rows]):.0f} ms/条\n{'='*64}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
