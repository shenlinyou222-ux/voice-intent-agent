# -*- coding: utf-8 -*-
r"""意图精度对比：CTC 文本（现在 built 走的） vs ASR 自己解码器的文本

同一批 val 音频，意图头不变，只换 V3 那一路的文本来源。
—— 这就是「ASR 解码 → 文本 → dmeta → 意图头」这条链路的直接验证。

用法: E:\voice_agent_build\Scripts\python.exe eval_intent_asr.py [条数]
"""
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def load_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def norm(s):
    return "".join(ch for ch in str(s) if not ch.isspace() and ch not in "。，、！？.,!?")


def main():
    val = []
    for line in io.open(HERE / "intent_dataset.jsonl", encoding="utf-8"):
        j = json.loads(line)
        if j.get("split") == "val" and (HERE / "wavs" / f"{j['id']}.wav").exists():
            val.append(j)
    val = val[::max(1, len(val) // N)][:N]
    print(f"val 抽样 {len(val)} 条\n")

    from model_runner import IntentModel
    m = IntentModel(model_dir=str(HERE / "model"), device="cpu")
    from asr_own import OwnASR
    own = OwnASR()

    c_ok = o_ok = 0
    print(f"{'真值意图':14s} {'CTC文本':26s} {'CTC判':12s} {'解码器文本':26s} {'解码器判':12s}")
    print("-" * 96)
    for j in val:
        wav, sr = load_wav(HERE / "wavs" / f"{j['id']}.wav")
        r1 = m.predict(wav)                       # 文本来自 CTC
        txt = own.transcribe(wav)
        r2 = m.predict(wav, text_override=txt)    # 文本来自 ASR 解码器
        ok1 = r1["intent"] == j["intent"]
        ok2 = r2["intent"] == j["intent"]
        c_ok += ok1
        o_ok += ok2
        print(f"{j['intent']:14s} {r1['text'][:24]:26s} "
              f"{r1['intent'] + ('✅' if ok1 else '❌'):12s} "
              f"{txt[:24]:26s} {r2['intent'] + ('✅' if ok2 else '❌'):12s}")
    n = len(val)
    print("-" * 96)
    print(f"意图精度:  CTC 文本 {c_ok}/{n} = {c_ok/n*100:.1f}%   "
          f"ASR 解码器文本 {o_ok}/{n} = {o_ok/n*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
