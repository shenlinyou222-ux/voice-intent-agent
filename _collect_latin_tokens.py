# -*- coding: utf-8 -*-
r"""收集真机 ASR 实际输出过的拉丁 token（危险子集）—— 驱动 bad_words 屏蔽表

口径：本地 OwnASR（CPU）跑 _hot_wavs 20 条 + utt 抽样，提取拉丁 run → token id。
本地直跑比生产 vLLM 拉丁残余更多，覆盖更保守（宁多勿漏）。
"""
import io
import json
import re
import sys
import wave
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"C:\Users\user\Desktop\ds harness\voice_agent")

import numpy as np  # noqa: E402

from asr_own import OwnASR  # noqa: E402

HERE = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
A_DIR = r"E:\models\qwen3-asr-0.6b-official"

LATIN_RUN = re.compile(r"[a-zA-Z]{2,}")   # ≥2 字母的拉丁 run


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(path)
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def main():
    asr = OwnASR(model_dir=A_DIR, device="cpu", dtype="float32",
                 language="Chinese", max_new_tokens=64)
    asr.load()
    tokz = asr.proc.tokenizer

    wavs = sorted((HERE / "_hot_wavs").glob("h*.wav"))
    utt_rows = [json.loads(l) for l in
                io.open(HERE / "utt" / "index.jsonl", encoding="utf-8")
                if l.strip()]
    utt_wavs = [HERE / "utt" / r["wav"] for r in utt_rows
                if r.get("src") == "mic" and (HERE / "utt" / r["wav"]).exists()]
    all_wavs = wavs + utt_wavs[::40]        # utt 抽 1/40

    tok_counter = Counter()                 # 拉丁 token 串 → 次数
    for p in all_wavs:
        wav, _ = read_wav(p)
        txt = asr.transcribe(wav)
        runs = LATIN_RUN.findall(txt or "")
        for r in runs:
            # 找组成它的 token id（贪心：优先最长 token）
            ids = tokz.encode(r, add_special_tokens=False)
            for tid in ids:
                tok_counter[tokz.decode([tid]).strip()] += 1

    print(f"扫 {len(all_wavs)} 条，拉丁 token 子集 {len(tok_counter)} 个")
    for tok, n in tok_counter.most_common(60):
        print(f"  {n:3d}  {tok!r}")


if __name__ == "__main__":
    main()
