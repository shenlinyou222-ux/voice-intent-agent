# -*- coding: utf-8 -*-
"""ASR 延迟对照：同一批音频，vLLM(1.7B fp8) 实测 vs 日志里 llama-asr(1.7B Q4) 的实测值。

为什么要这么量：我之前拿 1.7s 的 llama 数字去比 3.82s 的 vLLM 数字，得出"vLLM 更慢"的
错误结论 —— 这就是"拿不同长度的数字比大小"。这里每条都按同一段音频比。
llama 侧的数字来源（都是 app 日志里的 HTTP 往返 ms）：
  1.72s → 120ms (16:49:52)   1.76s → 85ms (16:50:00)
  3.82s → 345ms (17:37:56)   15.54s → 324ms (21:xx _dual_replay 的 B 路)
"""
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _cut_align import asr  # noqa: E402

CASES = [
    ("utt/164952_mic_dictation_现在应该可以了.wav", 120),
    ("utt/165000_mic_dictation_现在可以了吗.wav", 85),
    ("utt/173756_mic_dictation_试一试现在的战力是不是够了.wav", 345),
    ("_dual/211710_dual.wav", 324),
]
REP = 3
print(f"  目标端点见 intent_config.json 的 asr_http_url（现在是 vLLM 30805）\n")
print(f"  {'音频':<34}{'时长':>7}{'vLLM中位':>10}{'llama':>8}{'差':>9}   文本")
rows = []
for rel, llama in CASES:
    p = HERE / rel
    if not p.exists():
        print(f"  {rel} 不存在")
        continue
    with wave.open(str(p), "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    ts, txt = [], ""
    for _ in range(REP):
        t0 = time.time()
        txt = asr(x, sr=sr, gain=8.0)
        ts.append((time.time() - t0) * 1000)
    med = float(np.median(ts))
    name = Path(rel).name[:32]
    print(f"  {name:<34}{len(x)/sr:>6.2f}s{med:>9.0f}ms{llama:>7}ms"
          f"{(med-llama)/llama*100:>8.0f}%   {txt[:34]}")
    rows.append((len(x) / sr, med, llama))
if rows:
    print(f"\n  ── 归一化（每 1 秒音频的 ASR 耗时）──")
    for dur, v, l in rows:
        print(f"     {dur:5.2f}s  vLLM {v/dur:5.0f}ms/s   llama {l/dur:5.0f}ms/s")
