# -*- coding: utf-8 -*-
r"""验证「CPU 推理吃带宽」：同一段音频，三种量化 → 每 token 耗时是否随字节数线性增长

模型字节数（LLM 部分）：Q4_K_M 462MB / Q8_0 767MB / bf16 1439MB
（mmproj 都是同一份 Q8，音频编码开销相同，可以扣掉）

顺带测：EmptyWorkingSet（我们加的工作集整理）之后第一句要付多少 fault 代价。

用法: E:\voice_agent_build\Scripts\python.exe bw_test.py
"""
import ctypes
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
M = Path(r"C:\Users\user\Desktop\ds harness\aigc_demo_framework\models\qwen3-asr-gguf")
MMPROJ = M / "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"
CASES = [("Q4_K_M", M / "Qwen3-ASR-0.6B-Q4_K_M.gguf", 4),
         ("Q8_0", M / "Qwen3-ASR-0.6B-Q8_0.gguf", 4),
         ("bf16", M / "Qwen3-ASR-0.6B-bf16.gguf", 4),
         ("Q4_K_M/8线程", M / "Qwen3-ASR-0.6B-Q4_K_M.gguf", 8)]


def load_wav(p):
    with wave.open(str(p), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


wavs = [load_wav(f) for f in
        sorted((HERE / "wavs").glob("train_exact*.wav"))[:3]]

print(f"{'量化':14s} {'文件MB':>7s} {'线程':>4s} {'载入s':>6s} "
      f"{'编码ms':>7s} {'生成ms':>7s} {'token':>5s} {'ms/token':>9s} {'GB/s(按字节)':>12s}")
print("-" * 92)
from asr_gguf import GGUFASR                                   # noqa: E402

for name, gguf, nth in CASES:
    asr = GGUFASR(gguf=gguf, mmproj=MMPROJ, n_threads=nth)
    t0 = time.perf_counter()
    asr.load()
    load_s = time.perf_counter() - t0
    mb = gguf.stat().st_size / 1e6
    enc_ms = gen_ms = 0.0
    toks = 0
    for w in wavs:
        asr.transcribe(w)                 # 预热（页调入）
    for w in wavs:
        asr.transcribe(w)
        # 用一次只出 1 token 的调用估编码开销
        keep = asr.max_new_tokens
        asr.max_new_tokens = 1
        asr.transcribe(w)
        e = asr.last_ms
        asr.max_new_tokens = keep
        asr.transcribe(w)
        gen_ms += asr.last_ms - e
        toks += asr.last_tokens - 1
        enc_ms += e
    enc_ms /= len(wavs)
    per = gen_ms / max(toks, 1)
    gbs = (mb / 1024) / (per / 1000) if per > 0 else 0
    print(f"{name:14s} {mb:7.0f} {nth:4d} {load_s:6.1f} {enc_ms:7.0f} "
          f"{gen_ms:7.0f} {toks:5d} {per:9.1f} {gbs:12.1f}")

print("\n--- EmptyWorkingSet（工作集整理）对首句的代价 ---")
asr = GGUFASR(gguf=M / "Qwen3-ASR-0.6B-Q4_K_M.gguf", mmproj=MMPROJ, n_threads=4)
asr.load()
for w in wavs:
    asr.transcribe(w)
t_before = []
for w in wavs:
    asr.transcribe(w)
    t_before.append(asr.last_ms)
k = ctypes.WinDLL("kernel32")
p = ctypes.WinDLL("psapi")
k.GetCurrentProcess.restype = ctypes.c_void_p
p.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
p.EmptyWorkingSet(k.GetCurrentProcess())
t_after = []
for w in wavs:
    asr.transcribe(w)
    t_after.append(asr.last_ms)
print(f"  整理前 {[round(x) for x in t_before]} ms  →  整理后首轮 "
      f"{[round(x) for x in t_after]} ms")
