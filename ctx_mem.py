# -*- coding: utf-8 -*-
r"""量化 n_ctx 对内存的影响（KV cache 是按 n_ctx 全量预分配的）

用法: python ctx_mem.py 4096
      python ctx_mem.py 512
"""
import ctypes
import ctypes.wintypes as wt
import gc
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_k = ctypes.WinDLL("kernel32")
_p = ctypes.WinDLL("psapi")


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


_k.GetCurrentProcess.restype = wt.HANDLE
_p.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
_p.GetProcessMemoryInfo.restype = wt.BOOL


def rss():
    c = PMC()
    c.cb = ctypes.sizeof(c)
    _p.GetProcessMemoryInfo(_k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1048576


n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
r0 = rss()
print(f"n_ctx = {n_ctx}")
from asr_gguf import GGUFASR                       # noqa: E402
asr = GGUFASR(n_ctx=n_ctx)
t0 = time.perf_counter()
asr.load()
r1 = rss()
print(f"  载入后 RSS {r1:.0f} MB（载入前 {r0:.0f}，+{r1-r0:.0f}）  "
      f"{time.perf_counter()-t0:.1f}s")

wavs = sorted((HERE / "wavs").glob("train_exact*.wav"))[:3]
tot_t = 0.0
for f in wavs:
    with wave.open(str(f), "rb") as w:
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    t = asr.transcribe(a)
    tot_t += asr.last_ms
    r2 = rss()
    print(f"    {f.name[:30]:32s} {asr.last_ms:6.0f}ms {asr.last_tokens:3d}tok → "
          f"「{t[:26]}」   RSS {r2:.0f}")
print(f"  3 条合计 {tot_t:.0f} ms   最终 RSS {rss():.0f} MB")

# 理论 KV
layers, kv_heads, head_dim, bpe = 28, 8, 128, 2
print(f"  理论 KV = 2×{layers}×{n_ctx}×{kv_heads}×{head_dim}×{bpe}B = "
      f"{2*layers*n_ctx*kv_heads*head_dim*bpe/1e6:.0f} MB")
