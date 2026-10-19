# -*- coding: utf-8 -*-
r"""最终架构的内存分段报告（int8 意图模型 + llama.cpp GGUF ASR）

用法: E:\voice_agent_build\Scripts\python.exe mem_final.py
"""
import ctypes
import ctypes.wintypes as wt
import gc
import sys
import time
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


def mem():
    c = PMC()
    c.cb = ctypes.sizeof(c)
    _p.GetProcessMemoryInfo(_k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1048576, c.PeakWorkingSetSize / 1048576


_prev = [0.0]


def stage(label):
    gc.collect()
    cur, peak = mem()
    print(f"  {label:46s} {cur:8.0f} MB  (+{cur-_prev[0]:7.0f})   峰值 {peak:.0f}")
    _prev[0] = cur


print("=" * 84)
print("  A. 分段 RSS")
print("=" * 84)
stage("0  python 启动")
import torch                                    # noqa: E402
torch.set_num_threads(4)
stage("1  + import torch")
import transformers                             # noqa: E402
stage("2  + import transformers")

from model_runner import IntentModel            # noqa: E402
t0 = time.time()
m = IntentModel(model_dir=str(HERE / "model"), device="cpu", int8=True, verbose=False)
stage(f"3  + 意图模型 int8 产物（{time.time()-t0:.1f}s）")

from asr_gguf import GGUFASR                    # noqa: E402
t0 = time.time()
asr = GGUFASR()
asr.load()
stage(f"4  + llama.cpp GGUF ASR（{time.time()-t0:.1f}s）")

import wave                                     # noqa: E402
w = sorted((HERE / "wavs").glob("train_exact*.wav"))[:3]
for f in w:
    with wave.open(str(f), "rb") as ww:
        raw = ww.readframes(ww.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    asr.transcribe(a)
stage("5  + 跑 3 条转写（GGUF 页被调入）")
cur, peak = mem()
print(f"\n  最终 RSS {cur:.0f} MB   峰值 {peak:.0f} MB")

print()
print("=" * 84)
print("  B. 模型数据账（进程内实际驻留的权重）")
print("=" * 84)
gguf = 462 + 204
print(f"  {'ASR：llama.cpp GGUF（mmap，按需调页）':48s} {gguf:7.0f} MB"
      f"   （fp32 时 3,130 MB）")
print(f"  {'意图：A audio_tower int8':48s} {'198':>7s} MB   （fp32 739）")
print(f"  {'意图：dmeta-B encoder int8':48s} {'85':>7s} MB   （fp32 340）")
print(f"  {'意图：词嵌入表 fp32（查表要浮点）':48s} {'65':>7s} MB")
print(f"  {'意图：E_PAD 查表行 fp32':48s} {'20':>7s} MB")
print(f"  {'意图：ctc_head fp32':48s} {'24':>7s} MB")
print(f"  {'－':48s} {'-':>7s}")
print(f"  {'模型数据合计':48s} {gguf+198+85+65+20+24:7.0f} MB")
print(f"  {'其余 = torch/python/分词器/激活/分配器保留页':48s} "
      f"{cur-gguf-198-85-65-20-24:7.0f} MB")
