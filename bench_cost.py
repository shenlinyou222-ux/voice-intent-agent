# -*- coding: utf-8 -*-
r"""开销基准：意图模型 vs ASR 自己的解码器 —— 内存 + 每次转写的 CPU/墙钟

Windows 没有 resource 模块，用 psapi 直接读工作集。

用法: E:\voice_agent_build\Scripts\python.exe bench_cost.py
"""
import ctypes
import ctypes.wintypes as wt
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def mem():
    c = PMC()
    c.cb = ctypes.sizeof(c)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1048576, c.PeakWorkingSetSize / 1048576


def load_wav(p):
    with wave.open(str(p), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def bench(fn, *a, **kw):
    c0, w0 = time.process_time(), time.perf_counter()
    r = fn(*a, **kw)
    return r, (time.perf_counter() - w0) * 1000, (time.process_time() - c0) * 1000


def main():
    torch.set_num_threads(4)        # 与线上 app 一致（实测 4 线程比 14 线程省一半 CPU）
    print(f"torch {torch.__version__}  intra-op 线程={torch.get_num_threads()}  "
          f"dtype=float32  device=cpu")
    r0, _ = mem()
    print(f"起步（python+torch，无模型）RSS {r0:.0f} MB\n")

    from model_runner import IntentModel
    m = IntentModel(model_dir=str(HERE / "model"), device="cpu", verbose=False)
    r1, _ = mem()
    print(f"① 意图模型（A 186.38M + ctc_head + dmeta-B 85.05M + 头）载入后 RSS {r1:.0f} MB")

    from asr_own import OwnASR
    asr = OwnASR(share_encoder=m.A)      # 共享 A
    asr.load()
    r2, p2 = mem()
    print(f"② 再载入 ASR 自己的解码器（782.4M）→ RSS {r2:.0f} MB   "
          f"（+{r2-r1:.0f} MB）")
    print(f"   两者合计：当前 {r2:.0f} MB / 峰值 {p2:.0f} MB\n")

    n_i = (sum(p.numel() for p in m.A.parameters())
           + sum(p.numel() for p in m.bert.parameters())
           + sum(p.numel() for p in m.ctc.parameters())
           + sum(p.numel() for p in m.head.parameters()))
    n_o = sum(p.numel() for p in asr.model.parameters())
    print(f"   意图模型 {n_i/1e6:.2f} M 参数 ×4B = {n_i*4/1e6:.0f} MB")
    print(f"   解码器   {n_o/1e6:.2f} M 参数 ×4B = {n_o*4/1e6:.0f} MB")
    print(f"   ⚠️ audio_tower 186.38M 被装了【两遍】（意图模型的 A + 解码器内的 "
          f"audio_tower）→ 重复占 {186.38*4:.0f} MB\n")

    shorts = sorted((HERE / "wavs").glob("train_launch_app_*.wav"))[:1]
    w_short, _ = load_wav(shorts[0])
    cases = [(w_short, f"短句 {len(w_short)/16000:.1f}s"),
             (np.concatenate([w_short] * 4), "拼长句 ~11s")]

    print(f"{'音频':14s} {'路径':16s} {'墙钟ms':>7s} {'CPU ms':>7s} {'CPU/墙钟':>8s} {'token':>6s}")
    print("-" * 70)
    for wav, name in cases:
        _, w1, c1 = bench(m.predict, wav)
        print(f"{name:14s} {'意图预测(CTC)':16s} {w1:7.0f} {c1:7.0f} {c1/max(w1,1e-9):8.2f} {'-':>6s}")
        _, w2, c2 = bench(asr.transcribe, wav)
        print(f"{'':14s} {'ASR 解码器':16s} {w2:7.0f} {c2:7.0f} {c2/max(w2,1e-9):8.2f} {asr.last_tokens:6d}")
        keep = asr.max_new_tokens
        asr.max_new_tokens = 1
        _, w3, c3 = bench(asr.transcribe, wav)
        asr.max_new_tokens = keep
        print(f"{'':14s} {' └ 仅编码+首token':16s} {w3:7.0f} {c3:7.0f} {'':8s} {'1':>6s}"
              f"   → 解码 {asr.last_tokens if asr.last_tokens>0 else 0} token 净花 "
              f"{c2-c3:.0f} CPU ms")

    r3, p3 = mem()
    print(f"\n跑完：当前 RSS {r3:.0f} MB / 峰值 {p3:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
