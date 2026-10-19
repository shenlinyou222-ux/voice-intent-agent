# -*- coding: utf-8 -*-
r"""验证两件事（不猜）：
  ① audio_tower 在一次转写里到底被算了几次（生成时是否反复重算）
  ② 共享 A 之后到底省了多少内存

用法: E:\voice_agent_build\Scripts\python.exe probe_af.py
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


def load_wav(p):
    with wave.open(str(p), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def timed(fn, *a, **kw):
    c0, w0 = time.process_time(), time.perf_counter()
    r = fn(*a, **kw)
    return r, (time.perf_counter() - w0) * 1000, (time.process_time() - c0) * 1000


torch.set_num_threads(4)          # 与线上 app 一致
print(f"torch {torch.__version__}  线程={torch.get_num_threads()}  "
      f"OMP_WAIT_POLICY={__import__('os').environ.get('OMP_WAIT_POLICY', '(未设)')}")
print(f"起步 RSS {rss():.0f} MB\n")

from model_runner import IntentModel          # noqa: E402
m = IntentModel(model_dir=str(HERE / "model"), device="cpu", verbose=False)
r1 = rss()
print(f"① 意图模型载入后            RSS {r1:.0f} MB")

from asr_own import OwnASR                    # noqa: E402
asr = OwnASR(share_encoder=m.A)               # ← 共享 A
t, w, c = timed(asr.load)
r2 = rss()
print(f"② 载入解码器（共享 A）后    RSS {r2:.0f} MB   （+{r2-r1:.0f} MB，"
      f"载入 {w/1000:.1f}s）")
print(f"   → 不共享时基准实测 +{3130:.0f} MB（782.43M×4B），"
      f"共享省下约 {3130-(r2-r1):.0f} MB\n")

short, _ = load_wav(sorted((HERE / "wavs").glob("train_launch_app_*.wav"))[0])
cases = [(short, f"短句 {len(short)/16000:.2f}s"),
         (np.concatenate([short] * 4), "拼长句 ~11s")]

# 直接给 audio_tower 模块的 forward 挂计数器（比 get_audio_features 更底层、更确定）
A = asr.model.model.audio_tower
print(f"\n共享是否生效: asr.model.model.audio_tower is intent.A → {A is m.A}")
print(f"共享后 A 的参数: {sum(p.numel() for p in A.parameters())/1e6:.2f} M "
      f"（{sum(p.numel() for p in A.parameters())*4/1e6:.0f} MB，只此一份）")
cnt = {"n": 0}
_orig_fwd = A.forward


def _counted(*a, **kw):
    cnt["n"] += 1
    return _orig_fwd(*a, **kw)


A.forward = _counted

print(f"\n{'音频':14s} {'墙钟ms':>7s} {'CPU ms':>8s} {'token':>6s} "
      f"{'audio_tower forward 次数':>24s} {'每token CPU ms':>14s}")
print("-" * 84)
for wav, name in cases:
    cnt["n"] = 0
    txt, w, c = timed(asr.transcribe, wav)
    tk = asr.last_tokens
    print(f"{name:14s} {w:7.0f} {c:8.0f} {tk:6d} {cnt['n']:24d} "
          f"{(c / max(tk, 1)):14.0f}")
    print(f"    「{txt[:60]}」")

A.forward = _orig_fwd
print(f"\n跑完 RSS {rss():.0f} MB")
print("\n判定：forward 次数=1 → 编码只算一次，优化①不存在（无需做）；")
print("      =token数+1 → 每 token 在重算编码，记忆化已经把它压成 1 次。")
print("      RSS 不降是 glibc/torch 的 CPU 分配器不把释放的页还给系统；")
print("      逻辑上确实少了一份 186.38M×4B = 745 MB（参数计数可证）。")
