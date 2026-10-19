# -*- coding: utf-8 -*-
"""量一下砍掉音频塔到底省了多少内存（同进程内对比，用 RSS + 私有提交）。

Windows 上 RSS 会被 mmap/页缓存影响，所以同时看 WorkingSet 和 PrivateMemory。
"""
import ctypes
import ctypes.wintypes as wt
import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

HERE = Path(__file__).parent


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def rss_mb():
    """→ (WorkingSet MB, 私有提交 MB)。不依赖 psutil。"""
    p = _PMC()
    p.cb = ctypes.sizeof(_PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(p), p.cb)
    return p.WorkingSetSize / 1e6, p.PagefileUsage / 1e6


def main():
    import torch  # noqa: F401
    from model_runner import IntentModel

    gc.collect()
    r0, p0 = rss_mb()
    print(f"基线（已 import torch）           RSS {r0:7.1f} MB  私有 {p0:7.1f} MB")

    print("\n加载【完整产物】intent_int8b.pt（含音频塔 + CTC 头）...")
    m_full = IntentModel(model_dir=str(HERE / "model"), device="cpu",
                         int8=True, text_only=False, verbose=False)
    gc.collect()
    r1, p1 = rss_mb()
    print(f"  加载后                        RSS {r1:7.1f} MB  私有 {p1:7.1f} MB"
          f"   （Δ RSS {r1-r0:+.1f} / Δ 私有 {p1-p0:+.1f}）")

    del m_full
    gc.collect()
    r2, p2 = rss_mb()
    print(f"  卸载后                        RSS {r2:7.1f} MB  私有 {p2:7.1f} MB")

    print("\n加载【纯文本产物】intent_text.pt（无音频塔 / 无 CTC 头）...")
    m_txt = IntentModel(model_dir=str(HERE / "model"), device="cpu",
                        int8=True, text_only=True, verbose=False)
    gc.collect()
    r3, p3 = rss_mb()
    print(f"  加载后                        RSS {r3:7.1f} MB  私有 {p3:7.1f} MB"
          f"   （Δ RSS {r3-r2:+.1f} / Δ 私有 {p3-p2:+.1f}）")

    print(f"\n{'='*66}")
    print(f"完整产物净占用  RSS {r1-r0:7.1f} MB / 私有 {p1-p0:7.1f} MB")
    print(f"纯文本产物净占用 RSS {r3-r2:7.1f} MB / 私有 {p3-p2:7.1f} MB")
    print(f"→ 省            RSS {(r1-r0)-(r3-r2):7.1f} MB / "
          f"私有 {(p1-p0)-(p3-p2):7.1f} MB")

    # 顺便量推理延迟
    import time
    t = "打开知乎，我们看一下吧。"
    for _ in range(3):
        m_txt.predict_text(t)
    t0 = time.perf_counter()
    for _ in range(20):
        r = m_txt.predict_text(t)
    dt = (time.perf_counter() - t0) / 20 * 1000
    print(f"\n纯文本路推理：{dt:.1f} ms/句  → {r['intent']} {r['conf']:.3f}")
    print(f"（完整产物要额外跑一次音频塔 ≈ 80~120ms）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
