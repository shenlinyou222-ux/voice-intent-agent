# -*- coding: utf-8 -*-
r"""动态加载生命周期验证：启动不装 → 需要时装 → 空闲后卸载（RSS 与耗时）

用法: E:\voice_agent_build\Scripts\python.exe test_lazy.py
"""
import ctypes
import ctypes.wintypes as wt
import sys
import time
from pathlib import Path

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


def rss():
    c = PMC()
    c.cb = ctypes.sizeof(c)
    _p.GetProcessMemoryInfo(_k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1048576, c.PagefileUsage / 1048576


def line(tag, t0=None):
    r, cm = rss()
    extra = f"  {time.time()-t0:.1f}s" if t0 else ""
    print(f"  {tag:34s} RSS {r:7.0f} MB   提交 {cm:7.0f} MB{extra}")
    return r


from voice_agent import Agent, Config              # noqa: E402

print("=" * 78)
print("  动态加载生命周期")
print("=" * 78)
line("0 只 import（没装模型）")
cfg = Config(str(HERE / "intent_config.json"))
a = Agent(cfg, device="cpu")
line("1 造好 Agent（仍未装模型）")

t0 = time.time()
a.ensure_loaded()
line("2 ensure_loaded() 装好", t0)

t0 = time.time()
a.unload_idle(0)          # idle_s=0 → 立刻卸载
line("3 unload_idle(0) 卸载后", t0)

t0 = time.time()
a.ensure_loaded()
line("4 再次 ensure_loaded()", t0)
print(f"\n  配置：lazy_load={a.lazy}  idle_unload_s={a.idle_unload_s}  "
      f"preload_on_onset={a.preload_on_onset}")
