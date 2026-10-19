# -*- coding: utf-8 -*-
r"""定位 int8 产物加载到底把内存花在哪一步

用法: E:\voice_agent_build\Scripts\python.exe probe_load.py
"""
import ctypes
import ctypes.wintypes as wt
import gc
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch.set_num_threads(4)

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


def mem():
    c = PMC()
    c.cb = ctypes.sizeof(c)
    _p.GetProcessMemoryInfo(_k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1048576, c.PagefileUsage / 1048576


_prev = [0.0]


def stage(label):
    gc.collect()
    rss, commit = mem()
    print(f"  {label:44s} RSS {rss:7.0f} MB (+{rss-_prev[0]:6.0f})   提交 {commit:7.0f} MB")
    _prev[0] = rss
    return rss


from transformers import BertConfig, BertModel                    # noqa: E402
from torch.ao.quantization import quantize_dynamic, default_dynamic_qconfig  # noqa: E402

P = HERE / "model" / "intent_int8b.pt"
stage("0 起点")
t0 = time.perf_counter()
ck = torch.load(str(P), map_location="cpu", weights_only=False)
stage(f"1 torch.load({P.name}, {P.stat().st_size/1e6:.0f}MB)  {time.perf_counter()-t0:.1f}s")

cfg = BertConfig(**ck["bert_config"])
t0 = time.perf_counter()
bert = BertModel(cfg, add_pooling_layer=False)
stage(f"2 BertModel(cfg) 构造 fp32 壳  {time.perf_counter()-t0:.1f}s  ← 只看这一步")

emb = dict(ck["bert_emb"])
emb["word_embeddings.weight"] = ck["word_emb"]
bert.embeddings.load_state_dict(emb, strict=False)
stage("3 载入 embeddings（fp32 词表 65MB）")

old_enc = bert.encoder
bert.encoder = ck["Benc_module"]
del old_enc
stage("4 encoder 换成 int8 模块（旧 fp32 encoder 变垃圾）")

del emb
ck2 = {k: v for k, v in ck.items() if k not in ("A_module", "Benc_module")}
del ck
gc.collect()
stage("5 del ck（保留 A_module/Benc_module，其余字典丢掉）")
# 把 A_module 也放进来，模拟真实持有
A = ck2.pop("A_module", None)
stage("6 之后")

print(f"\n  结论线索：第 2 步（构造 fp32 壳）如果涨了几百 MB，而第 4/5 步不掉，")
print(f"            就说明是『先分配 fp32、再换掉』被分配器扣住了。")

try:
    h = _k.GetCurrentProcess()
    _k.SetProcessWorkingSetSize(h, ctypes.c_size_t(-1), ctypes.c_size_t(-1))
    stage("7 SetProcessWorkingSetSize(-1,-1) 之后")
except Exception as e:
    print("  trim 失败:", e)
