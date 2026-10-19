# -*- coding: utf-8 -*-
r"""完整内存报告：4.6 GB 到底花在哪

分段量 RSS（每载入一步量一次）+ 逐模块参数字节核对 + 验证"保留 safetensors 副本"
到底占多少、能不能回收。

用法: E:\voice_agent_build\Scripts\python.exe mem_report.py
"""
import ctypes
import ctypes.wintypes as wt
import gc
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
D_ASR = r"E:\models\qwen3-asr-0.6b-official"

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
    d = cur - _prev[0]
    _prev[0] = cur
    print(f"  {label:44s} {cur:8.0f} MB   (+{d:7.0f})")
    return cur


def mod_mb(mod):
    n = sum(p.numel() for p in mod.parameters())
    return n, n * 4 / 1e6


print("=" * 78)
print("  A. 分段 RSS（每步 gc 后测量）")
print("=" * 78)
print(f"  {'阶段':44s} {'RSS':>11s} {'增量':>10s}")
stage("0  python 刚启动")
import torch                                                        # noqa: E402
torch.set_num_threads(4)
stage("1  + import torch")
import transformers                                                 # noqa: E402
stage("2  + import transformers")
from transformers import AutoProcessor                              # noqa: E402
proc = AutoProcessor.from_pretrained(D_ASR)
stage("3  + Qwen3-ASR 处理器/分词器（151,643 词表+151,291 merges）")

from model_runner import IntentModel                                # noqa: E402
m = IntentModel(model_dir=str(HERE / "model"), device="cpu", verbose=False)
stage("4  + 意图模型（A + ctc_head + dmeta-B + 头）")

sd_bytes = sum(t.numel() * t.element_size() for t in m._sd.values()) / 1e6
print(f"      ↳ 它同时保留了 safetensors 原始字典 _sd = {sd_bytes:.0f} MB")
ck_bytes = sum(v.numel() * v.element_size() for v in m._ck.values()
               if hasattr(v, "numel")) / 1e6
print(f"      ↳ 以及 intent.pt 字典 _ck = {ck_bytes:.0f} MB")

m._sd = None
del sd_bytes
stage("5  + del m._sd（回收那份常驻副本）")

from asr_own import OwnASR                                          # noqa: E402
asr = OwnASR(share_encoder=m.A)
asr.load()
stage("6  + ASR 解码器（共享 A）")

with wave.open(str(sorted((HERE / "wavs").glob("train_launch_app_*.wav"))[0]), "rb") as w:
    raw = w.readframes(w.getnframes())
wav = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
asr.transcribe(wav)
cur, peak = mem()
stage("7  + 跑一次转写（11s 长句的 KV/激活）")
print(f"\n  峰值 RSS {peak:.0f} MB")

print()
print("=" * 78)
print("  B. 权重账（参数 × 4 字节，fp32）")
print("=" * 78)
rows = [
    ("A  audio_tower（A/B 两侧共享，只算一份）", *mod_mb(m.A)),
    ("ctc_head  896→6630（自训）", *mod_mb(m.ctc)),
    ("dmeta-B  12 层 text encoder", *mod_mb(m.bert.encoder)),
    ("dmeta-B  词嵌入表 E_ALL（=word_embeddings 本体，不额外占）",
     m.E_ALL.numel(), 0),
    ("dmeta-B  position/token_type/LayerNorm", 0.79e6, 0.79e6 * 4 / 1e6),
    ("E_PAD  6,628 个交集字符的查表行", m.E_PAD.numel(), m.E_PAD.numel() * 4 / 1e6),
    ("intent_head  1536→8", *mod_mb(m.head)),
]
dec_total = sum(p.numel() for p in asr.model.parameters())
shared_a = sum(p.numel() for p in m.A.parameters())
rows.append(("解码器 Qwen3 28 层 + 投影（已扣掉共享的 A）",
             dec_total - shared_a, (dec_total - shared_a) * 4 / 1e6))
rows.append(("解码器 embed_tokens/lm_head（tied，只一份）",
             151936 * 1024, 151936 * 1024 * 4 / 1e6))
tot_p = sum(r[1] for r in rows)
tot_mb = sum(r[2] for r in rows)
print(f"  {'部分':52s} {'参数':>12s} {'MB':>8s}")
for name, n, mb in rows:
    print(f"  {name:52s} {n/1e6:9.2f} M {mb:8.0f}")
print(f"  {'合计（权重）':52s} {tot_p/1e6:9.2f} M {tot_mb:8.0f}")
print()
print("  实测 RSS 与权重账的差 = 运行时（torch/python 解释器 + 两个分词器 +")
print("  激活/KV/分句缓冲 + 分配器不还给系统的页）")
