# -*- coding: utf-8 -*-
r"""进程内 int8 量化实测：ASR 解码器 + dmeta-B

背景：E:\prefix4_router\dmeta_quant 已证明 dmeta int8 几乎无损
      （cos 0.9972 / 准确率 0.8882 vs fp32 0.8868），但那只是离线评测，
      部署产物一直是 fp32。

计量方式（重要）：动态量化后的 Linear 权重放在 packed 参数里，遍历
`state_dict()` 会漏掉它们 —— 所以这里用【序列化落盘量真实字节】。

用法: E:\voice_agent_build\Scripts\python.exe quant_int8.py
"""
import gc
import io
import json
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch.set_num_threads(4)


def fp32_mb(mod):
    return sum(p.numel() * p.element_size() for p in mod.parameters()) / 1e6


def real_mb(mod):
    """序列化到临时文件量真实占用（量化权重也能算准）"""
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(mod.state_dict(), path)
        return os.path.getsize(path) / 1e6
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


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


print("=" * 78)
print("  ① 载入 fp32 基线")
print("=" * 78)
from model_runner import IntentModel                      # noqa: E402
m = IntentModel(model_dir=str(HERE / "model"), device="cpu", verbose=False)
from asr_own import OwnASR                               # noqa: E402
asr = OwnASR(share_encoder=m.A)
asr.load()

wavs = []
for f in sorted((HERE / "wavs").glob("train_exact*.wav"))[:6]:
    w, _ = load_wav(f)
    wavs.append((f.name, w))

parts = {"解码器 language_model": asr.model.model.language_model,
         "解码器 lm_head": asr.model.lm_head,
         "A audio_tower": asr.model.model.audio_tower,
         "dmeta-B": m.bert}
print()
base = {}
for k, mod in parts.items():
    base[k] = fp32_mb(mod)
    print(f"  {k:26s} fp32 {base[k]:8.0f} MB")
print(f"  {'E_PAD':26s} fp32 {m.E_PAD.numel()*4/1e6:8.0f} MB")
base["E_PAD"] = m.E_PAD.numel() * 4 / 1e6
print(f"  {'合计':26s} fp32 {sum(base.values()):8.0f} MB")

print("\n  fp32 跑 6 条 ...")
ref, w_fp32, c_fp32 = [], 0.0, 0.0
for name, w in wavs:
    t, wall, cpu = timed(asr.transcribe, w)
    ref.append(t)
    w_fp32 += wall
    c_fp32 += cpu
    print(f"    {name[:32]:34s} {wall:6.0f}ms {cpu:7.0f}CPUms  「{t[:38]}」")

print()
print("=" * 78)
print("  ② int8 动态量化（torch.ao.quantization.quantize_dynamic）")
print("=" * 78)
from torch.ao.quantization import (quantize_dynamic, default_dynamic_qconfig,
                                   float_qparams_weight_only_qconfig)   # noqa: E402
QSPEC = {torch.nn.Linear: default_dynamic_qconfig,
         torch.nn.Embedding: float_qparams_weight_only_qconfig}

t0 = time.perf_counter()
asr.model.model.language_model = quantize_dynamic(
    asr.model.model.language_model, QSPEC, dtype=torch.qint8)
print(f"  量化 language_model 完成")
# ⚠️ 量化 embed_tokens 会打断与 lm_head 的 tie：lm_head 会继续持有那张
#    fp32 旧表（622 MB）。所以必须把 lm_head 也量化，旧表才会被释放。
_t = type(asr.model.lm_head).__name__
asr.model.lm_head = quantize_dynamic(asr.model.lm_head, QSPEC, dtype=torch.qint8)
print(f"  量化 lm_head 完成（{_t} → {type(asr.model.lm_head).__name__}）")
# dmeta-B 只量化 encoder 的 Linear：词嵌入表要留给 E_ALL/E_PAD 查表用（必须是 fp32）
m.bert.encoder = quantize_dynamic(m.bert.encoder,
                                  {torch.nn.Linear: default_dynamic_qconfig},
                                  dtype=torch.qint8)
print(f"  量化 dmeta-B encoder 完成（词表保持 fp32 供查表）")
# A 的 Conv1d/Linear 试一把
try:
    asr.model.model.audio_tower = quantize_dynamic(
        asr.model.model.audio_tower,
        {torch.nn.Linear: default_dynamic_qconfig,
         torch.nn.Conv1d: default_dynamic_qconfig}, dtype=torch.qint8)
    a_quant_ok = True
    print(f"  量化 A audio_tower 完成")
except Exception as e:
    a_quant_ok = False
    print(f"  ⚠️ A 量化失败（保持 fp32）: {type(e).__name__}: {e}")
gc.collect()
print(f"  量化总耗时 {time.perf_counter()-t0:.1f}s")

after = {"解码器 language_model": real_mb(asr.model.model.language_model),
         "解码器 lm_head": real_mb(asr.model.lm_head),
         "A audio_tower": real_mb(asr.model.model.audio_tower),
         "dmeta-B": real_mb(m.bert),
         "E_PAD": m.E_PAD.numel() * 4 / 1e6}
print(f"\n  {'模块':26s} {'fp32':>9s} {'int8':>9s} {'倍率':>7s}")
for k in after:
    a, b = base[k], after[k]
    print(f"  {k:26s} {a:8.0f} MB {b:8.0f} MB {a/max(b,1e-9):6.2f}×")
print(f"  {'合计':26s} {sum(base.values()):8.0f} MB {sum(after.values()):8.0f} MB "
      f"{sum(base.values())/sum(after.values()):6.2f}×")

print("\n  int8 跑同样 6 条 ...")


def norm(s):
    """去标点/空格/大小写再比 —— 量化只会动到这类边角，不该按逐字比"""
    drop = "。，、！？；：,.!?;: \t"
    return "".join(c for c in str(s).lower() if c not in drop)


same = same_n = 0
w_i, c_i = 0.0, 0.0
for (name, w), r0 in zip(wavs, ref):
    t, wall, cpu = timed(asr.transcribe, w)
    w_i += wall
    c_i += cpu
    ok = (t.strip() == r0.strip())
    okn = (norm(t) == norm(r0))
    same += ok
    same_n += okn
    print(f"    {name[:32]:34s} {wall:6.0f}ms {cpu:7.0f}CPUms "
          f"{'逐字一致' if ok else ('去标点一致' if okn else '❌有实质差异')}  「{t[:38]}」")
n = len(wavs)
print(f"\n  逐字一致 {same}/{n}   去标点/空格/大小写后一致 {same_n}/{n}")
print(f"  墙钟 {w_fp32/n:.0f} → {w_i/n:.0f} ms（{w_fp32/max(w_i,1e-9):.2f}× 快）    "
      f"CPU {c_fp32/n:.0f} → {c_i/n:.0f} ms（{c_fp32/max(c_i,1e-9):.2f}× 省）")

print("\n  ③ 意图头（走量化后的 dmeta-B）12 条 val ...")
val = []
for line in io.open(HERE / "intent_dataset.jsonl", encoding="utf-8"):
    j = json.loads(line)
    if j.get("split") == "val" and (HERE / "wavs" / f"{j['id']}.wav").exists():
        val.append(j)
val = val[::max(1, len(val) // 12)][:12]
ok = 0
for j in val:
    w, _ = load_wav(HERE / "wavs" / f"{j['id']}.wav")
    r = m.predict(w)
    ok += (r["intent"] == j["intent"])
print(f"  量化后意图精度 {ok}/{len(val)} = {ok/len(val)*100:.1f}%"
      f"（fp32 基线 12/12 = 100%，见 eval_intent_asr.py）")
