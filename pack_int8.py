# -*- coding: utf-8 -*-
r"""离线把意图模型打成 int8 产物（**不再让进程装 fp32**）

为什么要离线打：前面实测过 —— 先装 fp32 再在进程内量化，释放的页不还给系统，
RSS 反而涨。只有「进程直接加载 int8」才真正省内存。

产物：model/intent_int8.pt
   A_q      : audio_tower 的动态 int8 state_dict（Linear+Conv1d）
   Benc_q   : dmeta-B encoder 的动态 int8 state_dict（Linear）
   word_emb : 词嵌入表（保持 fp32 —— E_ALL/E_PAD 查表要浮点）
   bert_emb : position/token_type/LayerNorm（fp32）
   head_w/b : **用 int8 特征重训过的意图头**（就是你们设计里那 0.2 秒的活）
   配置     : asr/bert 的 config、SHARED 字符表、INTENTS/TAU/R

用法: E:\voice_agent_build\Scripts\python.exe pack_int8.py
"""
import gc
import io
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch.set_num_threads(4)

from model_runner import IntentModel                        # noqa: E402
from torch.ao.quantization import (quantize_dynamic, default_dynamic_qconfig)  # noqa: E402

QLIN = {torch.nn.Linear: default_dynamic_qconfig}
QCONV = {torch.nn.Linear: default_dynamic_qconfig, torch.nn.Conv1d: default_dynamic_qconfig}
OUT = HERE / "model" / "intent_int8.pt"


def load_wav(p):
    with wave.open(str(p), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def collect():
    tr, va = [], []
    for line in io.open(HERE / "intent_dataset.jsonl", encoding="utf-8"):
        j = json.loads(line)
        p = HERE / "wavs" / f"{j['id']}.wav"
        if not p.exists():
            continue
        (tr if j.get("split") == "train" else va).append((j, p))
    return tr, va


print("=" * 78)
print("  ① 载入 fp32 意图模型")
print("=" * 78)
m = IntentModel(model_dir=str(HERE / "model"), device="cpu", verbose=False)
tr, va = collect()
print(f"  train {len(tr)} 条 / val {len(va)} 条")
fp32_head_w = m.head.weight.detach().clone()
fp32_head_b = m.head.bias.detach().clone()

print("\n" + "=" * 78)
print("  ② 量化 A + dmeta-B encoder（都是 Linear/Conv1d，词表保持 fp32）")
print("=" * 78)
t0 = time.perf_counter()
m.A = quantize_dynamic(m.A, QCONV, dtype=torch.qint8)
m.bert.encoder = quantize_dynamic(m.bert.encoder, QLIN, dtype=torch.qint8)
gc.collect()
print(f"  量化完成 {time.perf_counter()-t0:.1f}s")

print("\n" + "=" * 78)
print("  ③ 用【int8 特征】抽 train/val 特征向量")
print("=" * 78)


def feats(items):
    X, Y, ok = [], [], 0
    t0 = time.perf_counter()
    for i, (j, p) in enumerate(items):
        w, _ = load_wav(p)
        r = m.predict(w)
        X.append(m.last_feat.clone())
        idx = m.INTENTS.index(j["intent"]) if j["intent"] in m.INTENTS else -1
        Y.append(idx)
        ok += (r["intent"] == j["intent"])
        if (i + 1) % 400 == 0:
            print(f"    {i+1}/{len(items)}  {time.perf_counter()-t0:.0f}s")
    X = torch.stack(X)
    Y = torch.tensor(Y)
    print(f"  {len(items)} 条  {time.perf_counter()-t0:.0f}s  "
          f"特征 {tuple(X.shape)}  （旧头在 int8 特征上的精度 {ok}/{len(items)}"
          f" = {ok/len(items)*100:.1f}%）")
    return X, Y


Xtr, Ytr = feats(tr)
Xva, Yva = feats(va)

print("\n" + "=" * 78)
print("  ④ 用 int8 特征重训意图头（线性层，就是那 0.2 秒的活）")
print("=" * 78)
head = torch.nn.Linear(Xtr.shape[1], len(m.INTENTS))
opt = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-5)
t0 = time.perf_counter()
for ep in range(400):
    opt.zero_grad()
    loss = F.cross_entropy(head(Xtr), Ytr)
    loss.backward()
    opt.step()
print(f"  训练 {time.perf_counter()-t0:.2f}s  最终 loss {loss.item():.4f}")

with torch.no_grad():
    pv = head(Xva).argmax(-1)
    acc_new = (pv == Yva).float().mean().item()
    m.head.weight.data = fp32_head_w
    m.head.bias.data = fp32_head_b
    pv_old = m.head(Xva).argmax(-1)
    acc_old = (pv_old == Yva).float().mean().item()
print(f"  val 精度：  旧头(fp32训练)+int8特征 = {acc_old*100:.1f}%"
      f"    新头(int8特征重训) = {acc_new*100:.1f}%")

print("\n" + "=" * 78)
print("  ⑤ 打包 model/intent_int8.pt")
print("=" * 78)
bert_cfg = m.bert.config.to_dict() if hasattr(m.bert, "config") else None
asr_cfg = json.loads((Path(r"E:\models\qwen3-asr-0.6b-official") / "config.json")
                     .read_text(encoding="utf-8"))["thinker_config"]["audio_config"]
blob = {
    "fmt": "int8-v1",
    "intents": m.INTENTS, "tau": m.TAU, "R": m.R,
    "shared_chars": m.SHARED,
    "asr_audio_config": asr_cfg,
    "bert_config": bert_cfg,
    "A_q": m.A.state_dict(),
    "Benc_q": m.bert.encoder.state_dict(),
    "bert_emb": {k: m.bert.embeddings.state_dict()[k] for k in
                 ("position_embeddings.weight", "token_type_embeddings.weight",
                  "LayerNorm.weight", "LayerNorm.bias")},
    "word_emb": m.bert.embeddings.word_embeddings.weight.data.clone(),
    "ctc_head": m.ctc.weight.data.clone(), "ctc_bias": m.ctc.bias.data.clone(),
    "head_w": head.weight.data.clone(), "head_b": head.bias.data.clone(),
    "val_acc_int8": acc_new,
}
torch.save(blob, OUT)
mb = OUT.stat().st_size / 1e6
print(f"  → {OUT}   {mb:.0f} MB")
for k in ("A_q", "Benc_q", "word_emb", "ctc_head"):
    v = blob[k]
    n = sum(t.numel() * t.element_size() for t in v.values()) if isinstance(v, dict) \
        else v.numel() * v.element_size()
    print(f"     {k:10s} {n/1e6:7.0f} MB")
print(f"\n  对比 fp32（bundle 745MB + intent.pt 23MB = 768MB）→ {mb:.0f} MB"
      f"  （省 {(768-mb):.0f} MB）")
