# -*- coding: utf-8 -*-
r"""隔离实验：int8 到底能上哪一部分（意图精度角度）

意图头是在 fp32 特征上训的线性头。逐级量化，看每步对意图精度的影响：
  fp32 → A int8 → + 解码器 int8 → + dmeta-B int8

为了只测"特征侧"的影响，每条样本的文本用 fp32 解码器跑一次后【固定】下来
（text_override），这样换量化不会再改文本。

用法: E:\voice_agent_build\Scripts\python.exe quant_ab.py [条数]
"""
import gc
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch.set_num_threads(4)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24

from model_runner import IntentModel                       # noqa: E402
from asr_own import OwnASR                                # noqa: E402
from torch.ao.quantization import (quantize_dynamic, default_dynamic_qconfig,
                                   float_qparams_weight_only_qconfig)  # noqa: E402

QSPEC = {torch.nn.Linear: default_dynamic_qconfig,
         torch.nn.Embedding: float_qparams_weight_only_qconfig}
QLIN = {torch.nn.Linear: default_dynamic_qconfig}


def load_wav(p):
    with wave.open(str(p), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


m = IntentModel(model_dir=str(HERE / "model"), device="cpu", verbose=False)
asr = OwnASR(share_encoder=m.A)
asr.load()

val = []
for line in open(HERE / "intent_dataset.jsonl", encoding="utf-8"):
    j = json.loads(line)
    if j.get("split") == "val" and (HERE / "wavs" / f"{j['id']}.wav").exists():
        val.append(j)
val = val[::max(1, len(val) // N)][:N]

print(f"抽样 {len(val)} 条；先用 fp32 解码器固定文本 ...")
items = []
for j in val:
    w, _ = load_wav(HERE / "wavs" / f"{j['id']}.wav")
    txt = asr.transcribe(w)
    items.append((j, w, txt))
print("固定完毕\n")


def acc(tag):
    ok = 0
    for j, w, txt in items:
        r = m.predict(w, text_override=txt)
        ok += (r["intent"] == j["intent"])
    print(f"  {tag:44s} 意图精度 {ok}/{len(items)} = {ok/len(items)*100:5.1f}%")
    return ok


print("=" * 78)
acc("0 全 fp32（基线）")

asr.model.model.audio_tower = quantize_dynamic(
    asr.model.model.audio_tower,
    {torch.nn.Linear: default_dynamic_qconfig,
     torch.nn.Conv1d: default_dynamic_qconfig}, dtype=torch.qint8)
gc.collect()
acc("1 只有 A(audio_tower) int8")

asr.model.model.language_model = quantize_dynamic(
    asr.model.model.language_model, QSPEC, dtype=torch.qint8)
gc.collect()
acc("2 + 解码器 int8（文本已固定，理论上应无影响）")

m.bert.encoder = quantize_dynamic(m.bert.encoder, QLIN, dtype=torch.qint8)
gc.collect()
acc("3 + dmeta-B encoder int8")

print("\n  判定：第 2 行≈第 0 行 → 解码器量化对意图无害（只影响转写）；")
print("        第 1 行掉 → A 量化会扰动意图特征（A 是意图与转写共用的）；")
print("        第 3 行掉 → B 量化需要【用 int8 特征重训那个线性头】（0.2 秒的活）。")
