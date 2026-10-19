# -*- coding: utf-8 -*-
r"""把 intent_int8.pt 重打成「量化模块对象」版 —— 加载时不再物化 fp32

上一版的产物存的是 state_dict：加载时要先建 fp32 结构再量化，那些 fp32 页会留在
RSS 里（实测 RSS 没降）。改成直接 pickle 量化后的模块对象，加载即 int8。

用法: E:\voice_agent_build\Scripts\python.exe pack_int8b.py
"""
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch.set_num_threads(4)

from transformers import BertConfig, BertModel                       # noqa: E402
from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASREncoder  # noqa: E402
from transformers.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASREncoderConfig  # noqa: E402
from torch.ao.quantization import (quantize_dynamic, default_dynamic_qconfig)  # noqa: E402

P = HERE / "model" / "intent_int8.pt"
QLIN = {torch.nn.Linear: default_dynamic_qconfig}
QCONV = {torch.nn.Linear: default_dynamic_qconfig, torch.nn.Conv1d: default_dynamic_qconfig}

print(f"载入 {P.name} ...")
ck = torch.load(str(P), map_location="cpu", weights_only=False)
print(f"  旧格式键: {sorted(ck.keys())}")

t0 = time.perf_counter()
allowed = set(Qwen3ASREncoderConfig().to_dict().keys())
acfg = {k: v for k, v in ck["asr_audio_config"].items() if k in allowed}
A = Qwen3ASREncoder(Qwen3ASREncoderConfig(**acfg))
A = quantize_dynamic(A, QCONV, dtype=torch.qint8)
A.load_state_dict(ck["A_q"])

bert = BertModel(BertConfig(**ck["bert_config"]), add_pooling_layer=False)
emb = dict(ck["bert_emb"])
emb["word_embeddings.weight"] = ck["word_emb"]
bert.embeddings.load_state_dict(emb, strict=False)
bert.encoder = quantize_dynamic(bert.encoder, QLIN, dtype=torch.qint8)
bert.encoder.load_state_dict(ck["Benc_q"])
print(f"  重建量化模块 {time.perf_counter()-t0:.1f}s")

out = {k: v for k, v in ck.items() if not k.endswith("_q")}
out["A_module"] = A
out["Benc_module"] = bert.encoder
out["fmt"] = "int8-module-v2"
P2 = HERE / "model" / "intent_int8b.pt"
torch.save(out, P2)
mb = P2.stat().st_size / 1e6
print(f"  → {P2.name}  {mb:.0f} MB   (旧 {P.stat().st_size/1e6:.0f} MB)")
print("  加载时只需 torch.load —— 全程不出现 fp32 的 A / B encoder")
