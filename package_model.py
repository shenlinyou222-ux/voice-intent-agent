# -*- coding: utf-8 -*-
"""把推理需要的权重打成一个 bundle.safetensors

包含：
  audio_tower.*           186.38 M   Qwen3-ASR 的音频编码器（18 层）
  text_encoder.encoder.*   85.05 M   dmeta 的 12 层编码器
  bert_emb.*                0.79 M   B 的 position / token_type / LayerNorm
  bert_word_emb            16.23 M   B 的词嵌入表（V3 路径要按 token id 查）
  vocab_txt / bert_config_json      分词器和配置（文本）
"""
import json
from pathlib import Path

import torch

A_DIR = Path(r"E:\models\qwen3-asr-0.6b-official")
B_DIR = Path(r"E:\models\dmeta-embedding-zh")
ROOT = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
OUT = ROOT / "model" / "bundle.safetensors"

from safetensors import safe_open
from safetensors.torch import save_file

OUT.parent.mkdir(parents=True, exist_ok=True)
print("=" * 78)
print("  打包模型 bundle")
print("=" * 78)

out = {}
with safe_open(str(A_DIR / "model.safetensors"), framework="pt") as f:
    for k in f.keys():
        if k.startswith("thinker.audio_tower."):
            out["audio_tower." + k[len("thinker.audio_tower."):]] = f.get_tensor(k).clone()

Bfull = torch.load(str(B_DIR / "pytorch_model.bin"), map_location="cpu", weights_only=True)
for k, v in Bfull.items():
    if k.startswith("encoder."):
        out["text_encoder." + k] = v.clone()
for k in ("embeddings.position_embeddings.weight",
          "embeddings.token_type_embeddings.weight",
          "embeddings.LayerNorm.weight",
          "embeddings.LayerNorm.bias"):
    out["bert_emb." + k[len("embeddings."):]] = Bfull[k].clone()
out["bert_word_emb"] = Bfull["embeddings.word_embeddings.weight"].clone()

# 文本用 tensor 存（safetensors 只收 tensor）
vocab_txt = (B_DIR / "vocab.txt").read_bytes()
out["vocab_txt"] = torch.tensor(list(vocab_txt), dtype=torch.uint8)
cfg_json = (B_DIR / "config.json").read_bytes()
out["bert_config_json"] = torch.tensor(list(cfg_json), dtype=torch.uint8)

N = sum(t.numel() for t in out.values())
print(f"  张量 {len(out)} 个   参数 {N/1e6:.2f} M")
for g in ("audio_tower", "text_encoder", "bert_emb", "bert_word_emb"):
    n = sum(t.numel() for k, t in out.items() if k.startswith(g))
    print(f"    {g:16s} {n/1e6:8.2f} M")
print(f"    vocab/config     {len(vocab_txt)+len(cfg_json)} 字节")

save_file(out, str(OUT))
print(f"\n  -> {OUT}   {OUT.stat().st_size/1e6:.1f} MB")
