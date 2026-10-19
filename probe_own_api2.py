# -*- coding: utf-8 -*-
r"""量清 Audio features 与 audio tokens 到底哪里不等"""
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

D = r"E:\models\qwen3-asr-0.6b-official"
WAV = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\wavs\train_launch_app_0182322b5f.wav")

with wave.open(str(WAV), "rb") as w:
    raw = w.readframes(w.getnframes())
wav = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
pad = (-len(wav)) % 16000
if pad:
    wav = np.concatenate([wav, np.zeros(pad, np.float32)])

proc = AutoProcessor.from_pretrained(D)
msgs = [{"role": "user", "content": [{"type": "audio", "audio": wav}]}]
inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_dict=True, return_tensors="pt")
print("处理器输出:")
for k, v in inputs.items():
    print(f"   {k:22s} shape={tuple(v.shape)} dtype={v.dtype}")
print("   audio_token_id =", proc.audio_token_id,
      " 出现次数 =", int((inputs["input_ids"] == proc.audio_token_id).sum()))
print("   input_features_mask.sum() =", int(inputs["input_features_mask"].sum()))

ModelCls = None
for path in ("transformers", "transformers.models.qwen3_asr.modeling_qwen3_asr"):
    try:
        import importlib
        ModelCls = getattr(importlib.import_module(path), "Qwen3ASRForConditionalGeneration")
        break
    except Exception:
        pass
print("\n载入模型（注意看有没有 missing/unexpected keys 警告）...")
model = ModelCls.from_pretrained(D, dtype=torch.float32)
model.eval()
cfg = model.config
print("   config.thinker_config.text_config.hidden_size =",
      cfg.thinker_config.text_config.hidden_size)
print("   config.thinker_config.audio_config.output_dim  =",
      cfg.thinker_config.audio_config.output_dim)
print("   embed_tokens           :", tuple(model.get_input_embeddings().weight.shape))
proj = getattr(model, "multi_modal_projector", None)
print("   multi_modal_projector  :", type(proj).__name__ if proj else None)
if proj is not None:
    for name, mod in proj.named_modules():
        if isinstance(mod, torch.nn.Linear):
            print(f"      {name or '<root>'}: in={mod.in_features} out={mod.out_features}")

with torch.no_grad():
    af = model.get_audio_features(inputs["input_features"],
                                  inputs["input_features_mask"],
                                  return_dict=True).pooler_output
print("\n   audio_features(pooler_output) shape =", tuple(af.shape),
      "numel =", af.numel())
mask = (inputs["input_ids"] == cfg.thinker_config.audio_token_id)
emb = model.get_input_embeddings()(inputs["input_ids"])
m2 = mask.unsqueeze(-1).expand_as(emb)
print("   inputs_embeds shape =", tuple(emb.shape), " mask true =", int(mask.sum()))
print("   inputs_embeds[mask].numel() =", emb[m2].numel())
print("   → 相等?", emb[m2].numel() == af.numel(), "  ", emb[m2].numel(), "vs", af.numel())
