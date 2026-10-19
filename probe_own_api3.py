# -*- coding: utf-8 -*-
r"""钉死 4 个没映射上的张量（音频→文本投影层）"""
import json
import os

import torch
from safetensors import safe_open

D = r"E:\models\qwen3-asr-0.6b-official"
from transformers.models.qwen3_asr.configuration_qwen3_asr import (  # noqa: E402
    Qwen3ASRConfig, Qwen3ASREncoderConfig)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config  # noqa: E402
from transformers.models.qwen3_asr.modeling_qwen3_asr import (  # noqa: E402
    Qwen3ASRForConditionalGeneration)

raw = json.loads(open(os.path.join(D, "config.json"), encoding="utf-8").read())
tc = raw["thinker_config"]
aok = set(Qwen3ASREncoderConfig().to_dict().keys())
acfg = {k: v for k, v in tc["audio_config"].items() if k in aok}
acfg["model_type"] = "qwen3_asr_encoder"
tok = set(Qwen3Config().to_dict().keys())
tcfg = {k: v for k, v in tc["text_config"].items() if k in tok}
cfg = Qwen3ASRConfig(audio_config=acfg, text_config=tcfg,
                     audio_token_id=tc.get("audio_token_id"),
                     timestamp_token_id=tc.get("timestamp_token_id", 151705))
model = Qwen3ASRForConditionalGeneration(cfg)

mapped = {}
with safe_open(os.path.join(D, "model.safetensors"), framework="pt") as f:
    keys = list(f.keys())
    for k in keys:
        if k.startswith("thinker.audio_tower."):
            mapped["model.audio_tower." + k[len("thinker.audio_tower."):]] = f.get_tensor(k)
        elif k.startswith("thinker.model."):
            mapped["model.language_model." + k[len("thinker.model."):]] = f.get_tensor(k)
        elif k == "thinker.lm_head.weight":
            mapped["lm_head.weight"] = f.get_tensor(k)

missing, unexpected = model.load_state_dict(mapped, strict=False)
print(f"missing {len(missing)}:")
for m in missing:
    try:
        print(f"   {m}   {tuple(model.state_dict()[m].shape)}")
    except Exception:
        print(f"   {m}   ?")
print(f"\nunexpected {len(unexpected)}:")
for u in unexpected:
    print(f"   {u}   {tuple(mapped[u].shape)}")

print("\n模型里带 proj / multi_modal 的参数:")
for n, p in model.named_parameters():
    if "proj" in n or "multi_modal" in n:
        print(f"   {n:60s} {tuple(p.shape)}")

print("\ncheckpoint 里 audio_tower 下的 proj / conv_out 键:")
with safe_open(os.path.join(D, "model.safetensors"), framework="pt") as f:
    for k in f.keys():
        if "proj" in k or "conv_out" in k:
            print(f"   {k:60s} {tuple(f.get_slice(k).get_shape())}")
