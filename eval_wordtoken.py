# -*- coding: utf-8 -*-
"""用【大小写无关】重算整词 token 的效果

整词 token 输出的词是小写形式（<w:blender> → "blender"），
但参考文本是 "Blender"。之前的 exact 比较是大小写敏感的 → 误报 0%。
"""
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

A_DIR = Path(r"E:\models\qwen3-asr-0.6b-official")
B_DIR = Path(r"E:\models\dmeta-embedding-zh")
WAV_ORIG = HERE / "wavs"
WAV_EN = HERE / "wavs_en"

from safetensors import safe_open
from transformers import AutoTokenizer, BertConfig, BertModel
from transformers.models.qwen3_asr.feature_extraction_qwen3_asr import Qwen3ASRFeatureExtractor
from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASREncoder
from transformers.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASREncoderConfig


def byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BD = byte_decoder()
av = json.loads((A_DIR / "vocab.json").read_text(encoding="utf-8"))
a_single = set()
for k in av:
    try:
        s = bytes(BD[c] for c in k).decode("utf-8")
    except Exception:
        continue
    if len(s) == 1:
        a_single.add(s)
blines = (B_DIR / "vocab.txt").read_text(encoding="utf-8").splitlines()
b2i = {}
for i, t in enumerate(blines):
    t = t.replace("##", "")
    if len(t) == 1:
        b2i.setdefault(t, i)
SHARED_CH = sorted(set(a_single) & set(b2i))

for tag, md in [("整词 token (model_wt)", HERE / "model_wt"),
                ("旧版 (model)", HERE / "model")]:
    CK = torch.load(str(md / "intent.pt"), map_location="cpu", weights_only=False)
    SHARED = CK["shared_chars"]
    K = len(SHARED)
    E_ALL = torch.load(str(B_DIR / "pytorch_model.bin"), map_location="cpu",
                       weights_only=True)["embeddings.word_embeddings.weight"].float()
    tok = AutoTokenizer.from_pretrained(str(B_DIR))

    # E_pad：整词行用均值
    rows = [torch.zeros(1, 768)] + [E_ALL[b2i[c]].unsqueeze(0) for c in SHARED_CH]
    ENW = CK.get("en_words", [])
    if ENW:
        with torch.no_grad():
            for w in ENW:
                ids = tok(w, add_special_tokens=False)["input_ids"]
                rows.append(E_ALL[ids].mean(0, keepdim=True))
    E_PAD = torch.cat(rows, 0)

    ctc = torch.nn.Linear(896, K + 1)
    ctc.weight.data = CK["ctc_head"]; ctc.bias.data = CK["ctc_bias"]
    fe = Qwen3ASRFeatureExtractor.from_pretrained(str(A_DIR))
    raw = json.loads((A_DIR / "config.json").read_text(encoding="utf-8"))
    acfg = dict(raw["thinker_config"]["audio_config"])
    allowed = set(Qwen3ASREncoderConfig().to_dict().keys())
    A = Qwen3ASREncoder(Qwen3ASREncoderConfig(**{k: v for k, v in acfg.items() if k in allowed}))
    sdA = {}
    with safe_open(str(A_DIR / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("thinker.audio_tower."):
                sdA[k[len("thinker.audio_tower."):]] = f.get_tensor(k)
    A.load_state_dict(sdA, strict=False); A.eval()
    for p in A.parameters():
        p.requires_grad = False

    def decode(logits):
        idx = logits.argmax(-1).tolist()
        out, prev = [], -1
        for i in idx:
            if i != prev and i != 0:
                t = SHARED[i - 1]
                out.append(t[3:-1] if t.startswith("<w:") else t)
            prev = i
        return "".join(out)

    # ══════════ 英文专项 val ══════════
    en = [json.loads(l) for l in io.open(HERE / "en_dataset.jsonl", encoding="utf-8") if l.strip()]
    en_val = [j for j in en if j["split"] == "val" and (WAV_EN / f"{j['id']}.wav").exists()]
    n_exact = 0; n_ci = 0; bad = []
    with torch.no_grad():
        for j in en_val:
            wav, sr = sf.read(str(WAV_EN / f"{j['id']}.wav"), dtype="float32")
            f = fe(wav, sampling_rate=16000, return_tensors="pt", padding=False,
                   return_attention_mask=True)
            inp, msk = f["input_features"], f["attention_mask"].long()
            pad = (-inp.shape[-1]) % 100
            if pad:
                inp = F.pad(inp, (0, pad)); msk = F.pad(msk, (0, pad))
            hyp = decode(ctc(A(inp, msk).last_hidden_state))
            n_exact += int(hyp == j["text"])
            ci = int(hyp.lower() == j["text"].lower())
            n_ci += ci
            if not ci and len(bad) < 10:
                bad.append((j["text"], hyp))
    N = len(en_val)
    print("=" * 84)
    print(f"  {tag}")
    print("=" * 84)
    print(f"  英文专项 val ({N} 条)")
    print(f"    大小写敏感    {n_exact/N*100:6.2f}%")
    print(f"    ⭐ 大小写无关  {n_ci/N*100:6.2f}%")
    if bad:
        print(f"    错例:")
        for r, h in bad:
            print(f"      说 {r:26s} 转 {h}")
    print()
