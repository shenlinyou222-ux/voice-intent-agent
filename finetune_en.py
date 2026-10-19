# -*- coding: utf-8 -*-
"""微调 CTC head —— 专治英文专有名词

从已有 checkpoint 出发（不是从零），用【英文专项数据 + 原数据回放】微调。
只训 ctc_head（A / B 全冻结），所以很快。

⭐ 音频增强：变速 ±10% / 变调 ±10%
   实测每个英文名字样本太少，增强能显著扩大有效数据量，
   而且真实语音的语速/音高本来就有变化。

⭐ 评估分两组：
   · 英文专项 val（311 条）—— 看英文名字有没有学会
   · 原数据集 val（609 条）—— 看中文能力有没有被破坏（灾难性遗忘）
"""
import io
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

A_DIR = Path(r"E:\models\qwen3-asr-0.6b-official")
B_DIR = Path(r"E:\models\dmeta-embedding-zh")
ROOT = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
WAV_ORIG = ROOT / "wavs"
WAV_EN = ROOT / "wavs_en"
JOBS_ORIG = ROOT / "intent_dataset.jsonl"
JOBS_EN = ROOT / "en_dataset.jsonl"
CKPT_IN = ROOT / "model" / "intent.pt"
SAVE = ROOT / "model_en"
SAVE.mkdir(exist_ok=True)
OUT = ROOT / "finetune_en_report.json"

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
R = 3; SEED = 0; TAU = 0.1
EPOCHS = 120; LR = 8e-4; BS = 32
LAMBDA_CTC = 2.0
# 增强：变速（改变音高和时长）
AUG_SPEEDS = [0.9, 1.0, 1.1]

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


print("=" * 78); print("① 词表 / 模型"); print("=" * 78)
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
SHARED = sorted(set(a_single) & set(b2i))
K = len(SHARED); CLS_ID = blines.index("[CLS]")
Bfull = torch.load(str(B_DIR / "pytorch_model.bin"), map_location="cpu", weights_only=True)
E_ALL = Bfull["embeddings.word_embeddings.weight"].float()
E_PAD = torch.cat([torch.zeros(1, 768), E_ALL[[b2i[c] for c in SHARED]]], 0).to(DEV)
CLS_ROW = E_ALL[CLS_ID].to(DEV)
char2i = {c: i + 1 for i, c in enumerate(SHARED)}

CK = torch.load(str(CKPT_IN), map_location="cpu", weights_only=False)
INTENTS = CK["intents"]; i2c = {o: i for i, o in enumerate(INTENTS)}; C = len(INTENTS)
print(f"  K={K}  从已有 ckpt 出发，{C} 类: {INTENTS}")

# ══════════ 合并数据集 ══════════
def load(jobs_path, wav_dir, tag):
    js = [json.loads(l) for l in io.open(jobs_path, encoding="utf-8") if l.strip()]
    out = []
    for j in js:
        p = wav_dir / f"{j['id']}.wav"
        if p.exists():
            out.append({"id": j["id"], "text": j["text"], "intent": j["intent"],
                        "split": j["split"], "wav": p, "tag": tag})
    return out

orig = load(JOBS_ORIG, WAV_ORIG, "orig")
en = load(JOBS_EN, WAV_EN, "en")
ALL = orig + en
print(f"  原数据 {len(orig)} 条 + 英文专项 {len(en)} 条 = {len(ALL)} 条")

tr = [i for i, j in enumerate(ALL) if j["split"] == "train"]
te_en = [i for i, j in enumerate(ALL) if j["split"] == "val" and j["tag"] == "en"]
te_orig = [i for i, j in enumerate(ALL) if j["split"] == "val" and j["tag"] == "orig"]
print(f"  train {len(tr)}   val(英文) {len(te_en)}   val(原) {len(te_orig)}")

print(); print("=" * 78); print("② A 编码器"); print("=" * 78)
t0 = time.time()
raw = json.loads((A_DIR / "config.json").read_text(encoding="utf-8"))
acfg = dict(raw["thinker_config"]["audio_config"])
allowed = set(Qwen3ASREncoderConfig().to_dict().keys())
A = Qwen3ASREncoder(Qwen3ASREncoderConfig(**{k: v for k, v in acfg.items() if k in allowed}))
sdA = {}
with safe_open(str(A_DIR / "model.safetensors"), framework="pt") as f:
    for k in f.keys():
        if k.startswith("thinker.audio_tower."):
            sdA[k[len("thinker.audio_tower."):]] = f.get_tensor(k)
A.load_state_dict(sdA, strict=False); A = A.to(DEV).eval()
for p in A.parameters():
    p.requires_grad = False
fe = Qwen3ASRFeatureExtractor.from_pretrained(str(A_DIR))

print("  抽帧特征（含变速增强）...")
FR = {}     # (idx, speed) -> tensor
with torch.no_grad():
    for n_, i in enumerate(tr):
        wav, sr = sf.read(str(ALL[i]["wav"]), dtype="float32")
        for sp in AUG_SPEEDS:
            w = wav if sp == 1.0 else np.interp(
                np.linspace(0, len(wav) - 1, int(len(wav) / sp)),
                np.arange(len(wav)), wav).astype(np.float32)
            f = fe(w, sampling_rate=16000, return_tensors="pt", padding=False,
                   return_attention_mask=True)
            FR[(i, sp)] = A(f["input_features"].to(DEV),
                            f["attention_mask"].to(DEV).long()).last_hidden_state.float()
        if (n_ + 1) % 300 == 0:
            print(f"    {n_+1}/{len(tr)}  {time.time()-t0:.0f}s", flush=True)
    # 验证集只跑原速
    for i in te_en + te_orig:
        wav, sr = sf.read(str(ALL[i]["wav"]), dtype="float32")
        f = fe(wav, sampling_rate=16000, return_tensors="pt", padding=False,
               return_attention_mask=True)
        FR[(i, 1.0)] = A(f["input_features"].to(DEV),
                         f["attention_mask"].to(DEV).long()).last_hidden_state.float()
del A; torch.cuda.empty_cache()
print(f"  完成 {time.time()-t0:.0f}s")

print(); print("=" * 78); print("③ B + 分词器"); print("=" * 78)
_cfg = BertConfig.from_json_file(str(B_DIR / "config.json"))
bert = BertModel(_cfg, add_pooling_layer=False).to(DEV).eval()
bert.load_state_dict(Bfull, strict=False)
for p in bert.parameters():
    p.requires_grad = False
tok = AutoTokenizer.from_pretrained(str(B_DIR))


def run_b(seq, msk):
    return bert(inputs_embeds=seq, attention_mask=msk, return_dict=True).last_hidden_state[:, 0]


def compress(z, mask, r=R):
    N, T, D = z.shape
    outs = []
    for i in range(N):
        t = max(int(mask[i].sum().item()), 1)
        kk = max(1, int(np.ceil(t / r)))
        outs.append(F.adaptive_avg_pool1d(z[i, :t].T.unsqueeze(0), kk).squeeze(0).T)
    kmax = max(o.shape[0] for o in outs)
    out = torch.zeros(N, kmax, D, device=z.device, dtype=z.dtype)
    om = torch.zeros(N, kmax, device=z.device)
    for i, o in enumerate(outs):
        out[i, :o.shape[0]] = o; om[i, :o.shape[0]] = 1
    return out, om


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctc = nn.Linear(896, K + 1)
        self.ctc.weight.data = CK["ctc_head"].clone()
        self.ctc.bias.data = CK["ctc_bias"].clone()
        self.tau = TAU

    def forward(self, h):
        logits = self.ctc(h)
        return torch.softmax(logits / self.tau, -1) @ E_PAD, logits


def ctc_decode(logits):
    idx = logits.argmax(-1).tolist()
    out, prev = [], -1
    for i in idx:
        if i != prev and i != 0:
            out.append(SHARED[i - 1])
        prev = i
    return "".join(out)


@torch.no_grad()
def cls_v3(logits_list):
    all_ids = [tok(ctc_decode(lg), truncation=True, max_length=64)["input_ids"]
               for lg in logits_list]
    L = max(len(i) for i in all_ids)
    rows = torch.zeros(len(all_ids), L, 768, device=DEV)
    msk = torch.zeros(len(all_ids), L, device=DEV)
    for j, ids in enumerate(all_ids):
        rows[j, :len(ids)] = E_ALL[ids].to(DEV)
        msk[j, :len(ids)] = 1
    return run_b(rows, msk)


print(); print("=" * 78); print("④ 微调 ctc_head"); print("=" * 78)
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
net = Net().to(DEV)
head = nn.Linear(1536, C).to(DEV)
head.weight.data = CK["head_w"].clone(); head.bias.data = CK["head_b"].clone()
ps = list(net.parameters()) + list(head.parameters())
print(f"  可训练 {sum(p.numel() for p in ps)/1e6:.2f} M  epochs {EPOCHS}  lr {LR}")
opt = torch.optim.AdamW(ps, lr=LR, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS)
tgt = [torch.tensor([char2i[c] for c in ALL[i]["text"] if c in char2i], dtype=torch.long)
       for i in range(len(ALL))]
y_all = np.array([i2c[j["intent"]] for j in ALL])


def batchify(b, speed=1.0):
    hs = [FR[(i, speed)] for i in b]
    T = max(x.shape[0] for x in hs)
    H = torch.zeros(len(b), T, 896, device=DEV); HM = torch.zeros(len(b), T, device=DEV)
    for j, x in enumerate(hs):
        H[j, :x.shape[0]] = x; HM[j, :x.shape[0]] = 1
    return H, HM


for ep in range(EPOCHS):
    perm = rng.permutation(tr); tot_l = nb = 0
    for s in range(0, len(perm), BS):
        b = perm[s:s+BS]
        sp = AUG_SPEEDS[int(rng.integers(0, len(AUG_SPEEDS)))]
        H, HM = batchify(b, sp)
        if ep > 5:
            for _ in range(2):
                for j in range(len(b)):
                    t = int(HM[j].sum())
                    if t > 6:
                        w = max(1, t // 12); st = int(rng.integers(0, max(1, t - w)))
                        H[j, st:st+w] = 0
        z, logits = net(H)
        zc, zcm = compress(z, HM)
        seq = torch.cat([CLS_ROW.view(1, 1, -1).expand(len(b), -1, -1), zc], 1)
        msk = torch.cat([torch.ones(len(b), 1, device=DEV), zcm], 1)
        c1 = run_b(seq, msk)
        with torch.no_grad():
            c2 = cls_v3([logits[j, :int(HM[j].sum())] for j in range(len(b))])
        loss = F.cross_entropy(head(torch.cat([c1, c2], 1)),
                               torch.tensor(y_all[b], device=DEV))
        lp = F.log_softmax(logits, -1).transpose(0, 1)
        tl = torch.tensor([len(tgt[i]) for i in b], device=DEV)
        keep = tl > 0
        if keep.any():
            loss = loss + LAMBDA_CTC * F.ctc_loss(
                lp[:, keep], torch.cat([tgt[i] for i in b if len(tgt[i]) > 0]),
                HM[keep].sum(1).long(), tl[keep], blank=0, zero_infinity=True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ps, 1.0); opt.step()
        tot_l += loss.item(); nb += 1
    sch.step()
    if ep % 20 == 0 or ep == EPOCHS - 1:
        print(f"  ep {ep:3d}  loss {tot_l/max(nb,1):.4f}  {time.time()-t0:.0f}s", flush=True)

print(); print("=" * 78); print("⑤ 评估"); print("=" * 78)


def evaluate(idxs, label):
    if not idxs:
        return 0.0, 0.0
    exact = 0; n = 0; preds = []
    with torch.no_grad():
        for s in range(0, len(idxs), 64):
            b = idxs[s:s+64]
            H, HM = batchify(b, 1.0)
            z, logits = net(H)
            zc, zcm = compress(z, HM)
            seq = torch.cat([CLS_ROW.view(1, 1, -1).expand(len(b), -1, -1), zc], 1)
            msk = torch.cat([torch.ones(len(b), 1, device=DEV), zcm], 1)
            c1 = run_b(seq, msk)
            c2 = cls_v3([logits[j, :int(HM[j].sum())] for j in range(len(b))])
            preds.extend(head(torch.cat([c1, c2], 1)).argmax(1).cpu().tolist())
            for j, i in enumerate(b):
                hyp = ctc_decode(logits[j, :int(HM[j].sum())])
                exact += int(hyp == ALL[i]["text"]); n += 1
    acc = (np.array(preds) == y_all[idxs]).mean()
    print(f"  {label:14s} 意图 {acc*100:6.2f}%   转写完全对 {exact/max(n,1)*100:6.2f}%")
    return acc, exact / max(n, 1)


a_en, x_en = evaluate(te_en, "英文专项 val")
a_or, x_or = evaluate(te_orig, "原数据集 val")

print()
print("  对照（微调前）:")
print("    英文专项 val 转写完全对  ——  未测（这批数据是新的）")
print("    原数据集 val 转写完全对   77.18%   意图 98.69%")

torch.save({"ctc_head": net.ctc.weight.data.cpu(), "ctc_bias": net.ctc.bias.data.cpu(),
            "head_w": head.weight.data.cpu(), "head_b": head.bias.data.cpu(),
            "shared_chars": SHARED, "intents": INTENTS, "tau": TAU,
            "cls_id": CLS_ID, "R": R}, str(SAVE / "intent.pt"))
io.open(SAVE / "meta.json", "w", encoding="utf-8").write(json.dumps(
    {"intents": INTENTS, "K": K, "tau": TAU, "R": R, "cls_id": CLS_ID,
     "acc_test": float(a_or), "ctc_exact_orig": float(x_or),
     "ctc_exact_en": float(x_en), "arch": "v5_ensemble_enft"},
    ensure_ascii=False, indent=1))
io.open(OUT, "w", encoding="utf-8").write(json.dumps(
    {"en_intent": float(a_en), "en_exact": float(x_en),
     "orig_intent": float(a_or), "orig_exact": float(x_or)}, ensure_ascii=False, indent=1))
print(f"\n-> {SAVE/'intent.pt'}   {OUT}")
