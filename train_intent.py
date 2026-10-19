# -*- coding: utf-8 -*-
"""训练意图模型（V5 集成架构）

    wav → A(audio_tower 18层, 冻结) → ctc_head(trainable) → logits
        ├─ V1: p=softmax(logits/τ) → z=p@E_pad → 压缩r=3 → [CLS,z] → B(冻结) → CLS1
        └─ V3: CTC解码 → 字符串 → dmeta分词 → 精确词嵌入行 → B(冻结) → CLS2
        → concat(CLS1, CLS2) 1536维 → intent_head → 8 类

V1 提供软容错，V3 提供与文本路径的对齐（cos 0.95）。
V3 走硬解码不可导，所以它作为【固定特征】参与；梯度由 CTC loss + V1 的意图 loss 提供。

实测（上一版 42 类）：V1 97.87% / V3 95.51% / V5 集成 98.11%
"""
import io
import json
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
WAV = ROOT / "wavs"
JOBS = ROOT / "intent_dataset.jsonl"
SAVE = ROOT / "model"
SAVE.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "train_report.json"

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
R = 3; SEED = 0; TAU = 0.1
EPOCHS = 220; LR = 2e-3; BS = 32
# ⚠️ CTC loss 权重：必须 >1。
#    实测 λ=1 时，一旦加入难负样本，模型会牺牲 CTC 去换意图精度
#    → 转写退化成几个字（"帮我启动Steam" → "打开打"），
#      意图还看着不差（95.88%）但 V3 路径废了、增量加意图也没法用了。
#    λ=2 把 CTC 钉住。
LAMBDA_CTC = 2.0

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


print("=" * 78); print("① 词表交集"); print("=" * 78)
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
print(f"  K={K}  CLS_ID={CLS_ID}")

jobs = [json.loads(l) for l in io.open(JOBS, encoding="utf-8") if l.strip()]
INTENTS = sorted({j["intent"] for j in jobs})
i2c = {o: i for i, o in enumerate(INTENTS)}
C = len(INTENTS)
y_all = np.array([i2c[j["intent"]] for j in jobs])
tr = np.array([i for i, j in enumerate(jobs) if j["split"] == "train"])
te = np.array([i for i, j in enumerate(jobs) if j["split"] == "val"])
print(f"  {len(jobs)} 条  {C} 类  train {len(tr)}  test {len(te)}")
print(f"  intents: {INTENTS}")

print(); print("=" * 78); print("② 抽 A 帧特征"); print("=" * 78)
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
FR = []
with torch.no_grad():
    for n_, job in enumerate(jobs):
        wav, _ = sf.read(str(WAV / f"{job['id']}.wav"), dtype="float32")
        f = fe(wav, sampling_rate=16000, return_tensors="pt", padding=False,
               return_attention_mask=True)
        FR.append(A(f["input_features"].to(DEV), f["attention_mask"].to(DEV).long())
                  .last_hidden_state.float())
        if (n_ + 1) % 400 == 0:
            print(f"    {n_+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
del A; torch.cuda.empty_cache()
print(f"  抽完 {time.time()-t0:.0f}s")

print(); print("=" * 78); print("③ B（冻结）+ 分词器"); print("=" * 78)
_cfg = BertConfig.from_json_file(str(B_DIR / "config.json"))
bert = BertModel(_cfg, add_pooling_layer=False).to(DEV).eval()
m = bert.load_state_dict(Bfull, strict=False)
print(f"  missing={len(m.missing_keys)} unexpected={len(m.unexpected_keys)}")
for p in bert.parameters():
    p.requires_grad = False
tok = AutoTokenizer.from_pretrained(str(B_DIR))


def run_b(seq, msk):
    return bert(inputs_embeds=seq, attention_mask=msk, return_dict=True).last_hidden_state[:, 0]


def compress(z, mask, r=R):
    """逐条按有效长度池化（与部署一致；不传 mask 会把 padding 零帧平均进去）"""
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
        nn.init.zeros_(self.ctc.bias)
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
    """V3：CTC 解码 → dmeta 分词 → 精确词嵌入行 → B → CLS（不可导，作为固定特征）

    ⚠️ 必须批处理：逐样本过 B 会让每个 epoch 慢 30 倍（实测 14s/epoch → 0.5s/epoch）
    """
    all_ids = [tok(ctc_decode(lg), truncation=True, max_length=64)["input_ids"]
               for lg in logits_list]
    L = max(len(i) for i in all_ids)
    rows = torch.zeros(len(all_ids), L, 768, device=DEV)
    msk = torch.zeros(len(all_ids), L, device=DEV)
    for j, ids in enumerate(all_ids):
        rows[j, :len(ids)] = E_ALL[ids].to(DEV)
        msk[j, :len(ids)] = 1
    return run_b(rows, msk)


print(); print("=" * 78); print("④ 训练"); print("=" * 78)
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
net = Net().to(DEV)
head = nn.Linear(1536, C).to(DEV)
ps = list(net.parameters()) + list(head.parameters())
print(f"  可训练 {sum(p.numel() for p in ps)/1e6:.2f} M  epochs {EPOCHS}  bs {BS}")
opt = torch.optim.AdamW(ps, lr=LR, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS)
tgt = [torch.tensor([char2i[c] for c in jobs[i]["text"] if c in char2i], dtype=torch.long)
       for i in range(len(jobs))]


def batchify(b):
    hs = [FR[i] for i in b]
    T = max(x.shape[0] for x in hs)
    H = torch.zeros(len(b), T, 896, device=DEV); HM = torch.zeros(len(b), T, device=DEV)
    for j, x in enumerate(hs):
        H[j, :x.shape[0]] = x; HM[j, :x.shape[0]] = 1
    return H, HM


for ep in range(EPOCHS):
    perm = rng.permutation(tr); tot_l = nb = 0
    for s in range(0, len(perm), BS):
        b = perm[s:s+BS]
        H, HM = batchify(b)
        if ep > 10:
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
        za = torch.cat([c1, c2], 1)
        loss = F.cross_entropy(head(za), torch.tensor(y_all[b], device=DEV))
        lp = F.log_softmax(logits, -1).transpose(0, 1)
        tl = torch.tensor([len(tgt[i]) for i in b], device=DEV)
        keep = tl > 0
        if keep.any():
            loss = loss + LAMBDA_CTC * F.ctc_loss(
                lp[:, keep],
                torch.cat([tgt[i] for i in b if len(tgt[i]) > 0]),
                HM[keep].sum(1).long(), tl[keep],
                blank=0, zero_infinity=True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ps, 1.0); opt.step()
        tot_l += loss.item(); nb += 1
    sch.step()
    if ep % 20 == 0 or ep == EPOCHS - 1:
        print(f"  ep {ep:3d}  loss {tot_l/max(nb,1):.4f}  {time.time()-t0:.0f}s", flush=True)

print(); print("=" * 78); print("⑤ 评估"); print("=" * 78)


def evaluate(split):
    preds = []; texts = []
    with torch.no_grad():
        for s in range(0, len(split), 64):
            b = split[s:s+64]
            H, HM = batchify(b)
            z, logits = net(H)
            zc, zcm = compress(z, HM)
            seq = torch.cat([CLS_ROW.view(1, 1, -1).expand(len(b), -1, -1), zc], 1)
            msk = torch.cat([torch.ones(len(b), 1, device=DEV), zcm], 1)
            c1 = run_b(seq, msk)
            c2 = cls_v3([logits[j, :int(HM[j].sum())] for j in range(len(b))])
            p = head(torch.cat([c1, c2], 1)).argmax(1).cpu().numpy()
            preds.extend(p.tolist())
            texts.extend(ctc_decode(logits[j, :int(HM[j].sum())]) for j in range(len(b)))
    return np.array(preds), texts


pr_tr, tx_tr = evaluate(tr)
pr_te, tx_te = evaluate(te)
a_tr = (pr_tr == y_all[tr]).mean(); a_te = (pr_te == y_all[te]).mean()
print(f"  总体意图精度   train {a_tr*100:6.2f}%   test {a_te*100:6.2f}%")

print()
print("  按意图拆解（test）:")
per = defaultdict(lambda: [0, 0])
for p, y in zip(pr_te, y_all[te]):
    per[INTENTS[y]][0] += 1
    per[INTENTS[y]][1] += int(p == y)
for it in sorted(per):
    n_, ok = per[it]
    print(f"    {it:14s} {ok:>3}/{n_:<4} = {ok/n_*100:6.2f}%")

exact = sum(1 for i, k in enumerate(te) if tx_te[i] == jobs[k]["text"])
print(f"\n  CTC 转写完全正确 {exact}/{len(te)} = {exact/len(te)*100:.2f}%")

cm = Counter()
for p, y in zip(pr_te, y_all[te]):
    if p != y:
        cm[(INTENTS[y], INTENTS[p])] += 1
print(f"\n  最常混淆（top 10）:")
for (t, p), c in cm.most_common(10):
    print(f"    {t:14s} → {p:14s} {c} 次")

torch.save({"ctc_head": net.ctc.weight.data.cpu(), "ctc_bias": net.ctc.bias.data.cpu(),
            "head_w": head.weight.data.cpu(), "head_b": head.bias.data.cpu(),
            "shared_chars": SHARED, "intents": INTENTS, "tau": TAU,
            "cls_id": CLS_ID, "R": R}, str(SAVE / "intent.pt"))
io.open(SAVE / "meta.json", "w", encoding="utf-8").write(json.dumps(
    {"intents": INTENTS, "K": K, "tau": TAU, "R": R, "cls_id": CLS_ID,
     "acc_test": float(a_te), "acc_train": float(a_tr),
     "n_train": len(tr), "n_test": len(te), "arch": "v5_ensemble"}, ensure_ascii=False, indent=1))
io.open(OUT, "w", encoding="utf-8").write(json.dumps(
    {"acc_train": float(a_tr), "acc_test": float(a_te),
     "per_intent": {k: (v[1] / v[0]) for k, v in per.items()},
     "ctc_exact": exact / len(te), "intents": INTENTS},
    ensure_ascii=False, indent=1))
print(f"\n-> {SAVE/'intent.pt'}   {OUT}")
