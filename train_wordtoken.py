# -*- coding: utf-8 -*-
"""⭐ 用【整词 token】重训 CTC head —— 从根上修英文丢首字母

输出表：6628 字符 + N 个英文整词 token（<w:blender> 这种）
    Blender → <w:blender>   1 个尖峰（原来 7 个）
    PowerShell → <w:powershell>  1 个（原来 10 个）

整词 token 的 E_pad 行 = 该词在 dmeta 分词后所有 token 嵌入的均值，
这样 V1 的软查表路径也能用。

从已有 ckpt 微调：字符部分的行原样继承，新增的整词行随机初始化。
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
WAV_ORIG = ROOT / "wavs"
WAV_EN = ROOT / "wavs_en"
JOBS_ORIG = ROOT / "intent_dataset.jsonl"
JOBS_EN = ROOT / "en_dataset.jsonl"
CKPT_IN = ROOT / "model" / "intent.pt"
SAVE = ROOT / "model_wt"
SAVE.mkdir(exist_ok=True)
OUT = ROOT / "wordtoken_report.json"

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
R = 3; SEED = 0; TAU = 0.1
EPOCHS = 160; LR = 1.5e-3; BS = 32
LAMBDA_CTC = 2.0
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


print("=" * 78); print("① 词表 = 字符 + 整词 token"); print("=" * 78)
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
SHARED_CH = sorted(set(a_single) & set(b2i))          # 6628 字符
K_CH = len(SHARED_CH)

WJ = json.loads((ROOT / "en_words.json").read_text(encoding="utf-8"))
EN_WORDS = sorted(WJ["words"], key=len, reverse=True)   # 长的优先匹配
WORD_TOKEN = {w: f"<w:{w}>" for w in EN_WORDS}
SHARED = SHARED_CH + [WORD_TOKEN[w] for w in EN_WORDS]  # ⭐ 扩展词表
K = len(SHARED)
char2i = {c: i + 1 for i, c in enumerate(SHARED)}       # 0 = blank
word2i = {w: char2i[WORD_TOKEN[w]] for w in EN_WORDS}
print(f"  字符 {K_CH} + 整词 {len(EN_WORDS)} = {K}")
print(f"  整词 token 例: {[WORD_TOKEN[w] for w in EN_WORDS[:5]]}")

CLS_ID = blines.index("[CLS]")
Bfull = torch.load(str(B_DIR / "pytorch_model.bin"), map_location="cpu", weights_only=True)
E_ALL = Bfull["embeddings.word_embeddings.weight"].float()
tok = AutoTokenizer.from_pretrained(str(B_DIR))

# ── E_pad：字符行 + 整词行（整词 = 该词 dmeta 分词后嵌入的均值）──
rows = [torch.zeros(1, 768)] + [E_ALL[b2i[c]].unsqueeze(0) for c in SHARED_CH]
with torch.no_grad():
    for w in EN_WORDS:
        ids = tok(w, add_special_tokens=False)["input_ids"]
        rows.append(E_ALL[ids].mean(0, keepdim=True) if ids else torch.zeros(1, 768))
E_PAD = torch.cat(rows, 0).to(DEV)
CLS_ROW = E_ALL[CLS_ID].to(DEV)
print(f"  E_pad {tuple(E_PAD.shape)}（整词行 = 该词 dmeta 嵌入均值）")


def encode_target(text):
    """⭐ 文本 → CTC 目标序列：英文整词塌成 1 个 token"""
    out, i = [], 0
    tl = text.lower()
    while i < len(text):
        hit = None
        for w in EN_WORDS:
            if tl.startswith(w, i):
                hit = w; break
        if hit:
            out.append(word2i[hit]); i += len(hit)
        else:
            c = text[i]
            if c in char2i:
                out.append(char2i[c])
            i += 1
    return out


def ctc_decode(logits):
    """⭐ 整词 token 解码回词"""
    idx = logits.argmax(-1).tolist()
    out, prev = [], -1
    for i in idx:
        if i != prev and i != 0:
            t = SHARED[i - 1]
            out.append(t[3:-1] if t.startswith("<w:") else t)
        prev = i
    return "".join(out)


# ══════════ 数据 ══════════
def load(jp, wd, tag):
    js = [json.loads(l) for l in io.open(jp, encoding="utf-8") if l.strip()]
    return [{"id": j["id"], "text": j["text"], "intent": j["intent"], "split": j["split"],
             "wav": wd / f"{j['id']}.wav", "tag": tag}
            for j in js if (wd / f"{j['id']}.wav").exists()]

ALL = load(JOBS_ORIG, WAV_ORIG, "orig") + load(JOBS_EN, WAV_EN, "en")
CK = torch.load(str(CKPT_IN), map_location="cpu", weights_only=False)
INTENTS = CK["intents"]; i2c = {o: i for i, o in enumerate(INTENTS)}; C = len(INTENTS)
tr = [i for i, j in enumerate(ALL) if j["split"] == "train"]
te_en = [i for i, j in enumerate(ALL) if j["split"] == "val" and j["tag"] == "en"]
te_or = [i for i, j in enumerate(ALL) if j["split"] == "val" and j["tag"] == "orig"]
print(f"\n  数据 {len(ALL)} 条   train {len(tr)}   val(英文) {len(te_en)}   val(原) {len(te_or)}")

# 目标长度对比
n_old = sum(len([c for c in ALL[i]["text"] if c in {x: 1 for x in SHARED_CH}]) for i in tr)
n_new = sum(len(encode_target(ALL[i]["text"])) for i in tr)
print(f"  训练集目标总长度: 字符编码 {n_old} → 整词编码 {n_new}（省 {n_old-n_new}，{(1-n_new/n_old)*100:.1f}%）")

print(); print("=" * 78); print("② A 编码器（含变速增强）"); print("=" * 78)
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
FR = {}
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
        if (n_ + 1) % 400 == 0:
            print(f"    {n_+1}/{len(tr)}  {time.time()-t0:.0f}s", flush=True)
    for i in te_en + te_or:
        wav, sr = sf.read(str(ALL[i]["wav"]), dtype="float32")
        f = fe(wav, sampling_rate=16000, return_tensors="pt", padding=False,
               return_attention_mask=True)
        FR[(i, 1.0)] = A(f["input_features"].to(DEV),
                         f["attention_mask"].to(DEV).long()).last_hidden_state.float()
del A; torch.cuda.empty_cache()
print(f"  完成 {time.time()-t0:.0f}s")

print(); print("=" * 78); print("③ B（冻结）"); print("=" * 78)
bert = BertModel(BertConfig.from_json_file(str(B_DIR / "config.json")),
                 add_pooling_layer=False).to(DEV).eval()
bert.load_state_dict(Bfull, strict=False)
for p in bert.parameters():
    p.requires_grad = False


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
        # ⭐ 字符行原样继承，整词行随机初始化
        with torch.no_grad():
            self.ctc.weight.data[:K_CH + 1] = CK["ctc_head"][:K_CH + 1]
            self.ctc.bias.data[:K_CH + 1] = CK["ctc_bias"][:K_CH + 1]
            nn.init.normal_(self.ctc.weight.data[K_CH + 1:], std=0.02)
            nn.init.zeros_(self.ctc.bias.data[K_CH + 1:])
        self.tau = TAU

    def forward(self, h):
        logits = self.ctc(h)
        return torch.softmax(logits / self.tau, -1) @ E_PAD, logits


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


print(); print("=" * 78); print("④ 训练"); print("=" * 78)
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
net = Net().to(DEV)
# ⚠️ 必须先赋权重再 .to(DEV) —— 反过来会把 CPU 张量塞进 CUDA 模块，
#    报 "Expected all tensors to be on the same device"（实测踩过两次）
head = nn.Linear(1536, C)
head.weight.data = CK["head_w"].clone(); head.bias.data = CK["head_b"].clone()
head = head.to(DEV)
ps = list(net.parameters()) + list(head.parameters())
print(f"  可训练 {sum(p.numel() for p in ps)/1e6:.2f} M  epochs {EPOCHS}  lr {LR}")
opt = torch.optim.AdamW(ps, lr=LR, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS)
tgt = [torch.tensor(encode_target(ALL[i]["text"]), dtype=torch.long) for i in range(len(ALL))]
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
        return 0.0, 0.0, []
    exact = 0; n = 0; preds = []; bad = []
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
                ok = hyp == ALL[i]["text"]
                exact += int(ok); n += 1
                if not ok and len(bad) < 8:
                    bad.append((ALL[i]["text"], hyp))
    acc = (np.array(preds) == y_all[idxs]).mean()
    print(f"  {label:14s} 意图 {acc*100:6.2f}%   转写完全对 {exact/max(n,1)*100:6.2f}%")
    for ref, hyp in bad[:6]:
        print(f"      说 {ref:26s} 转 {hyp}")
    return acc, exact / max(n, 1), bad


a_en, x_en, bad_en = evaluate(te_en, "英文专项 val")
a_or, x_or, _ = evaluate(te_or, "原数据集 val")

print()
print("  对照（整词 token 之前）:")
print("    原数据集 val 转写 77.18%  意图 98.69%")
print("    英文名字转写 —— 系统性丢首字母（Blender→lender）")

torch.save({"ctc_head": net.ctc.weight.data.cpu(), "ctc_bias": net.ctc.bias.data.cpu(),
            "head_w": head.weight.data.cpu(), "head_b": head.bias.data.cpu(),
            "shared_chars": SHARED, "en_words": EN_WORDS, "intents": INTENTS,
            "tau": TAU, "cls_id": CLS_ID, "R": R, "K_ch": K_CH},
           str(SAVE / "intent.pt"))
io.open(SAVE / "meta.json", "w", encoding="utf-8").write(json.dumps(
    {"intents": INTENTS, "K": K, "K_ch": K_CH, "en_words": EN_WORDS,
     "tau": TAU, "R": R, "cls_id": CLS_ID, "acc_test": float(a_or),
     "ctc_exact_orig": float(x_or), "ctc_exact_en": float(x_en),
     "arch": "v5_wordtoken"}, ensure_ascii=False, indent=1))
io.open(OUT, "w", encoding="utf-8").write(json.dumps(
    {"en_intent": float(a_en), "en_exact": float(x_en),
     "orig_intent": float(a_or), "orig_exact": float(x_or),
     "K": K, "K_ch": K_CH, "n_en_words": len(EN_WORDS),
     "target_len_old": n_old, "target_len_new": n_new}, ensure_ascii=False, indent=1))
print(f"\n-> {SAVE/'intent.pt'}   {OUT}")
