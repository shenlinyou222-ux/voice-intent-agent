# -*- coding: utf-8 -*-
r"""解剖 E:\models\qwen3-asr-0.6b-official：解码器架构 + 词表可裁性

用法: E:\py312\python.exe asr_inspect.py
"""
import collections
import json
import os
import struct

D = r"E:\models\qwen3-asr-0.6b-official"


def hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ── 1. config：架构 ──
hr("① config.json —— 架构")
cfg = json.load(open(os.path.join(D, "config.json"), encoding="utf-8"))
print("model_type      :", cfg.get("model_type"))
print("architectures   :", cfg.get("architectures"))
print("tie_word_emb    :", cfg.get("tie_word_embeddings"))
for k in ("torch_dtype", "dtype", "num_parameters"):
    if k in cfg:
        print(f"{k:16s}: {cfg[k]}")

tc = cfg.get("thinker_config") or cfg.get("text_config") or {}
print("\n-- thinker_config 顶层键:", list(tc.keys()))
txt = tc.get("text_config", tc)
aud = tc.get("audio_config", {})
print("\n[文本/解码器 text_config]")
for k in ("model_type", "num_hidden_layers", "hidden_size", "intermediate_size",
          "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
          "max_position_embeddings", "rope_theta", "rms_norm_eps",
          "tie_word_embeddings", "hidden_act", "sliding_window", "use_cache"):
    if k in txt:
        print(f"   {k:26s}: {txt[k]}")
print("\n[音频编码器 audio_config]")
for k in ("model_type", "num_hidden_layers", "hidden_size", "intermediate_size",
          "num_attention_heads", "num_mel_bins", "output_dim", "n_window",
          "n_window_infer", "max_source_positions"):
    if k in aud:
        print(f"   {k:26s}: {aud[k]}")

# ── 2. safetensors 头：张量清单（不加载权重）──
hr("② model.safetensors 张量清单（只读文件头）")
p = os.path.join(D, "model.safetensors")
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
tensors = {k: v for k, v in hdr.items() if k != "__metadata__"}
print(f"文件 {os.path.getsize(p)/1e6:.1f} MB   张量 {len(tensors)} 个")

tot = collections.Counter()
for k, v in tensors.items():
    shape = v["shape"]
    numel = 1
    for s in shape:
        numel *= s
    parts = k.split(".")
    grp = ".".join(parts[:2]) if parts[0] == "thinker" else parts[0]
    tot[grp] += numel
print("\n-- 按模块统计参数 --")
for g, v in tot.most_common():
    print(f"   {g:38s} {v/1e6:10.2f} M")

print("\n-- 关键张量形状 --")
for pat in ("thinker.model.embed_tokens.weight", "thinker.lm_head.weight",
            "thinker.model.norm.weight", "thinker.model.layers.0.self_attn.q_proj.weight",
            "thinker.model.layers.0.self_attn.k_proj.weight",
            "thinker.model.layers.0.self_attn.v_proj.weight",
            "thinker.model.layers.0.self_attn.o_proj.weight",
            "thinker.model.layers.0.mlp.gate_proj.weight",
            "thinker.model.layers.0.mlp.down_proj.weight",
            "thinker.model.layers.0.input_layernorm.weight",
            "thinker.audio_tower.layers.0.self_attn.q_proj.weight",
            "thinker.audio_tower.conv2d1.weight",
            "thinker.audio_tower.layers.0.fc1.weight"):
    if pat in tensors:
        print(f"   {pat:58s} {tensors[pat]['shape']}  {tensors[pat]['dtype']}")
nlay = len([k for k in tensors if k.startswith("thinker.model.layers.")]) 
nlay = len({k.split(".")[3] for k in tensors if k.startswith("thinker.model.layers.")})
nalay = len({k.split(".")[3] for k in tensors if k.startswith("thinker.audio_tower.layers.")})
print(f"\n   解码器层数 = {nlay}    音频编码器层数 = {nalay}")

# ── 3. 词表 ──
hr("③ 词表")
vocab = json.load(open(os.path.join(D, "vocab.json"), encoding="utf-8"))
merges = open(os.path.join(D, "merges.txt"), encoding="utf-8").read().splitlines()
merges = [m for m in merges if m and not m.startswith("#")]
print(f"vocab.json 条目 : {len(vocab)}   (id 最大 {max(vocab.values())})")
print(f"merges 条数     : {len(merges)}")
emb_rows = tensors.get("thinker.model.embed_tokens.weight", {}).get("shape", [None])[0]
lm_rows = tensors.get("thinker.lm_head.weight", {}).get("shape", [None])[0]
print(f"embed_tokens    : {emb_rows} 行      lm_head: {lm_rows} 行 (tokenizer 只到 {len(vocab)})")

specials = [t for t in vocab if t.startswith("<|") or t.startswith("<")]
print(f"特殊 token      : {len(specials)} 个，例: {specials[:6]}")

# byte-level BPE 的 byte→unicode 映射（GPT-2/Qwen 同款）
def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


b2u = bytes_to_unicode()
u2b = {v: k for k, v in b2u.items()}


def tok_bytes(t):
    try:
        return bytes(u2b[c] for c in t)
    except KeyError:
        return None


def classify(txt):
    """给一个已解码的字符串归类"""
    cats = set()
    for ch in txt:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF:
            cats.add("CJK")
        elif 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF:
            cats.add("CJK标点")
        elif ch.isdigit():
            cats.add("数字")
        elif ch.isalpha() and o < 128:
            cats.add("ASCII字母")
        elif 32 <= o < 127:
            cats.add("ASCII符号")
        elif ch.isspace():
            cats.add("空白")
        else:
            cats.add("其他")
    return cats


KEEP = {"CJK", "CJK标点", "数字", "ASCII字母", "ASCII符号", "空白"}
keep, drop, nbytes = set(), collections.Counter(), 0
for t in vocab:
    b = tok_bytes(t)
    if b is None:
        keep.add(t)          # 特殊 token 一律保留
        continue
    if len(b) == 1:
        nbytes += 1
        keep.add(t)          # 256 个基础字节 token 必须全留（汉字靠字节拼）
        continue
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        drop["非法UTF8多字节"] += 1
        continue
    cats = classify(s)
    if cats <= KEEP:
        keep.add(t)
    else:
        for c in cats - KEEP:
            drop[c] += 1

print(f"\n-- 裁剪分析（保留 256 个基础字节 + CJK/ASCII/数字/标点/空白）--")
print(f"   保留 : {len(keep)}  ({len(keep)/len(vocab)*100:.1f}%)")
print(f"   丢弃 : {len(vocab)-len(keep)}   丢弃原因分布: {dict(drop.most_common(6))}")
print(f"   其中单字节 token 全部保留: {nbytes} 个")

if emb_rows and lm_rows:
    hidden = tensors["thinker.model.embed_tokens.weight"]["shape"][1]
    old = (emb_rows + lm_rows) * hidden
    new = (len(keep) + (emb_rows - len(vocab))) * hidden * 2
    print(f"\n   hidden_size = {hidden}")
    print(f"   embed+lm_head 现在   : {old/1e6:8.2f} M 参数")
    print(f"   embed+lm_head 裁剪后 : {(len(keep)+(emb_rows-len(vocab)))*hidden*2/1e6:8.2f} M 参数")
    print(f"   省                   : {(old-new)/1e6:8.2f} M 参数"
          f"  (fp32 ~{(old-new)*4/1e6:.0f} MB / bf16 ~{(old-new)*2/1e6:.0f} MB)")
    print(f"   原 lm_head 单块       : {lm_rows*hidden/1e6:.2f} M")

print("\n-- 保留样例（前 20 个非单字节）--")
shown = 0
for t in vocab:
    b = tok_bytes(t)
    if b and len(b) > 1 and t in keep:
        print(f"   {t!r:24s} -> {b.decode('utf-8', 'replace')!r}")
        shown += 1
        if shown >= 20:
            break
