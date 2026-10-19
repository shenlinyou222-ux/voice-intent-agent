# -*- coding: utf-8 -*-
r"""Qwen3-ASR 词表裁剪核算（不改文件，只出账）

两种口径：
  A 宽口径：256 字节 token + 所有能解成 CJK/ASCII/数字/标点/空白的 token（含英文多字符合并）
  B 严口径（设计 step1_merge 用的那条，151,643 -> 25,554）：256 字节 + CJK/标点/数字 + 【单字符】ASCII

⚠️ tie_word_embeddings=True → embed_tokens 与 lm_head 是同一块 151936×1024 权重，
   所以节省只算一次（我上一条按两块算多了，这里更正）。

用法: E:\py312\python.exe asr_vocab_trim.py
"""
import json
import os
import struct

D = r"E:\models\qwen3-asr-0.6b-official"

cfg = json.load(open(os.path.join(D, "config.json"), encoding="utf-8"))
tc = cfg["thinker_config"]
txt = tc["text_config"]
HID = txt["hidden_size"]
TIED = txt.get("tie_word_embeddings", False)
AUDIO_SPECIAL = {tc.get("audio_token_id"), tc.get("audio_start_token_id"),
                 tc.get("audio_end_token_id")}

with open(os.path.join(D, "model.safetensors"), "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
rows = hdr["thinker.model.embed_tokens.weight"]["shape"][0]

vocab = json.load(open(os.path.join(D, "vocab.json"), encoding="utf-8"))
id2tok = {v: k for k, v in vocab.items()}


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    k = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + k)
            k += 1
    return dict(zip(bs, [chr(c) for c in cs]))


u2b = {v: k for k, v in bytes_to_unicode().items()}


def tok_bytes(t):
    try:
        return bytes(u2b[c] for c in t)
    except (KeyError, ValueError):
        return None


def is_special(t, idx):
    return idx in AUDIO_SPECIAL or (t.startswith("<|") and t.endswith("|>"))


def kind_of(s):
    """把已解码字符串归类成集合"""
    cats = set()
    for ch in s:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            cats.add("CJK")
        elif 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF:
            cats.add("CJK标点")
        elif ch.isdigit():
            cats.add("数字")
        elif o < 128 and (ch.isalpha() or ch.isdigit() or ch in " \t\n"):
            cats.add("ASCII单字")
        elif 32 <= o < 127:
            cats.add("ASCII符号")
        else:
            cats.add("其他")
    return cats


def analyze(strict):
    keep, drop = [], {}
    for t, idx in vocab.items():
        if is_special(t, idx):
            keep.append(t)
            continue
        b = tok_bytes(t)
        if b is None or len(b) == 1:
            keep.append(t)          # 256 个基础字节 token，汉字靠它们拼，必须留
            continue
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            drop["非法UTF8"] = drop.get("非法UTF8", 0) + 1
            continue
        cats = kind_of(s)
        ok = cats <= {"CJK", "CJK标点", "数字", "ASCII单字", "ASCII符号"}
        if strict:
            # 严口径：ASCII 只留单字符（丢掉英文多字符合并）
            ok = ok and not (len(s) > 1 and all(ord(c) < 128 for c in s))
        if ok:
            keep.append(t)
        else:
            for c in cats - {"CJK", "CJK标点", "数字", "ASCII单字", "ASCII符号"}:
                drop[c] = drop.get(c, 0) + 1
    return keep, drop


print(f"hidden_size={HID}  tie_word_embeddings={TIED}  词表行数={rows}  tokenizer 条目={len(vocab)}")
print(f"特殊 token（<|...|> 与 audio_*）: {sum(1 for t, i in vocab.items() if is_special(t, i))} 个")
old = rows * HID * (1 if TIED else 2)
print(f"\nembed(+lm_head{'' if TIED else '×2'}) 现状: {old/1e6:.2f} M 参数  "
      f"(bf16 {old*2/1e6:.0f} MB)")
print(f"   其中未使用的 padding 行: {rows - len(vocab)} 行 "
      f"= {(rows-len(vocab))*HID/1e6:.2f} M")

for name, strict in (("A 宽口径(留英文合并)", False), ("B 严口径(只留单字符ASCII)", True)):
    keep, drop = analyze(strict)
    new = len(keep) * HID * (1 if TIED else 2)
    print(f"\n-- {name} --")
    print(f"   保留 {len(keep):>7} / {len(vocab)}  ({len(keep)/len(vocab)*100:5.1f}%)"
          f"   丢弃 {len(vocab)-len(keep)}")
    print(f"   裁剪后 embed(+lm_head) : {new/1e6:7.2f} M 参数  (bf16 {new*2/1e6:.0f} MB)")
    print(f"   节省                  : {(old-new)/1e6:7.2f} M 参数"
          f"  (bf16 {(old-new)*2/1e6:.0f} MB)")
    if drop:
        print(f"   丢弃原因: {dict(sorted(drop.items(), key=lambda x: -x[1])[:5])}")

# 编码自检：常用命令词在两种口径下还能不能被编码
words = ["打开知乎", "启动Blender", "精确查找llama3-8b", "搜索文件季度报告", "今天天气怎么样"]
print("\n-- 自检：这些词的字都在保留集里吗（按严口径）--")
keep_strict, _ = analyze(True)
keepset = set(keep_strict)
for w in words:
    bs = w.encode("utf-8")
    missing = [hex(b) for b in bs if bytes_to_unicode()[b] not in keepset]
    print(f"   {w:20s} 缺字节 token: {missing if missing else '无 ✅'}")
