# -*- coding: utf-8 -*-
r"""统计词表里拉丁 token 的规模（决定 bad_words 还是 logit_bias 策略）"""
import re
import sys

sys.path.insert(0, r"C:\Users\user\Desktop\ds harness\voice_agent")

from transformers import AutoProcessor  # noqa: E402

D = r"E:\models\qwen3-asr-0.6b-official"
proc = AutoProcessor.from_pretrained(D)
tokz = proc.tokenizer

raw = tokz.convert_ids_to_tokens(list(range(len(tokz))))
bs = (list(range(ord("!"), ord("~") + 1))
      + list(range(ord("\xa1"), ord("\xac") + 1))
      + list(range(ord("\xae"), ord("\xff") + 1)))
cs = bs[:]
k = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + k)
        k += 1
c2b = {chr(c): b for b, c in zip(bs, cs)}


def dec(t):
    try:
        return bytes(c2b[ch] for ch in t).decode("utf-8", "ignore")
    except Exception:
        return t or ""


LATIN_RUN = re.compile(r"^[ ]*[a-zA-Z]+$")
STRIP = re.compile(r"[^a-zA-Z]")

latin = [dec(t) for t in raw if LATIN_RUN.match(dec(t))]
print("纯拉丁 token 数（含前导空格）:", len(latin))
print("  其中带前导空格:", sum(1 for s in latin if s.startswith(" ")))
print("  样例:", latin[:20])

# 拉丁+数字/符号混合（resnet50 / B站 类，非纯 run）
MIX = re.compile(r"[a-zA-Z]")
mixed = [dec(t) for t in raw
         if dec(t) and not LATIN_RUN.match(dec(t)) and MIX.search(dec(t))]
print("含拉丁但非纯 run 的 token:", len(mixed), mixed[:15])

# 大写/小写分布：屏蔽后对缩写的影响
upper = [s for s in latin if any(c.isupper() for c in s)]
print("含大写的拉丁 token:", len(upper), upper[:15])
