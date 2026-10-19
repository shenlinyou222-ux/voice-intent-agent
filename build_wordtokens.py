# -*- coding: utf-8 -*-
"""⭐ 整词 token 方案 —— 从根上修「英文丢首字母」

═══ 问题 ═══
CTC 输出表是 6628 个字符（中文为主 + a-z）。英文单词要 6~12 个连续单字符 token：
    Blender → B|l|e|n|d|e|r     7 个尖峰
而音频里是 2~3 个中文口音的音节。CTC 单调对齐要在 7 个音素事件上各落一个尖峰，
首字母的尖峰被前一个汉字吞掉 → 丢首字母。词越长丢得越多（PowerShell→owerhell）。

═══ 方案（业界标准做法）═══
给闭集里那 22 个英文词各加【一个整词 token】：
    Blender → <w:blender>       1 个尖峰
6628 → 6650，softmax 几乎不变，但对齐问题消失。

参考：
  · arXiv:2407.04368 romanization + balanced tokenizer（词表 −63.51%，SEAME +13.72%）
  · arXiv:1811.00241 BPE subword for CS-ASR
  · arXiv:1811.09021 bytes are all you need（日英 CS −38.6%）

═══ 配套 ═══
  · 整词 token 的 E_pad 行 = 该词在 dmeta 分词后所有 token 嵌入的【均值】
    （这样 V1 的软查表路径也能用）
  · V3 路径不变（还是把整词送进 dmeta 分词器）
"""
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
CFG = json.loads((ROOT / "intent_config.json").read_text(encoding="utf-8"))

# ══════════ 抽出闭集里的英文专有名词 ══════════
# ⭐⭐ 关键：要收【中英混合名】—— B站 / Vulkan测试 / Edge浏览器
#    第一版只收了「纯 ASCII」，结果这三个还是丢首字母：
#      精确查找B站 → 精确查找站      精确查找Vulkan测试 → 精确查找ulkan测试
#    判据改成：只要含至少一个 ASCII 字母就收（长度 >= 2）
EN_WORDS = set()
for group in (CFG["apps"], CFG["urls"], CFG["files"]):
    for e in group:
        for f in [e["name"]] + list(e.get("aliases", [])):
            s = str(f).lower().strip()
            if len(s) >= 2 and any(c.isascii() and c.isalpha() for c in s):
                EN_WORDS.add(s)
EN_WORDS = sorted(EN_WORDS, key=len, reverse=True)
print(f"  闭集英文专有名词 {len(EN_WORDS)} 个（含中英混合名）:")
for w in EN_WORDS:
    print(f"    {w}")

WORD_TOKEN = {w: f"<w:{w}>" for w in EN_WORDS}
io.open(ROOT / "en_words.json", "w", encoding="utf-8").write(
    json.dumps({"words": EN_WORDS, "tokens": WORD_TOKEN}, ensure_ascii=False, indent=1))
print(f"\n-> en_words.json")

# ══════════ 演示：目标文本编码对比 ══════════
print()
print("=" * 74)
print("  目标文本编码对比（字符 vs 整词）")
print("=" * 74)
SHARED_BASE = 6628          # 原字符表大小（实测值）
for text in ["打开Blender", "启动PowerShell", "打开ModelScope", "启动Ollama",
             "打开记事本", "搜索文件季度报告"]:
    tl = text.lower()
    chars = [c for c in text]
    # 整词编码
    out, i = [], 0
    while i < len(text):
        hit = None
        for w in EN_WORDS:
            if tl.startswith(w, i):
                hit = w; break
        if hit:
            out.append(WORD_TOKEN[hit]); i += len(hit)
        else:
            out.append(text[i]); i += 1
    print(f"  {text:20s}")
    print(f"    字符编码 {len(chars):2d} 个: {'|'.join(chars)}")
    print(f"    整词编码 {len(out):2d} 个: {'|'.join(out)}")
    print(f"    → 长度 {len(chars)} → {len(out)}（省 {len(chars)-len(out)} 个尖峰）")
