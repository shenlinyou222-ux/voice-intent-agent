# -*- coding: utf-8 -*-
r"""诊断：拉丁词被 tokenize 成什么？为什么单碎片屏蔽没生效？

打印 "blender"/"GitHub"/"Olimpia" 等的 token id 序列，
对比 _collect_latin_tokens.py 收集的 27 个碎片 id，找出遗漏。
"""
import sys

sys.path.insert(0, r"C:\Users\user\Desktop\ds harness\voice_agent")

from transformers import AutoProcessor  # noqa: E402

D = r"E:\models\qwen3-asr-0.6b-official"
proc = AutoProcessor.from_pretrained(D)
tokz = proc.tokenizer

WORDS = ["blender", "Blender", "github", "GitHub", "Olimpia",
         "deepseek", "DeepSeek", "modelscope", "ModelScope"]

for w in WORDS:
    ids = tokz.encode(w, add_special_tokens=False)
    parts = [tokz.decode([i]) for i in ids]
    print(f"{w:<12} → {ids}  {parts}")
