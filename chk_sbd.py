# -*- coding: utf-8 -*-
"""查一下业界现成的分句轮子装了没 / 版本。"""
import importlib
mods = ["pysbd", "hanlp", "ltp", "jieba", "spacy", "nltk", "stanza",
        "pyltp", "sentencepiece", "regex"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", "")
        print(f"  ✓ {m:16s} {v}")
    except Exception as e:
        print(f"  ✗ {m:16s} {type(e).__name__}")
