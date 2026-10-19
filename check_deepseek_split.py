# -*- coding: utf-8 -*-
"""验证「应用 vs 网页」能不能分开（用户要求：打开deepseek网页）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, parse, resolve, resolve_closed   # noqa: E402

c = Config("intent_config.json")
CASES = ["打开DeepSeek", "启动deepseek", "打开DeepSeek网页", "打开deepseek官网",
         "打开deepseek网站", "打开知乎网页", "打开知乎", "启动Steam"]
print(f"{'说的':<22}{'意图':<11}{'参数':<12}{'命中':<14}目标")
print("-" * 100)
for t in CASES:
    ri, ra = parse(t, c)
    if ri == "unknown":
        print(f"{t!r:<22}{ri:<11}{ra!r:<12}{'-':<14}-")
        continue
    r = resolve(ri, t, c)
    hit = resolve_closed(ri, r, c)
    tgt = (hit.get("path") or hit.get("url")) if hit else "-"
    name = hit["name"] if hit else None
    print(f"{t!r:<22}{ri:<11}{r!r:<12}{str(name):<14}{tgt[:52]}")
