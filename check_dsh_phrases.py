# -*- coding: utf-8 -*-
"""日志里那几句的真实解析结果（含「启动一下DSH」）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, parse, resolve, resolve_closed   # noqa: E402

c = Config("intent_config.json")
CASES = ["启动一下DSH", "启动DSH", "打开DSH", "启动一下DeepSeek",
         "启动DSH会话", "新建DSH会话"]
for t in CASES:
    ri, ra = parse(t, c)
    if ri == "unknown":
        print(f"  {t!r:20s} → unknown  {ra!r}")
        continue
    r = resolve(ri, t, c)
    hit = resolve_closed(ri, r, c)
    print(f"  {t!r:20s} → {ri:11s} 参数={r!r:12s} 命中={bool(hit)}"
          + (f" ({hit['name']})" if hit else ""))
