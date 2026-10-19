# -*- coding: utf-8 -*-
"""把规则层最大的两类错例捞出来看句式（web_search→unknown 120 条最值钱）。"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, parse       # noqa: E402

CFG = Config("intent_config.json")
rows = [json.loads(l) for l in
        open(Path(__file__).parent / "intent_dataset.jsonl", encoding="utf-8")]

ws_unknown = [r["text"] for r in rows
              if r["intent"] == "web_search" and parse(r["text"], CFG)[0] == "unknown"]
la_unknown = [r["text"] for r in rows
              if r["intent"] == "launch_app" and parse(r["text"], CFG)[0] == "unknown"]
ou_unknown = [r["text"] for r in rows
              if r["intent"] == "open_url" and parse(r["text"], CFG)[0] == "unknown"]

print(f"web_search→unknown {len(ws_unknown)} 条：")
for t in ws_unknown[:40]:
    print(f"   「{t}」")
print(f"\nlaunch_app→unknown {len(la_unknown)} 条：")
for t in la_unknown[:25]:
    print(f"   「{t}」")
print(f"\nopen_url→unknown {len(ou_unknown)} 条：")
for t in ou_unknown[:20]:
    print(f"   「{t}」")

# 开头两字分布，找缺失的触发词
print("\nweb_search 漏网句的开头 3 字分布：")
for k, v in Counter(t[:3] for t in ws_unknown).most_common(15):
    print(f"   {k!r:8s} {v}")
