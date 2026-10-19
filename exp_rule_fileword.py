# -*- coding: utf-8 -*-
"""量「②b 文件宾语规则」在语料上的净收益（不是凭感觉）。

对 2031 条语料文本跑规则 parse，比标签：
  · 新规则修好了多少条（原来 web_search、现在 file_search、标签 file_search）
  · 又弄坏了多少条（标签本来是 web_search，被新规则抢走）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, parse       # noqa: E402

CFG = Config("intent_config.json")
rows = [json.loads(l) for l in
        open(Path(__file__).parent / "intent_dataset.jsonl", encoding="utf-8")]

fix = []
broke = []
for r in rows:
    got, _arg = parse(r["text"], CFG)
    truth = r["intent"]
    has_fileword = ("文件" in r["text"] or "文档" in r["text"])
    if got == truth:
        continue
    if has_fileword and got == "file_search" and truth == "web_search":
        broke.append(r["text"])
    elif has_fileword and truth == "file_search":
        fix.append((r["text"], got))

print(f"语料 {len(rows)} 条")
print(f"\n被新规则判成 file_search 但标签是 web_search（可能弄坏）: {len(broke)}")
for t in broke[:20]:
    print(f"   「{t}」")
print(f"\n含文件/文档且标签是 file_search（应修好，看 parse 现在给什么）: {len(fix)}")
ok = sum(1 for _t, g in fix if g == "file_search")
for t, g in fix[:20]:
    print(f"   {g:12s} 「{t}」")
print(f"\n其中 parse 现在判对 {ok}/{len(fix)}")
print(f"\n净收益（修好 - 弄坏）= {ok - len(broke)}")
