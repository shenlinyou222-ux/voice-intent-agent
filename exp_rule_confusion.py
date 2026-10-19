# -*- coding: utf-8 -*-
"""规则层在 2031 条语料上的混淆统计（纯规则，不跑 ASR，几秒出结果）。

关心两件事：
  ① 整体规则准确率（parse 的意图 == 标签）
  ② file_search 的漏/误（新加的「在文件里搜」「查找X文件」规则到底赚还是赔）
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, parse       # noqa: E402

CFG = Config("intent_config.json")
rows = [json.loads(l) for l in
        open(Path(__file__).parent / "intent_dataset.jsonl", encoding="utf-8")]

ok = 0
conf = Counter()
miss_fs = []      # 标签 file_search，规则没给 file_search
fp_fs = []        # 规则给 file_search，标签不是
for r in rows:
    got, _a = parse(r["text"], CFG)
    truth = r["intent"]
    ok += int(got == truth)
    if got != truth:
        conf[(truth, got)] += 1
    if truth == "file_search" and got != "file_search":
        miss_fs.append((r["text"], got))
    if got == "file_search" and truth != "file_search":
        fp_fs.append((r["text"], truth))

print(f"规则准确率（parse == 标签）: {ok}/{len(rows)} = {ok/len(rows)*100:.2f}%")
print(f"\nfile_search 漏报 {len(miss_fs)} 条（标签是 file_search，规则没给）:")
for t, g in miss_fs[:25]:
    print(f"   {g:12s} 「{t}」")
print(f"\nfile_search 误报 {len(fp_fs)} 条（规则给了，标签不是）:")
for t, g in fp_fs[:15]:
    print(f"   标签{g:12s} 「{t}」")

print(f"\n主要混淆 top12:")
for (truth, got), n in conf.most_common(12):
    print(f"   {truth:14s} → {got:14s} {n}")
