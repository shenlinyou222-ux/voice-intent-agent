# -*- coding: utf-8 -*-
"""删掉用户明确用不上的条目：记事本 / 计算器（还有它们的别名 notepad、calc）。
热词表已改成只跟随闭集，所以删条目=热词里也一起消失。"""
import json
from pathlib import Path

P = Path(__file__).parent / "intent_config.json"
d = json.loads(P.read_text(encoding="utf-8"))
DROP = {"记事本", "计算器"}
before = [e["name"] for e in d["apps"]]
d["apps"] = [e for e in d["apps"] if e["name"] not in DROP]
after = [e["name"] for e in d["apps"]]
d["behavior"]["_comment_pruned"] = (
    "删掉用户明确用不上的条目：" + "、".join(sorted(DROP)) +
    "（别名 notepad/calc 一并消失）。用户原话："
    "「把这个傻逼notepad删掉我他妈根本用不上」。热词表已改成【只跟随闭集】，"
    "所以删条目 = 热词里同时去掉，不会再从 meta.json 的静态英文词表漏回来。")
P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("apps:", len(before), "→", len(after))
print("  删掉:", [n for n in before if n in DROP])
print("  剩余:", "、".join(after))
