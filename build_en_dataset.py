# -*- coding: utf-8 -*-
"""英文专有名词专项数据集

问题：CTC head 学不会「中文口音的音节 → 英文字母拼写」这个映射。
      每个英文名字只有 16 条训练样本，远远不够。

做法：只针对闭集里那 22 个英文名字，造大量样本。
      模板尽量多样（覆盖名字出现在句首/句尾/中间、不同动词搭配），
      再加音频增强（变速/变调）在训练时用。

为什么只做英文名字：
  · 中文部分已经 100%（实测 7 个意图 98.7~100%）
  · 英文名字是唯一的短板，集中火力
"""
import hashlib
import io
import json
import random
from pathlib import Path

ROOT = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
CFG = json.loads((ROOT / "intent_config.json").read_text(encoding="utf-8"))
SEED = 20261011
rng = random.Random(SEED)


def sid(split, intent, text):
    return f"{split}_{intent}_{hashlib.md5(text.encode('utf-8')).hexdigest()[:10]}"


# ══════════ 抽出闭集里所有含英文的条目 ══════════
EN_APPS, EN_URLS = [], []
for a in CFG["apps"]:
    if any(c.isascii() and c.isalpha() for c in a["name"]):
        EN_APPS.append(a["name"])
for u in CFG["urls"]:
    if any(c.isascii() and c.isalpha() for c in u["name"]):
        EN_URLS.append(u["name"])
print(f"  含英文的程序 {len(EN_APPS)}: {EN_APPS}")
print(f"  含英文的网址 {len(EN_URLS)}: {EN_URLS}")

# ══════════ 模板：覆盖名字的各种位置和搭配 ══════════
T_APP = [
    # 名字在句尾
    "打开{名}", "启动{名}", "运行{名}", "开一下{名}", "开个{名}", "帮我打开{名}",
    "帮我启动{名}", "给我开{名}", "我要用{名}", "来一下{名}", "打开一下{名}",
    "启动一下{名}", "运行一下{名}", "帮我运行{名}", "现在打开{名}", "现在启动{名}",
    "马上打开{名}", "赶紧打开{名}", "打开{名}吧", "启动{名}吧", "运行{名}吧",
    "帮我开一下{名}", "帮我开{名}", "给我启动{名}", "我要打开{名}", "我要启动{名}",
    # 名字在句首
    "{名}打开", "{名}启动", "{名}打开一下", "{名}启动一下", "{名}开一下",
    "{名}运行一下", "{名}给我开一下",
    # 名字在中间
    "把{名}打开", "把{名}启动", "把{名}启动起来", "把{名}打开一下",
    "帮我用{名}", "我要用一下{名}", "打开电脑上的{名}", "启动电脑上的{名}",
    "打开我的{名}", "启动我的{名}", "把电脑上的{名}打开",
    "用{名}打开", "用{名}运行",
]
T_URL = [
    "打开{名}", "开{名}", "帮我打开{名}", "访问{名}", "上{名}", "我要上{名}",
    "去{名}", "看看{名}", "打开{名}网页", "打开{名}网站", "打开{名}页面",
    "帮我开{名}", "帮我访问{名}", "我要看{名}", "上个{名}", "上一下{名}",
    "{名}打开", "{名}打开一下", "把{名}打开", "打开一下{名}", "现在打开{名}",
    "马上打开{名}", "打开{名}吧", "我想看看{名}", "帮我打开{名}网页",
    "打开{名}看看", "去{名}看看", "访问一下{名}", "打开{名}首页",
]
# 搜索类也带英文（web_search / exact_search 的参数常有英文词）
T_WEB = ["搜索{名}", "搜一下{名}", "查一下{名}", "帮我搜{名}", "帮我查{名}",
         "搜个{名}", "搜索一下{名}", "网上搜一下{名}", "查查{名}", "帮我搜索{名}"]
T_EXACT = ["精确查找{名}", "精确搜索{名}", "精确搜{名}", "帮我精确查找{名}",
           "精确查找一下{名}", "精确搜索一下{名}", "我要精确查找{名}"]
T_FSEARCH = ["搜索文件{名}", "查找文件{名}", "搜文件{名}", "找文件{名}",
             "帮我搜索文件{名}", "帮我找文件{名}", "搜一下文件{名}"]

rows = []
for name in EN_APPS:
    for t in T_APP:
        rows.append({"text": t.format(名=name), "intent": "launch_app", "kind": "en", "arg": name})
for name in EN_URLS:
    for t in T_URL:
        rows.append({"text": t.format(名=name), "intent": "open_url", "kind": "en", "arg": name})
for name in EN_APPS + EN_URLS:
    for t in T_WEB:
        rows.append({"text": t.format(名=name), "intent": "web_search", "kind": "en", "arg": name})
    for t in T_EXACT:
        rows.append({"text": t.format(名=name), "intent": "exact_search", "kind": "en", "arg": name})
    for t in T_FSEARCH:
        rows.append({"text": t.format(名=name), "intent": "file_search", "kind": "en", "arg": name})

# 去重
seen = set(); ded = []
for r in rows:
    if r["text"] in seen:
        continue
    seen.add(r["text"]); ded.append(r)
rng.shuffle(ded)

# 分层
by_i = {}
for r in ded:
    by_i.setdefault(r["intent"], []).append(r)
jobs = []
for it in sorted(by_i):
    pool = by_i[it][:]
    rng.shuffle(pool)
    k = max(1, int(round(len(pool) * 0.7)))
    for j, r in enumerate(pool):
        sp = "train" if j < k else "val"
        jobs.append({"id": sid(sp, it, r["text"]), "text": r["text"], "intent": it,
                     "kind": r["kind"], "split": sp, "arg": r["arg"]})

from collections import Counter
c = Counter(j["intent"] for j in jobs)
print()
print(f"  总样本 {len(jobs)}   train {sum(1 for j in jobs if j['split']=='train')}"
      f"   val {sum(1 for j in jobs if j['split']=='val')}")
for it in sorted(c):
    print(f"    {it:14s} {c[it]}")

io.open(ROOT / "en_dataset.jsonl", "w", encoding="utf-8").write(
    "\n".join(json.dumps(j, ensure_ascii=False) for j in jobs))
io.open(ROOT / "en_tts.tsv", "w", encoding="utf-8").write(
    "\n".join(f"{j['id']}\t{j['text']}" for j in jobs))
print(f"\n-> en_dataset.jsonl  en_tts.tsv")
