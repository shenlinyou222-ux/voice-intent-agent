# -*- coding: utf-8 -*-
"""按画像（科技技术宅）重排 urls：删购物/娱乐，补技术站，外网标 needs_proxy。

对标"打开外网站点前自动确保 Clash 在跑"（actions._ensure_clash）。
"""
import json
from pathlib import Path

P = Path(__file__).parent / "intent_config.json"
d = json.loads(P.read_text(encoding="utf-8"))

DROP = {"淘宝", "京东", "小红书", "微博", "腾讯视频", "爱奇艺", "网易云音乐", "百度"}
before = [e["name"] for e in d["urls"]]
d["urls"] = [e for e in d["urls"] if e["name"] not in DROP]

# 保留的：B站、知乎、GitHub、掘金、CSDN、HuggingFace、ModelScope、DeepSeek网页、豆包
# 给已知的外网条目补 needs_proxy
FOR_PROXY = {"github", "huggingface"}
for e in d["urls"]:
    if e["name"].lower() in FOR_PROXY:
        e["needs_proxy"] = True

NEW = [
    # ── 外网（needs_proxy=True → 打开前自动起 Clash）──
    ("Stack Overflow", ["stackoverflow", "so", "栈溢出"], "https://stackoverflow.com", True),
    ("arXiv", ["arxiv", "论文库"], "https://arxiv.org", True),
    ("PapersWithCode", ["paperswithcode", "论文代码"], "https://paperswithcode.com", True),
    ("Kaggle", ["kaggle"], "https://www.kaggle.com", True),
    ("HackerNews", ["hackernews", "hn", "黑客新闻"], "https://news.ycombinator.com", True),
    ("GoogleScholar", ["googlescholar", "谷歌学术"], "https://scholar.google.com", True),
    ("PyPI", ["pypi"], "https://pypi.org", True),
    ("DockerHub", ["dockerhub", "docker"], "https://hub.docker.com", True),
    ("Claude", ["claude", "anthropic"], "https://claude.ai", True),
    ("OpenAI", ["openai", "chatgpt", "gpt"], "https://chatgpt.com", True),
    ("YouTube", ["youtube", "油管"], "https://www.youtube.com", True),
    ("维基百科", ["wikipedia", "wiki", "维基"], "https://en.wikipedia.org", True),
    # ── 国内（直连）──
    ("开源中国", ["开源中国", "oschina"], "https://www.oschina.net", False),
    ("博客园", ["博客园", "cnblogs"], "https://www.cnblogs.com", False),
    ("Gitee", ["gitee", "码云"], "https://gitee.com", False),
    ("机器之心", ["机器之心", "jiqizhixin"], "https://www.jiqizhixin.com", False),
    ("InfoQ", ["infoq"], "https://www.infoq.cn", False),
]
have = {e["name"].lower() for e in d["urls"]}
for name, aliases, url, px in NEW:
    if name.lower() in have:
        continue
    d["urls"].append({"name": name, "aliases": aliases, "url": url,
                      "needs_proxy": px})

d["behavior"]["_comment_urls_profile"] = (
    "按用户画像（科技技术宅）重排 urls：删淘宝/京东/小红书/微博/腾讯视频/爱奇艺/"
    "网易云/百度；补技术站（外网标 needs_proxy，打开前自动确保 Clash 在跑："
    "Stack Overflow/arXiv/PapersWithCode/Kaggle/HackerNews/GoogleScholar/PyPI/"
    "DockerHub/Claude/OpenAI/YouTube/维基 + 国内 开源中国/博客园/Gitee/机器之心/InfoQ）。"
    "系统代理由 Clash Verge 自己设置，浏览器走系统代理即可。")

P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"urls {len(before)} → {len(d['urls'])}")
print("  删:", "、".join(n for n in before if n in DROP))
names = [e["name"] for e in d["urls"]]
print("  现在:", "、".join(names))
print("  需代理:", "、".join(e["name"] for e in d["urls"] if e.get("needs_proxy")))
