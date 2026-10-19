# -*- coding: utf-8 -*-
"""urls: 删豆瓣，加豆包（用户画像是科技技术宅，豆瓣不用）。"""
import json
from pathlib import Path

P = Path(__file__).parent / "intent_config.json"
d = json.loads(P.read_text(encoding="utf-8"))

before = [e["name"] for e in d["urls"]]
d["urls"] = [e for e in d["urls"] if e["name"] != "豆瓣"]
if not any(e["name"] == "豆包" for e in d["urls"]):
    d["urls"].append({
        "name": "豆包",
        "aliases": ["豆包", "doubao", "豆包ai", "字节豆包"],
        "url": "https://www.doubao.com",
    })
after = [e["name"] for e in d["urls"]]

d["behavior"]["_comment_urls_profile"] = (
    "用户画像=科技技术宅：删掉豆瓣（不用），加豆包 https://www.doubao.com。"
    "热词表只跟随闭集，所以豆包会自动进热词、豆瓣自动出热词。")

P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("urls:", len(before), "→", len(after))
print("  删:", [n for n in before if n == "豆瓣"], " 加: 豆包")
print("  现在:", "、".join(after))
