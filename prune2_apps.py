# -*- coding: utf-8 -*-
"""① apps 只留用户点名的 13 个  ② 加 DeepSeek 网址条目（与应用区分）。"""
import json
from pathlib import Path

P = Path(__file__).parent / "intent_config.json"
d = json.loads(P.read_text(encoding="utf-8"))

KEEP = ["任务管理器", "文件管理器", "命令行", "PowerShell", "Edge浏览器",
        "Steam", "Blender", "Ollama", "DeepSeek", "Trae", "Clash", "Python",
        "媒体播放器"]
before = [e["name"] for e in d["apps"]]
d["apps"] = [e for e in d["apps"] if e["name"] in KEEP]
after = [e["name"] for e in d["apps"]]
missing = [k for k in KEEP if k not in after]

# DeepSeek 网址：与应用同名/同别名，但靠「网页/网站/页面」后缀路由到 open_url
#   规则里 _finish_open 先看 url_suffix/file_suffix，再看 apps → urls → files，
#   所以「打开DeepSeek网页」会被强制走 open_url ✓
if not any(e["name"] in ("DeepSeek网页", "DeepSeek官网") for e in d["urls"]):
    d["urls"].append({
        "name": "DeepSeek网页",
        "aliases": ["deepseek", "deepseek网址", "deepseek官网", "deepseek网页"],
        "url": "https://chat.deepseek.com",
    })

d["behavior"]["_comment_pruned2"] = (
    "apps 按用户点名的 13 个收敛（删掉 注册表/磁盘清理/资源监视器/系统信息/"
    "远程桌面/屏幕键盘/放大镜/Vulkan测试/Node 等）。新增 urls:DeepSeek网页 "
    "→ https://chat.deepseek.com；与应用 DeepSeek 同别名，靠触发词后缀区分："
    "「打开DeepSeek」=启动应用，「打开DeepSeek网页/官网/网址」=开网页。"
    "热词表只跟随闭集，所以条目增减会自动同步到热词。")

P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"apps {len(before)} → {len(after)}")
print("  保留:", "、".join(after))
if missing:
    print("  ⚠️ 没找到(原配置里没有):", missing)
print("  删除:", "、".join(n for n in before if n not in KEEP))
print("  urls 里的 DeepSeek 条目:", [e for e in d["urls"] if "deepseek" in
                                     str(e).lower()])
