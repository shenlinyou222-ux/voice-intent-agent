# -*- coding: utf-8 -*-
"""导出全部闭集条目 + 路径存在性检查 + 分类，供人工审减。

输出：
  ① 终端表格（分类 / 名称 / 别名 / 目标 / 路径在不在）
  ② 命令清单_待审.md —— 你可以直接在表格里划掉不要的
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).parent
CFG = json.loads((HERE / "intent_config.json").read_text(encoding="utf-8"))

# 分类规则（按名字/路径猜用途，只用于分组，不影响执行）
CATS = [
    ("浏览器", ("edge", "chrome", "firefox", "浏览器")),
    ("开发/AI 工具", ("powershell", "命令行", "cmd", "python", "node", "trae",
                  "clash", "ollama", "blender", "vulkan", "deepseek", "harness",
                  "code", "git", "docker", "wsl", "terminal")),
    ("游戏/娱乐", ("steam", "game", "游戏", "epic", "wegame", "原神", "minecraft",
                "媒体播放器", "网易云", "腾讯视频", "爱奇艺", "b站", "哔哩哔哩")),
    ("系统工具", ("任务管理器", "文件管理器", "资源管理器", "注册表", "磁盘清理",
               "资源监视器", "系统信息", "远程桌面", "屏幕键盘", "放大镜",
               "设置", "控制面板")),
    ("记事/计算（你说用不上）", ("记事本", "notepad", "计算器", "calc")),
    ("agent 启动器（未来）", ("workbuddy", "qwen", "agent", "dsh")),
    ("网址-常用", ("知乎", "github", "huggingface", "modelscope", "csdn", "掘金",
                "百度", "微博")),
    ("网址-购物/视频", ("淘宝", "京东", "豆瓣", "小红书", "腾讯视频", "爱奇艺",
                   "网易云", "b站")),
    ("文档/资料", ("cuda", "规约", "矩阵", "密码", "语音综述", "数据集", "方案")),
]


def cat_of(name, target=""):
    s = (name + " " + target).lower()
    for c, keys in CATS:
        if any(k.lower() in s for k in keys):
            return c
    return "其它"


def exists(target, kind):
    if kind == "urls":
        return "—"
    p = Path(target)
    return "✅ 在" if p.exists() else "❌ 丢了"


rows = []
for kind in ("apps", "urls", "files"):
    for e in CFG.get(kind, []):
        tgt = e.get("path") or e.get("url") or ""
        rows.append({
            "kind": kind,
            "name": e["name"],
            "aliases": "、".join(e.get("aliases", [])),
            "target": tgt,
            "exists": exists(tgt, kind),
            "cat": cat_of(e["name"], tgt),
        })

# 终端表格
print(f"{'类别':<22}{'类型':<7}{'名称':<16}{'别名':<24}{'路径':<8}目标")
print("-" * 118)
for c, _ in CATS + [("其它", ())]:
    for r in rows:
        if r["cat"] != c:
            continue
        print(f"{r['cat']:<22}{r['kind']:<7}{r['name']:<16}"
              f"{r['aliases'][:22]:<24}{r['exists']:<8}{r['target'][:60]}")

print(f"\n合计：apps {sum(1 for r in rows if r['kind']=='apps')} / "
      f"urls {sum(1 for r in rows if r['kind']=='urls')} / "
      f"files {sum(1 for r in rows if r['kind']=='files')}")
dead = [r for r in rows if r["exists"] == "❌ 丢了"]
print(f"路径失效（建议直接删）：{len(dead)}")
for r in dead:
    print(f"   {r['name']}  → {r['target']}")

# 写一份可审的 md
out = HERE / "命令清单_待审.md"
L = ["# 语音命令清单（待你审减）", "",
     "用法：在【保留?】列写 `y` / `n`，或直接删掉整行。改完发我，我同步进 "
     "`intent_config.json`（**不用重训模型** —— 规则直接查闭集）。", "",
     f"- 可执行程序 apps：{sum(1 for r in rows if r['kind']=='apps')} 条",
     f"- 网址 urls：{sum(1 for r in rows if r['kind']=='urls')} 条",
     f"- 文档 files：{sum(1 for r in rows if r['kind']=='files')} 条", ""]
for c, _ in CATS + [("其它", ())]:
    sub = [r for r in rows if r["cat"] == c]
    if not sub:
        continue
    L += [f"## {c}", "", "| 保留? | 名称 | 别名（都能说） | 目标 | 路径 |",
          "|---|---|---|---|---|"]
    for r in sub:
        L.append(f"|  | {r['name']} | {r['aliases']} | `{r['target'][:70]}` | "
                 f"{r['exists']} |")
    L.append("")
out.write_text("\n".join(L), encoding="utf-8")
print(f"\n已写 {out}")
