#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OpenRA 玩家/脚本可控 API 清点器（只读源码，不改动 OpenRA）。

输出:
  openra/api_inventory.json   —— 机器可读的全量清单
  openra/API_INVENTORY.md     —— 人读的统计报告

清点四类「玩家一般会调用」的 API:
  1) Lua 脚本 API   : Scripting/Global/*.cs（引擎级全局）+ Scripting/Properties/*.cs（Actor/Player 级）
  2) 指令(order) API: 所有 order 字符串（谁能处理 / 谁会发出）
  3) 聊天命令 API   : Commands/*.cs（玩家在聊天框敲的 /xxx）
  4) Bot/AI API     : Traits/BotModules/*.cs（自动化控制器用的同一套指令面）
"""
import json
import os
import re
import sys
from collections import defaultdict, OrderedDict

ROOT = r"E:\OpenRA"
OUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RE_PUBLIC = re.compile(
    r"^\s*public\s+(?:(?:static|virtual|override|sealed|async|new|readonly|partial|extern)\s+)*"
    r"([A-Za-z_][\w<>\[\],\.\?\s]*?)\s+([A-Za-z_]\w*)\s*\(",
)
RE_CLASS = re.compile(r"^\s*(?:public|internal|sealed|abstract|static|\s)*class\s+([A-Za-z_]\w*)", re.M)
RE_REGISTER_CMD = re.compile(r'RegisterCommand\(\s*"([^"]+)"')
RE_DICT_KEY = re.compile(r'\{\s*"([a-z0-9\-]+)"\s*,')
RE_CONST_CMD = re.compile(r'public\s+const\s+string\s+\w+\s*=\s*"([^"]+)"')
RE_DESC = re.compile(r'\[Desc\("([^"]*)"')
RE_ORDER_CMP = re.compile(r'OrderString\s*(?:==|!=)\s*"([^"]+)"')
RE_ORDER_CASE = re.compile(r'^\s*case\s+"([^"]+)"\s*:')
RE_NEW_ORDER = re.compile(r'new\s+Order\(\s*"([^"]+)"')
RE_CHAT_CMD = re.compile(r'Command\s*=>\s*"([^"]+)"')
RE_SCRIPT_ALIAS = re.compile(r'\[ScriptPropertyGroup\("([^"]+)"\)\]')
RE_SCRIPT_PROP = re.compile(r'\[ScriptProperty(?:Group)?\("([^"]+)"\)\]')


def read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def cs_files(*subdirs):
    out = []
    for sub in subdirs:
        base = os.path.join(ROOT, sub)
        for dirpath, _dirs, files in os.walk(base):
            if os.sep + "obj" + os.sep in dirpath + os.sep:
                continue
            for fn in files:
                if fn.endswith(".cs"):
                    out.append(os.path.join(dirpath, fn))
    return sorted(out)


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"^\s*//.*$", "", text, flags=re.M)
    return text


def public_methods(text):
    """粗解析 public 方法：名字 + 参数个数 + 返回类型。"""
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = RE_PUBLIC.match(line)
        if m:
            ret, name = m.group(1).strip(), m.group(2)
            if name in ("get", "set", "if", "for", "while", "switch", "return"):
                i += 1
                continue
            # 收集参数（可能有换行）
            buf = line
            depth = line.count("(") - line.count(")")
            j = i
            while depth > 0 and j + 1 < len(lines):
                j += 1
                buf += " " + lines[j].strip()
                depth += lines[j].count("(") - lines[j].count(")")
            args = buf.split("(", 1)[1].rsplit(")", 1)[0] if "(" in buf else ""
            argc = 0 if not args.strip() else len(re.split(r",(?![^<]*>)", args))
            out.append({"name": name, "ret": ret, "argc": argc, "args": args.strip()[:160]})
            i = j + 1
            continue
        i += 1
    return out


def desc_for(text, name):
    """取方法前面最近的一个 [Desc("...")]。"""
    idx = text.find(name + "(")
    if idx < 0:
        return ""
    head = text[max(0, idx - 400):idx]
    ds = RE_DESC.findall(head)
    return ds[-1] if ds else ""


def inventory_scripting():
    """Lua 脚本 API。"""
    groups = OrderedDict()
    src = {
        "global": cs_files(r"OpenRA.Mods.Common\Scripting\Global", r"OpenRA.Game\Scripting"),
        "actor": [p for p in cs_files(r"OpenRA.Mods.Common\Scripting\Properties")
                  if os.path.basename(p).endswith("Properties.cs")],
        "cnc": cs_files(r"OpenRA.Mods.Cnc\Scripting"),
    }
    for kind, files in src.items():
        for path in files:
            text = strip_comments(read(path))
            cls = None
            m = RE_CLASS.search(text)
            if m:
                cls = m.group(1)
            methods = public_methods(text)
            if not methods:
                continue
            groups.setdefault(kind, []).append({
                "class": cls,
                "file": os.path.relpath(path, ROOT),
                "methods": [{"name": x["name"], "argc": x["argc"], "desc": desc_for(text, x["name"])}
                            for x in methods],
            })
    return groups


def inventory_orders():
    """order 字符串：谁处理、谁发出。"""
    handled = defaultdict(set)
    issued = defaultdict(set)
    for path in cs_files(r"OpenRA.Mods.Common", r"OpenRA.Mods.Cnc", r"OpenRA.Mods.D2k", r"OpenRA.Game"):
        text = strip_comments(read(path))
        rel = os.path.relpath(path, ROOT)
        cls = None
        m = RE_CLASS.search(text)
        if m:
            cls = m.group(1)
        if "IResolveOrder" in text:
            for name in RE_ORDER_CMP.findall(text):
                handled[name].add(cls or rel)
            for line in text.splitlines():
                cm = RE_ORDER_CASE.match(line)
                if cm:
                    handled[cm.group(1)].add(cls or rel)
        for name in RE_NEW_ORDER.findall(text):
            issued[name].add(cls or rel)
    return handled, issued


def inventory_chat_commands():
    out = []
    for path in cs_files(r"OpenRA.Mods.Common\Commands"):
        text = strip_comments(read(path))
        cmds = set(RE_REGISTER_CMD.findall(text)) | set(RE_DICT_KEY.findall(text)) | set(RE_CONST_CMD.findall(text))
        cmds.discard("name")
        if not cmds:
            continue
        cls = None
        m = RE_CLASS.search(text)
        if m:
            cls = m.group(1)
        out.append({
            "file": os.path.relpath(path, ROOT),
            "class": cls,
            "commands": sorted(cmds),
            "desc": (RE_DESC.findall(text) or [""])[0],
        })
    return out


def inventory_bots():
    out = []
    for path in cs_files(r"OpenRA.Mods.Common\Traits\BotModules", r"OpenRA.Mods.Common\Traits"):
        if "BotModules" not in path:
            continue
        text = strip_comments(read(path))
        cls = None
        m = RE_CLASS.search(text)
        if m:
            cls = m.group(1)
        if not cls:
            continue
        orders = sorted(set(RE_NEW_ORDER.findall(text)))
        out.append({
            "class": cls,
            "file": os.path.relpath(path, ROOT),
            "desc": (RE_DESC.findall(text) or [""])[0],
            "issues_orders": orders,
        })
    return out


def inventory_traits():
    """模组/规则层可见的 trait 总数（= modding API 面）。"""
    stats = defaultdict(int)
    descs = []
    for path in cs_files(r"OpenRA.Mods.Common", r"OpenRA.Mods.Cnc", r"OpenRA.Mods.D2k"):
        text = strip_comments(read(path))
        for m in re.finditer(r"class\s+(\w+)Info\s*:\s*([\w<>,\.\s]+?)(?:\n|\{)", text):
            stats[os.path.relpath(path, ROOT).split(os.sep)[0]] += 1
            d = RE_DESC.findall(text[:m.start() + 200])
            if d:
                descs.append({"trait": m.group(1), "desc": d[-1], "file": os.path.relpath(path, ROOT)})
    return stats, descs


def main():
    scripting = inventory_scripting()
    handled, issued = inventory_orders()
    chat = inventory_chat_commands()
    bots = inventory_bots()
    trait_stats, trait_descs = inventory_traits()

    data = {
        "openra_root": ROOT,
        "scripting": scripting,
        "orders": {
            "handled": {k: sorted(v) for k, v in sorted(handled.items())},
            "issued": {k: sorted(v) for k, v in sorted(issued.items())},
        },
        "chat_commands": chat,
        "bots": bots,
        "trait_counts": dict(trait_stats),
        "trait_total": sum(trait_stats.values()),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "api_inventory.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    # ---- 报告 ----
    L = []
    A = L.append
    A("# OpenRA 玩家可控 API 清点（自动生成）\n")
    A(f"源码：`{ROOT}`　生成脚本：`openra/tools/api_inventory.py`\n")

    total_lua = sum(len(g["methods"]) for kind in scripting.values() for g in kind)
    A("## 0. 总量\n")
    A("| 面 | 数量 |")
    A("|---|---|")
    for kind, groups in scripting.items():
        n = sum(len(g["methods"]) for g in groups)
        A(f"| Lua 脚本 API（{kind}）| {n} 个方法 / {len(groups)} 个对象 |")
    A(f"| **Lua 脚本 API 合计** | **{total_lua}** |")
    A(f"| 指令(order)字符串（有实现者）| {len(handled)} |")
    A(f"| 指令(order)字符串（代码里被发出）| {len(issued)} |")
    A(f"| 聊天命令（/xxx）| {sum(len(c['commands']) for c in chat)} |")
    A(f"| Bot 模块 | {len(bots)} |")
    A(f"| Trait（模组 API 面）| {data['trait_total']} |")
    A("")

    A("## 1. Lua 脚本 API（玩家/地图脚本直接调用）\n")
    for kind, groups in scripting.items():
        A(f"### 1.{list(scripting).index(kind)+1} {kind}（{sum(len(g['methods']) for g in groups)} 方法）\n")
        A("| 对象 | 文件 | 方法数 | 方法 |")
        A("|---|---|---|---|")
        for g in sorted(groups, key=lambda x: -(len(x["methods"]))):
            names = ", ".join(f"`{m['name']}`" for m in g["methods"][:24])
            if len(g["methods"]) > 24:
                names += f" … (+{len(g['methods'])-24})"
            A(f"| {g['class']} | {g['file']} | {len(g['methods'])} | {names} |")
        A("")

    A("## 2. 指令(order) API\n")
    A("### 2.1 引擎/模组能处理的指令（玩家点一下就会产生）\n")
    A("| 指令 | 处理者 |")
    A("|---|---|")
    for k in sorted(handled):
        A(f"| `{k}` | {', '.join(sorted(handled[k]))} |")
    A("")
    A("### 2.2 代码里会被发出的指令（含 AI/Bot）\n")
    A("| 指令 | 发出者 |")
    A("|---|---|")
    for k in sorted(issued):
        A(f"| `{k}` | {', '.join(sorted(issued[k])[:6])} |")
    A("")

    A("## 3. 聊天命令 API（玩家在聊天框输入）\n")
    A("| 文件 | 类 | 命令 |")
    A("|---|---|---|")
    for c in chat:
        A(f"| {c['file']} | {c['class']} | {', '.join('`/'+x+'`' for x in c['commands'])} |")
    A("")

    A("## 4. Bot/AI 模块（自动化控制的既有范式）\n")
    A("| 模块 | 说明 | 会发出的指令 |")
    A("|---|---|---|")
    for b in sorted(bots, key=lambda x: x["class"]):
        A(f"| {b['class']} | {b['desc'][:60]} | {', '.join('`'+x+'`' for x in b['issues_orders'][:8])} |")
    A("")

    A("## 5. Trait 统计（模组层 API 面）\n")
    A("| 程序集 | TraitInfo 数 |")
    A("|---|---|")
    for k, v in sorted(data["trait_counts"].items(), key=lambda x: -x[1]):
        A(f"| {k} | {v} |")
    A("")

    with open(os.path.join(OUT_DIR, "API_INVENTORY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    print(f"lua methods      : {total_lua}")
    print(f"orders handled   : {len(handled)}")
    print(f"orders issued    : {len(issued)}")
    print(f"chat commands    : {sum(len(c['commands']) for c in chat)}")
    print(f"bot modules      : {len(bots)}")
    print(f"traits           : {data['trait_total']}")
    print(f"written          : {os.path.join(OUT_DIR, 'api_inventory.json')}")
    print(f"written          : {os.path.join(OUT_DIR, 'API_INVENTORY.md')}")


if __name__ == "__main__":
    sys.exit(main())
