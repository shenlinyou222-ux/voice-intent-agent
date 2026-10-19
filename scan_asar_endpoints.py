# -*- coding: utf-8 -*-
"""从 app.asar 里挖 GUI 用的 RPC endpoint 名字。

GUI ↔ host 是 RPC over HTTP：POST /api/<endpoint>，客户端 JS 里会有 endpoint 字面量
（已见：goals/create、commands/list、commands/execute）。
这里把 asar 当字节流扫「xxx/yyy」形式的候选，按关键词过滤。
"""
import re
import sys
from pathlib import Path

ASAR = Path(r"C:\Users\user\AppData\Local\Programs\DeepSeek Harness\resources\app.asar")

PAT = re.compile(rb"[a-z][a-z0-9\-]{2,24}/[a-z][a-z0-9\-]{2,24}")
KEY = (b"session", b"agent", b"chat", b"conversation", b"thread", b"turn",
       b"workspace", b"project")


def main():
    data = ASAR.read_bytes()
    print(f"asar {len(data)/1e6:.0f} MB，扫描中 ...")
    hits = {}
    for m in PAT.finditer(data):
        s = m.group(0)
        if not any(k in s for k in KEY):
            continue
        # 排除明显不是 endpoint 的（路径/资源/域名）
        if s.startswith((b"http", b"node_modules", b"dist/", b"lib/")):
            continue
        hits.setdefault(s, 0)
        hits[s] += 1
    print(f"候选 {len(hits)} 个（按出现次数）:")
    for s, n in sorted(hits.items(), key=lambda kv: -kv[1])[:60]:
        print(f"  {n:5d}  {s.decode('utf-8', 'ignore')}")
    # 单独找“创建/新建”语义
    print("\n含 create/new/open/start 的:")
    for s, n in sorted(hits.items(), key=lambda kv: -kv[1]):
        if any(w in s for w in (b"create", b"/new", b"open", b"start", b"prompt",
                                b"send")):
            print(f"  {n:5d}  {s.decode('utf-8', 'ignore')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
