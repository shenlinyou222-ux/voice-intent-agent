# -*- coding: utf-8 -*-
"""从 app.asar 里解出 session 相关包的 JS，扫出 GUI 用的 RPC endpoint 名字。

asar 格式：16 字节头 + JSON 索引，文件数据从 (16+header_len) 开始，
每个文件条目的 offset 相对该起点。
"""
import json
import re
import struct
import sys
from pathlib import Path

ASAR = Path(r"C:\Users\user\AppData\Local\Programs\DeepSeek Harness\resources\app.asar")
OUT = Path(__file__).parent / "_asar_x"

# 只关心这几个包的 JS（会话控制 / 会话 UI / agent）
WANT = ("dsh-api-session-controller", "dsh-client-ui-session", "dsh-agent-loop",
        "dsh-session-persistence", "dsh-api-workspace-controller")


def walk(node, prefix=""):
    if isinstance(node, dict):
        if "files" in node:
            for name, child in node["files"].items():
                yield from walk(child, prefix + "/" + name)
        else:
            yield prefix, node


def main():
    data = ASAR.read_bytes()
    hlen = struct.unpack("<I", data[12:16])[0]
    header = json.loads(data[16:16 + hlen].decode("utf-8"))
    base = 16 + hlen
    files = list(walk(header))
    print(f"asar 内文件 {len(files)} 个；头长 {hlen}")

    OUT.mkdir(exist_ok=True)
    got = []
    for path, meta in files:
        if not any(w in path for w in WANT):
            continue
        if not path.endswith((".js", ".mjs", ".cjs")):
            continue
        off = int(meta.get("offset", -1))
        size = int(meta.get("size", 0))
        if off < 0:
            continue
        blob = data[base + off: base + off + size]
        dst = OUT / re.sub(r"[^\w.\-]", "_", path.strip("/"))
        dst.write_bytes(blob)
        got.append((path, dst, size))
    print(f"解出 {len(got)} 个 JS：")
    for path, dst, size in got[:20]:
        print(f"  {size/1024:8.0f}KB  {path}")

    # 扫 endpoint 字面量：形如 xxx/yyy 且含 session/agent/chat
    PAT = re.compile(rb"[\"']([a-z][a-z0-9\-]{2,24}/[a-z][a-z0-9\-]{2,24})[\"']")
    hits = {}
    for path, dst, _s in got:
        for m in PAT.finditer(dst.read_bytes()):
            s = m.group(1).decode("utf-8", "ignore")
            if any(k in s for k in ("session", "agent", "chat", "workspace",
                                    "turn", "project")):
                hits[s] = hits.get(s, 0) + 1
    print(f"\n候选 endpoint {len(hits)} 个：")
    for s, n in sorted(hits.items(), key=lambda kv: -kv[1])[:40]:
        print(f"  {n:4d}  {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
