# -*- coding: utf-8 -*-
"""拿 session/create 的请求字段 + 找出信任栅栏要的请求头。"""
import json
import re
import struct
from pathlib import Path

X = Path(__file__).parent / "_asar_x"
RC = X / "dsh_node_modules__deepseek-ai_dsh-api-session-controller_lib_typert.remote-client.js"
ASAR = Path(r"C:\Users\user\AppData\Local\Programs\DeepSeek Harness\resources\app.asar")


def main():
    t = RC.read_text(encoding="utf-8", errors="ignore")
    # 1) session/create 的参数 schema
    m = re.search(r"(_deepseek_ai_dsh_api_session_controller_session_create_parameter_0\$schema\s*=\s*\{)",
                  t)
    if m:
        start = m.start(1)
        # 取到该赋值的结尾（下一个顶层 };\n）
        end = t.find("\n};", start)
        block = t[start:end if end > 0 else start + 2500]
        print("=== session/create 参数 schema ===")
        # 只挑字段名/必填/类型
        for mm in re.finditer(r"(\w+)\s*:\s*\{([^{}]{0,200})", block):
            print(f"  {mm.group(1):22s} {mm.group(2).strip()[:90]}")
        req = re.search(r"required\s*:\s*\[([^\]]*)\]", block)
        print("  required:", req.group(1) if req else "(未标注)")
    else:
        print("没找到 schema；直接看 SessionCreateRequest 相关行")
        for line in t.splitlines():
            if "SessionCreateRequest" in line or "createSession" in line:
                print("  ", line.strip()[:120])

    # 2) 信任栅栏：扫 asar 里的自定义请求头名
    print("\n=== asar 里可能的鉴权/信任请求头 ===")
    data = ASAR.read_bytes()
    pats = [rb"x-dsh[a-z0-9\-]{2,30}", rb"x-harness[a-z0-9\-]{2,30}",
            rb"x-browser[a-z0-9\-]{2,30}", rb"authorization", rb"bearer ",
            rb"trusted-host", rb"x-dsh-secret", rb"browser-session"]
    for p in pats:
        hits = {}
        for m2 in re.finditer(p, data, re.I):
            s = m2.group(0).decode("utf-8", "ignore").lower()
            hits[s] = hits.get(s, 0) + 1
        if hits:
            top = sorted(hits.items(), key=lambda kv: -kv[1])[:6]
            print(f"  {p.decode():18s} → " + ", ".join(f"{k}({v})" for k, v in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
