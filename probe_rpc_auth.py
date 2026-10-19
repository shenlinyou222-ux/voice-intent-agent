# -*- coding: utf-8 -*-
"""探测：用哪种鉴权头能调通这个正在运行的 desktop 应用的 RPC。

目标 endpoint: POST http://127.0.0.1:19387/api/session/create
凭据：~/.dsh/.credentials.yaml → records["client-connection/browser-session"].payload.secret
body 故意传 {} —— 让它的校验错误告诉我们必填字段（比翻代码快）。
**不打印密钥值**，只打印状态码和响应体。
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

URL = "http://127.0.0.1:19387/api/session/create"
CRED = Path.home() / ".dsh" / ".credentials.yaml"


def secret():
    txt = CRED.read_text(encoding="utf-8")
    if yaml:
        d = yaml.safe_load(txt)
    else:
        # 退化：正则抠
        import re
        m = re.search(r"browser-session:.*?secret:\s*(\S+)", txt, re.S)
        return m.group(1).strip("\"'") if m else None
    try:
        return d["records"]["client-connection/browser-session"]["payload"]["secret"]
    except Exception:
        return None


def post(headers, body=b"{}"):
    req = urllib.request.Request(URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Host", "127.0.0.1:19387")
    req.add_header("Origin", "http://127.0.0.1:19387")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read()[:600]
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:600]
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}".encode()


def main():
    sec = secret()
    print(f"browser-session secret: {'已读到（长度 %d，不打印）' % len(sec) if sec else '读不到'}")
    TRIES = [
        ("无鉴权头", {}),
        ("Authorization: Bearer", {"Authorization": f"Bearer {sec}"} if sec else {}),
        ("X-DSH-Auth-Token", {"X-DSH-Auth-Token": sec} if sec else {}),
        ("Cookie: dsh-session", {"Cookie": f"dsh-session={sec}"} if sec else {}),
        ("x-browser-session", {"x-browser-session": sec} if sec else {}),
    ]
    for name, h in TRIES:
        if not h:
            continue
        code, body = post(h)
        txt = body.decode("utf-8", "ignore")
        print(f"\n--- {name} → HTTP {code}")
        print("   ", txt[:400].replace("\n", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
