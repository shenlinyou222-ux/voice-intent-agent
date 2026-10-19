# -*- coding: utf-8 -*-
"""把 build_exe.ps1 重新存成 UTF-8 with BOM

Windows PowerShell 5.1 对【无 BOM 的 UTF-8】按 ANSI 读 → 中文乱码 →
字符串引号配对错乱 → 报 UnexpectedToken / MissingExpression。
这个坑我已经踩了三次，所以单独写个脚本固化住。
"""
import io
import pathlib
import sys

for name in sys.argv[1:]:
    p = pathlib.Path(name)
    if not p.exists():
        print(f"  [!] 不存在 {p}")
        continue
    raw = p.read_bytes()
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    s = io.open(p, encoding="utf-8-sig").read()
    io.open(p, "w", encoding="utf-8-sig", newline="\r\n").write(s)
    print(f"  [OK] {p.name}  BOM {had_bom} → True   {len(s.splitlines())} 行")
