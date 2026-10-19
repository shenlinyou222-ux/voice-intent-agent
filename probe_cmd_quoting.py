# -*- coding: utf-8 -*-
"""cmd.exe 引号陷阱：四种传法实测（路径含空格），选能用的那个。

背景：dsh.cmd 在 `…\DeepSeek Harness\…` 里，路径带空格。
["cmd.exe","/c",路径,…] 会被 cmd 吃掉外层引号 → 路径被劈开报
"'C:\\Users\\user\\AppData\\Local\\Programs\\DeepSeek' is not recognized"。
"""
import os
import subprocess
import tempfile
import time
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="q test "))
CLI = TMP / "fake cli.cmd"          # 名字和目录都带空格
MARK = TMP / "got.txt"
CLI.write_text(f'@echo off\r\necho %* > "{MARK}"\r\necho {{"ok":1}}\r\n',
               encoding="utf-8")
ARG = "跑一下测试"


def run(tag, cmd, **kw):
    MARK.unlink(missing_ok=True)
    try:
        p = subprocess.Popen(cmd, **kw)
        p.wait(timeout=10)
    except Exception as e:
        print(f"  {tag}: 起进程失败 {type(e).__name__}: {e}")
        return False
    for _ in range(20):
        if MARK.exists():
            break
        time.sleep(0.1)
    ok = MARK.exists()
    got = MARK.read_text(encoding="utf-8", errors="ignore").strip() if ok else "(没生成)"
    print(f"  {'✅' if ok else '❌'} {tag}: {got}")
    return ok


def main():
    print(f"CLI 路径: {CLI}\n")
    cli = str(CLI)
    PATCH = TMP / "my patch.yml"          # 第二个含空格的参数（真实场景就是这样）
    PATCH.write_text("[]\n", encoding="utf-8")
    pa = str(PATCH)
    # ⭐ 关键场景：两个含空格的参数 → 命令行里有 4 个引号，cmd 会剥外层引号
    a2 = ["cmd.exe", "/c", cli, "run", "--patch", pa, ARG]
    d2 = ["cmd.exe", "/c", "call", cli, "run", "--patch", pa, ARG]
    print("== 两个含空格参数的场景（真实情形）==")
    res = {
        "A 列表直传（多引号）": run("A 列表直传（多引号）", a2),
        "D call 内建（多引号）": run("D call 内建（多引号）", d2),
    }
    print()
    good = [k for k, v in res.items() if v]
    print(f"可用: {good}")
    print("结论：用 `cmd /c call \"<路径含空格>\" args…`（call 是内建，"
          "引号由它处理，不受 cmd 剥引号规则影响）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
