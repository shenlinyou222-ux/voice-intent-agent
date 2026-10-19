# -*- coding: utf-8 -*-
"""修补构建环境里 setuptools 的 vendored jaraco.text

问题：PyInstaller 打包后，`Lorem ipsum.txt`（文件名带空格）不会被收进来，
      `files(__name__).joinpath('Lorem ipsum.txt').read_text()` 抛
      FileNotFoundError → exe 一启动就崩在 pyi_rth_pkgres 运行时钩子上。
修法：读不到就置空字符串。
"""
import ast
import io
import pathlib
import re

p = pathlib.Path(r"E:\voice_agent_build\Lib\site-packages\setuptools\_vendor\jaraco\text\__init__.py")
s = io.open(p, encoding="utf-8").read()

m = re.search(r"^lorem_ipsum: str = \(\n.*?\n\)$", s, re.M | re.S)
if not m:
    print("  ❌ 没找到 lorem_ipsum 赋值")
    raise SystemExit(1)
print("  原代码:")
for line in m.group(0).splitlines():
    print("    " + line)

new = (
    "# ⚠️ 本地修补（2026-10-11）：PyInstaller 打包后 'Lorem ipsum.txt'（带空格）\n"
    "#    不会被收进来 → read_text() 抛 FileNotFoundError → exe 一启动就崩。\n"
    "#    读不到就置空。\n"
    "try:\n"
    "    lorem_ipsum: str = (\n"
    "        files(__name__).joinpath('Lorem ipsum.txt').read_text(encoding='utf-8')\n"
    "    )\n"
    "except Exception:\n"
    "    lorem_ipsum = ''"
)
s = s.replace(m.group(0), new, 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
ast.parse(s)
print("  ✅ 已打补丁，语法 OK")

# 顺手也确认 _internal 里没有那个文件（打包后就是没有，所以才要这个补丁）
print(f"  文件本身在源码树里存在: {(p.parent / 'Lorem ipsum.txt').exists()}")
