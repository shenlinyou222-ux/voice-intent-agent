# -*- coding: utf-8 -*-
"""运维语音域的【长输出窗口层】—— 拉一个显式 Windows Terminal 窗口跑命令并高亮。

═══ 为什么需要 ═══
    短输出用 rich 渲染在 agent 窗口就够了；但「看这个 yaml」「看 pod 为什么挂了」
    这种输出几十上百行，需要：
      · 能滚动、能 grep、能接着手动操作
      · 不污染 agent 自己的日志流
    → 拉一个**显式窗口**，跑完用 `exec bash` 留住。

═══ ⚠️ 血债：别把命令内联传给 wt.exe ═══
    实测：内联传参时 **PowerShell 会剥引号、吃转义** —— 窗口标题变成
        `196m红色警告\033[0m`     ← \033 被吃掉，ANSI 根本没生效
    这和 cmd.exe 剥引号是同一类问题（路径含空格时多引号命令行会被劈开）。

    ✅ 解法：**写脚本文件，让窗口跑文件**。
       命令行里只剩一个路径（无空格、无转义、无引号），命令内容全在文件里。

═══ ⚠️ 第二个坑：换行必须是 LF ═══
    `Path.write_text(..., newline="\\n")` —— 写 CRLF 的话 bash 会报
        `$'\\r': command not found`

═══ 第三个坑：窗口要留住 ═══
    末尾加 `exec bash`，否则命令跑完窗口就关了，看不到输出。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

WSL_DISTRO = "Ubuntu-24.04"

# Windows 临时目录 → WSL 路径（C:\Users\x\AppData\Local\Temp → /mnt/c/Users/x/AppData/Local/Temp）
def _win_to_wsl(p: str) -> str:
    p = str(p)
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return p.replace("\\", "/")


# 高亮管道（与 profile.json 的 display.highlight 对应）
HL_YAML = "pygmentize -l yaml -f terminal256 -O style=monokai"
HL_JSON = "jq -C ."
HL_LOG = "pygmentize -l accesslog -f terminal256"


def available() -> bool:
    """wt.exe 是否可用。"""
    return shutil.which("wt.exe") is not None or shutil.which("wt") is not None


def build_script(shell_cmd: str, title: str = "ops", keep_open: bool = True,
                 hl: str | None = None) -> str:
    """把 shell 命令包成脚本内容。

    hl: 可选高亮管道名（'yaml' | 'json' | 'log' | None）。
        给了就把命令输出管进高亮器（`<cmd> | <highlighter>`）。
    """
    pipe = ""
    if hl == "yaml":
        pipe = f" | {HL_YAML}"
    elif hl == "json":
        pipe = f" | {HL_JSON}"
    elif hl == "log":
        pipe = f" | {HL_LOG}"

    lines = [
        "#!/bin/bash",
        f'echo -e "\\033[1;36m══ {title} ══\\033[0m"',
        "echo",
        f"{shell_cmd}{pipe}",
        "echo",
    ]
    if keep_open:
        lines += [
            'echo -e "\\033[38;5;245m（窗口留着，可继续手动操作）\\033[0m"',
            "exec bash",
        ]
    return "\n".join(lines) + "\n"


def open_in_window(shell_cmd: str, title: str = "ops", hl: str | None = None,
                   keep_open: bool = True, dry: bool = False):
    """在显式拉起的 Windows Terminal 里跑命令（带高亮），跑完留窗口。

    返回 (ok, msg)。
    """
    if dry:
        return True, f"[dry] 会在窗口跑：{shell_cmd}" + (f" | {hl} 高亮" if hl else "")

    if not available():
        return False, "找不到 wt.exe（Windows Terminal），无法拉窗口"

    # ① 写脚本文件（★ newline="\n" —— 别写 CRLF）
    tmpdir = Path(tempfile.gettempdir())
    name = f"ops_{time.strftime('%H%M%S')}_{abs(hash(shell_cmd)) % 10000:04d}.sh"
    sh = tmpdir / name
    try:
        sh.write_text(build_script(shell_cmd, title, keep_open, hl),
                      encoding="utf-8", newline="\n")
    except Exception as e:
        return False, f"写脚本失败：{type(e).__name__}: {e}"

    # ② 拉起窗口（★ 命令行里只有一个路径 —— 绕开全部引号问题）
    wsl_path = _win_to_wsl(sh)
    try:
        subprocess.Popen(
            ["wt.exe", "-w", "0", "nt",
             "wsl.exe", "-d", WSL_DISTRO, "--",
             "bash", wsl_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        return False, f"拉窗口失败：{type(e).__name__}: {e}"

    return True, f"已在窗口打开：{title}"


# ── 常用封装 ──────────────────────────────────────────────────────────
def show_yaml_file(path: str, title: str | None = None):
    """看一个 yaml 文件（高亮）。"""
    return open_in_window(f"cat '{path}'", title or f"yaml: {Path(path).name}", hl="yaml")


def show_pod_yaml(pod: str, ns: str = ""):
    """看 pod 的 yaml（高亮）。"""
    n = f" -n {ns}" if ns else " -A"
    return open_in_window(f"kubectl get pod {pod}{n} -o yaml",
                          f"pod yaml: {pod}", hl="yaml")


def show_pod_why(pod: str, ns: str = ""):
    """★ 看 pod 为什么挂了 —— describe 的 Events 段是首选线索。"""
    n = f" -n {ns}" if ns else " -A"
    cmd = (f"kubectl describe pod {pod}{n} | head -80\n"
           f"echo\n"
           f"echo -e '\\033[1;33m--- 上一个容器的日志（CrashLoop 时必看）---\\033[0m'\n"
           f"kubectl logs {pod}{n} --tail=80 -p 2>&1 | tail -80")
    return open_in_window(cmd, f"为什么挂了: {pod}", hl=None)


def show_promql(promql: str, title: str = "PromQL"):
    """跑一条 PromQL 并高亮 JSON 结果。"""
    cmd = ("curl -sG 'http://127.0.0.1:30090/api/v1/query' "
           f"--data-urlencode 'query={promql}'")
    return open_in_window(cmd, title, hl="json")


def show_logs_journal(unit: str):
    return open_in_window(f"journalctl -u {unit} -n 200 --no-pager", f"日志: {unit}", hl="log")


def open_terminal(cwd_wsl: str = "~"):
    """只开一个 WSL 终端（不跑命令）。"""
    return open_in_window(f"cd {cwd_wsl}", "WSL 终端", hl=None)
