# -*- coding: utf-8 -*-
"""运维语音域的【短输出渲染层】—— 用 rich 在 agent 窗口里渲染。

为什么需要：
    运维命令输出又长又密，**念给用户听没用，得让他看**。
    但 agent 自己的窗口是日志流（[HH:MM:SS] TAG msg），直接把 df/kubectl 的原始
    输出打进去会糊成一片。所以：
      · key:value 类（nvidia-smi / 指标）→ 渲染成表格
      · yaml/json 类（describe / 配置）  → 语法高亮
      · 日志类（kubectl logs）           → 按级别着色（error 红 / warn 黄 / CrashLoop 红底）

两个必须遵守的约定：
    ① sys.stdout.isatty() —— 输出被重定向到文件时**自动关色**。
       启动器用 Tee-Object 写 run_ops_voice.log，不判断的话日志里全是 ANSI 转义码。
    ② NO_COLOR —— 尊重业界约定（https://no-color.org/）。
"""
from __future__ import annotations

import json
import os
import re
import sys

# ── 是否着色 ──────────────────────────────────────────────────────────
def color_on() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


try:
    from rich.console import Console
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich.text import Text
    _HAS_RICH = True
except Exception:                                    # rich 没装时优雅退化
    _HAS_RICH = False

_console = None


def console():
    global _console
    if _console is None:
        _console = Console(no_color=not color_on(), highlight=False) if _HAS_RICH else None
    return _console


def _plain(msg: str):
    """没有 rich（或被重定向）时的退化路径。"""
    print(re.sub(r"\[/?[a-z0-9 #_]+\]", "", msg), flush=True)


# ── 标题 ──────────────────────────────────────────────────────────────
def title(text: str):
    c = console()
    if c and color_on():
        c.print(f"[bold cyan]══ {text} ══[/]")
    else:
        print(f"== {text} ==", flush=True)


# ── key:value 表格（nvidia-smi / 指标）────────────────────────────────
def render_kv(title_text: str, rows, subtitle: str | None = None):
    """rows: [(key, value), ...]"""
    c = console()
    if not (c and color_on()):
        if title_text:
            print(f"== {title_text} ==", flush=True)
        for k, v in rows:
            print(f"  {k}: {v}", flush=True)
        return
    t = Table(title=title_text or None, border_style="dim", title_style="bold cyan",
              show_header=True, header_style="bold")
    t.add_column("指标", style="cyan", no_wrap=True)
    t.add_column("值", style="bold yellow", justify="right")
    for k, v in rows:
        t.add_row(str(k), str(v))
    c.print(t)
    if subtitle:
        c.print(f"[dim]{subtitle}[/]")


# ── 语法高亮（yaml / json）────────────────────────────────────────────
def render_code(text: str, lexer: str = "yaml", title_text: str | None = None,
                max_lines: int = 200):
    if title_text:
        title(title_text)
    body = "\n".join((text or "").splitlines()[:max_lines])
    c = console()
    if c and color_on():
        c.print(Syntax(body, lexer, theme="monokai", line_numbers=False,
                       word_wrap=False, background_color="default"))
    else:
        print(body, flush=True)


def render_yaml(text: str, title_text: str | None = None):
    render_code(text, "yaml", title_text)


def render_json(text: str, title_text: str | None = None):
    # 先尽量格式化（缩进 2），格式化失败就原样高亮
    try:
        text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except Exception:
        pass
    render_code(text, "json", title_text)


# ── 日志着色（kubectl logs / journalctl）──────────────────────────────
_ERR = re.compile(r"\b(error|err|fail|failed|fatal|panic|exception|traceback)\b", re.I)
_WARN = re.compile(r"\b(warn|warning|deprecated)\b", re.I)
_BADSTATE = re.compile(r"\b(CrashLoopBackOff|ImagePullBackOff|OOMKilled|Evicted|Error|Pending)\b")
_OK = re.compile(r"\b(Running|Ready|Succeeded|Healthy|Active: active)\b")


def render_log(text: str, title_text: str | None = None, max_lines: int = 200):
    """按级别着色 —— 让 error/CrashLoop 一眼跳出来。"""
    if title_text:
        title(title_text)
    c = console()
    lines = (text or "").splitlines()[:max_lines]
    if not (c and color_on()):
        print("\n".join(lines), flush=True)
        return
    for ln in lines:
        if _BADSTATE.search(ln):
            c.print(f"[bold red on grey11]{ln}[/]")
        elif _ERR.search(ln):
            c.print(f"[bold red]{ln}[/]")
        elif _WARN.search(ln):
            c.print(f"[bold yellow]{ln}[/]")
        elif _OK.search(ln):
            c.print(f"[green]{ln}[/]")
        else:
            c.print(ln)


# ── 自动选渲染方式 ────────────────────────────────────────────────────
def render_auto(text: str, title_text: str | None = None):
    """按内容猜格式：yaml → 高亮 / json → 高亮 / 日志 → 着色 / 其它 → 原样。"""
    t = (text or "").strip()
    if not t:
        title(title_text or "（无输出）")
        print("  （空）", flush=True)
        return
    head = t[:400]
    if t.startswith("{") or t.startswith("["):
        try:
            json.loads(t)
            return render_json(t, title_text)
        except Exception:
            pass
    if re.search(r"^[a-zA-Z_][\w\-.]*:\s", head, re.M) and "apiVersion" in head:
        return render_yaml(t, title_text)
    if _ERR.search(head) or _WARN.search(head) or _BADSTATE.search(head):
        return render_log(t, title_text)
    title(title_text or "输出")
    print(t, flush=True)


# ── 结果面板（动作返回时用）───────────────────────────────────────────
def render_result(ok: bool, msg: str, body: str = "", title_text: str | None = None):
    """动作的统一出口：一句话结论 + 可选正文。"""
    c = console()
    if c and color_on():
        icon = "[bold green]✓[/]" if ok else "[bold red]✗[/]"
        c.print(f"{icon} {msg}")
    else:
        print(("OK  " if ok else "ERR ") + msg, flush=True)
    if body:
        render_auto(body, title_text)
