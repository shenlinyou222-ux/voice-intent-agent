# -*- coding: utf-8 -*-
"""voice_agent 实时性能监控（终端仪表盘）。

看什么（都是实测需要的）：
  · 进程：RSS / 私有提交 / CPU / 线程数 —— 内存涨到 3GB+ 就是"感觉慢"的原因之一
  · ASR：最近 N 句的耗时 min/中位/max、ms/token、句/分钟
  · 采集：队列满/驱动溢出（丢音！）—— 这是"转写变烂"的第一嫌疑
  · 模型：加载/卸载事件、意图模型构造耗时
  · 错误：❌/异常 计数
  · 最近几行日志（带上下文）

用法：
    python perf_monitor.py                 # 每秒刷新，Ctrl+C 退出
    python perf_monitor.py --once          # 只打一屏（我自测用）
    python perf_monitor.py --interval 2 --log run_src.log
"""
import argparse
import re
import statistics
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).parent
CLEAR = "\x1b[2J\x1b[H"


def find_agent():
    """找 voice_agent.py 的 python 进程 → (pid, rss_mb, priv_mb, cpu_s, threads)"""
    try:
        ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              "Where-Object { $_.CommandLine -match 'voice_agent\\.py' } | "
              "ForEach-Object { $p=Get-Process -Id $_.ProcessId; "
              "'{0}|{1}|{2}|{3}|{4}' -f $_.ProcessId, $p.WorkingSet64, "
              "$p.PrivateMemorySize64, $p.TotalProcessorTime.TotalSeconds, "
              "$p.Threads.Count }")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=20).stdout
        rows = [l.strip() for l in out.splitlines() if "|" in l]
        if not rows:
            return None
        pid, ws, priv, cpu, th = rows[-1].split("|")
        return (int(pid), int(ws) / 1e6, int(priv) / 1e6, float(cpu), int(th))
    except Exception:
        return None


RE_ASR = re.compile(r"\[(\d\d:\d\d:\d\d)\]\s+ASR\s+解.*?(\d+)ms\s+(\d+)tok")
RE_SEG = re.compile(r"ASR\s+「.*?」\s+\(?([\d.]+)s\)?")
RE_DROP = re.compile(r"队列满\s+(\d+)\s*/\s*驱动溢出\s+(\d+)")
RE_LEVEL = re.compile(r"电平\s+([\d.]+).*?VAD概率\s+([\d.]+)")
RE_MODEL = re.compile(r"(int8 产物|纯文本产物|载入\s+\d+\s+MB GGUF|已卸载模型|模型就绪)")
RE_ERR = re.compile(r"❌|Traceback|执行异常|ValueError|AttributeError")


def _read_log(p):
    """读日志：Tee-Object（Windows PowerShell 5.1）默认写 UTF-16LE，
    按 UTF-8 读会变成 '[ 0 9 : 2 5 ]' 这种夹空字节的乱码 —— 要探测编码。"""
    try:
        raw = Path(p).read_bytes()
    except Exception:
        return "", 0
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):          # BOM
        enc = "utf-16"
    elif raw.count(b"\x00") > max(8, len(raw) // 8):   # 大量 NUL → UTF-16
        enc = "utf-16-le"
    else:
        enc = "utf-8"
    return raw.decode(enc, errors="ignore"), len(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(HERE / "run_src.log"))
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=30, help="统计最近多少句 ASR")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()

    log = Path(a.log)
    lat = deque(maxlen=a.window)
    tok = deque(maxlen=a.window)
    segs = deque(maxlen=a.window)
    recent = deque(maxlen=8)
    off, drops, errs = 0, (0, 0), 0
    started = time.time()
    last_ts = None

    while True:
        txt, blen = _read_log(log)
        if len(txt) > off:
            for line in txt[off:].splitlines():
                off += len(line) + 1
                recent.append(line)
                m = RE_ASR.search(line)
                if m:
                    lat.append(int(m.group(2)))
                    tok.append(max(1, int(m.group(3))))
                    last_ts = m.group(1)
                m = RE_DROP.search(line)
                if m:
                    drops = (int(m.group(1)), int(m.group(2)))
                if RE_ERR.search(line):
                    errs += 1
            if not txt.endswith("\n"):
                off -= 1
        proc = find_agent()
        now = time.time()
        lines = []
        lines.append("=" * 78)
        lines.append(f"  voice_agent 实时性能   刷新 {time.strftime('%H:%M:%S')}"
                     f"   监控已运行 {now-started:.0f}s")
        lines.append("=" * 78)
        if proc:
            pid, rss, priv, cpu, th = proc
            lines.append(f"  进程     PID {pid}   RSS {rss:,.0f} MB   "
                         f"私有 {priv:,.0f} MB   CPU {cpu:.1f}s   线程 {th}")
        else:
            lines.append("  进程     ⚠️ 没找到 voice_agent.py 进程（是不是没在跑？）")
        if lat:
            lines.append(f"  ASR      最近 {len(lat)} 句  min {min(lat)}ms  "
                         f"中位 {statistics.median(lat):.0f}ms  max {max(lat)}ms")
            lines.append(f"           {statistics.mean(lat)/max(1,statistics.mean(tok)):.0f}"
                         f" ms/token   平均 {statistics.mean(lat):.0f}ms/句")
        else:
            lines.append("  ASR      还没有转写（说句话）")
        if segs:
            lines.append(f"  语音段   最近 {len(segs)} 段，平均 {statistics.mean(segs):.2f}s")
        flag = "⚠️ 在丢音！" if drops[0] or drops[1] else "✓"
        lines.append(f"  采集     队列满 {drops[0]} / 驱动溢出 {drops[1]}  {flag}")
        lines.append(f"  错误     ❌/异常 累计 {errs}")
        lines.append("-" * 78)
        lines.append("  最近日志：")
        for l in list(recent)[-6:]:
            lines.append("    " + l[:110])
        out = "\n".join(lines) + "\n"
        if a.once:
            sys.stdout.write(out)
            return 0
        sys.stdout.write(CLEAR + out)
        sys.stdout.flush()
        time.sleep(a.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
