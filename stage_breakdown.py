# -*- coding: utf-8 -*-
"""按【你每一次说话】拆解时间开销：说话时长 / ASR 转写 / 意图判定 / 动作执行。

回答"到底慢在哪"：如果 ASR 占绝大部分，那就确定是转写的问题。
用法： python stage_breakdown.py [--n 12] [--log run_src.log]
"""
import argparse
import re
from pathlib import Path

HERE = Path(__file__).parent


def read_log(p):
    raw = Path(p).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        enc = "utf-16"
    elif raw.count(b"\x00") > max(8, len(raw) // 8):
        enc = "utf-16-le"
    else:
        enc = "utf-8"
    return raw.decode(enc, errors="ignore")


RE_SEG = re.compile(r"\[(\d\d:\d\d:\d\d)\]\s+ASR\s+「(?P<txt>.*?)」\s+\((?P<dur>[\d.]+)s\)")
RE_ASR = re.compile(r"\[(\d\d:\d\d:\d\d)\]\s+ASR\s+解「(?P<txt>.*?)」→(?P<int>\S+)\s+"
                    r"(?P<ms>\d+)ms\s+(?P<tok>\d+)tok")
RE_INT = re.compile(r"\[(\d\d:\d\d:\d\d)\]\s+INT\s+.*?(?P<ms>\d+)ms")
RE_ACT = re.compile(r"\[(\d\d:\d\d:\d\d)\]\s+ACT\s+.*?->\s*(?P<msg>.*?)\s+(?P<ms>\d+)ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(HERE / "run_src.log"))
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()

    rows, cur = [], {}
    for line in read_log(a.log).splitlines():
        m = RE_ASR.search(line)
        if m:
            if cur.get("asr_ms"):
                rows.append(cur)
            cur = {"t": m.group(1), "txt": m.group("txt"), "int": m.group("int"),
                   "asr_ms": int(m.group("ms")), "tok": int(m.group("tok"))}
            continue
        m = RE_SEG.search(line)
        if m and cur:
            cur.setdefault("dur", float(m.group("dur")))
            continue
        m = RE_INT.search(line)
        if m and cur:
            cur.setdefault("int_ms", int(m.group("ms")))
            continue
        m = RE_ACT.search(line)
        if m and cur:
            cur.setdefault("act_ms", int(m.group("ms")))
    if cur.get("asr_ms"):
        rows.append(cur)
    rows = rows[-a.n:]
    if not rows:
        print("  没有解析到转写记录")
        return 0

    print(f"{'时间':<9}{'说话':>6}{'ASR':>8}{'ms/tok':>7}{'意图':>6}{'动作':>6}"
          f"{' 转写后延迟':>10}  {'识别':<10} 内容")
    print("-" * 104)
    tot_asr = tot_int = tot_act = 0
    for r in rows:
        d = r.get("dur", 0.0)
        i, ac = r.get("int_ms", 0), r.get("act_ms", 0)
        tot_asr += r["asr_ms"]; tot_int += i; tot_act += ac
        print(f"{r['t']:<9}{d:>5.2f}s{r['asr_ms']:>7}ms"
              f"{r['asr_ms']/max(1,r['tok']):>7.0f}{i:>5}ms{ac:>5}ms"
              f"{r['asr_ms']+i+ac:>9}ms  {r['int']:<10} {r['txt'][:26]}")
    n = len(rows)
    print("-" * 104)
    print(f"  平均：ASR {tot_asr/n:.0f}ms   意图 {tot_int/n:.0f}ms   "
          f"动作 {tot_act/n:.0f}ms   合计 {(tot_asr+tot_int+tot_act)/n:.0f}ms")
    share = tot_asr / max(1, tot_asr + tot_int + tot_act) * 100
    print(f"  → ASR（转写）占总延迟的 {share:.1f}%"
          f"（{'转写就是瓶颈' if share > 80 else '瓶颈不在转写'}）")
    print(f"  → 平均 {sum(r['asr_ms'] for r in rows)/max(1,sum(r['tok'] for r in rows)):.0f} ms/token"
          f"（1.7B Q4_K_M 纯 CPU，14 线程）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
