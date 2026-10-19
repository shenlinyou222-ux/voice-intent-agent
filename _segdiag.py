import re
from pathlib import Path
p = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\run_src.log")
raw = p.read_bytes()
enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-16-le" if raw.count(b"\x00") > max(8, len(raw)//8) else "utf-8")
lines = raw.decode(enc, "ignore").splitlines()
seg, asr, drops = [], [], []
for l in lines:
    m = re.search(r"\[(\d\d:\d\d:\d\d)\]\s+MIC\s+📥 段进入 handle: ([\d.]+)s\s+VAD峰([\d.]+)\s+RMS ([\d.]+)", l)
    if m:
        seg.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)), l.strip()))
        continue
    m = re.search(r"\[(\d\d:\d\d:\d\d)\]\s+ASR\s+解「(.+?)」→(\S+)\s+(\d+)ms", l)
    if m:
        asr.append((m.group(1), m.group(2), m.group(3), int(m.group(4)), l.strip()))
    if any(k in l for k in ("🔇", "⛔", "不像人话", "当噪声丢掉", "太短")):
        drops.append(l.strip())
print("  段进入 handle: %d   转写成功: %d   门限丢弃记录: %d" % (len(seg), len(asr), len(drops)))
if seg:
    import statistics
    rs = [s[3] for s in seg]
    print("  原始段 RMS: 中位 %.5f  min %.5f  max %.5f   （噪声底≈0.0006 为界）" % (
        statistics.median(rs), min(rs), max(rs)))
    print("  原始段 RMS < 0.002（疑似只到噪声底/太轻）: %d/%d" % (sum(1 for r in rs if r < 0.002), len(rs)))
    vp = [s[2] for s in seg]
    print("  VAD 峰概率: 中位 %.3f  min %.3f" % (statistics.median(vp), min(vp)))
    print("  最近 10 个段:")
    for t, d, v, r, _ in seg[-10:]:
        print("    %s  %5.2fs  VAD峰%.3f  RMS %.5f" % (t, d, v, r))
print("  最近 6 条转写:")
for t, txt, it, ms, _ in asr[-6:]:
    print("    %s  %-30s ->%-12s %dms" % (t, txt[:30], it, ms))
if drops:
    print("  门限丢弃记录（最近 8）:")
    for d in drops[-8:]: print("    " + d[:110])
