from pathlib import Path
v = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\voice_agent.py")
t = v.read_text(encoding="utf-8")
old = 'f"VAD概率 {getattr(self.seg, \'last_prob\', 0):.3f}  {bar}",'
new = ('# ⚠️ 这个数字是【采样瞬间】的概率（多数时刻落在静音上 -> 显示 0.000）\n'
       '            #   真正判断"有没有听到人声"要看 ASR 那行的【VAD峰概率】（本段峰值）。\n'
       '            #   标签改清楚，免得再被它误导（我自己就被误导过一轮）。\n'
       '            f"VAD瞬时 {getattr(self.seg, \'last_prob\', 0):.3f}  {bar}",')
if old in t:
    v.write_text(t.replace(old, new, 1), encoding="utf-8"); print("  ✅ 表头已改为 VAD瞬时（不再冒充整体判断）")
else:
    print("  ⚠️ 锚点未匹配，打印实际行：")
    for i, l in enumerate(t.splitlines()):
        if "VAD概率" in l: print("    %d: %s" % (i+1, l.strip()))
