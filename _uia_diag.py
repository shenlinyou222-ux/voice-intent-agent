import time
import uiautomation as auto
# 找 DeepSeek 窗口（按名字）
wins = []
def cb(c, d):
    try:
        if c.ControlTypeName == "WindowControl" and c.Name and "DeepSeek" in c.Name:
            wins.append(c)
    except Exception:
        pass
    return True
auto.WalkTree(auto.GetRootControl(), cb, maxDepth=2)
print("  含 DeepSeek 的顶层窗口:", [w.Name[:30] for w in wins])
fg = auto.GetForegroundWindow()
print("  当前前台窗口:", (fg.Name or "")[:40] if fg else None, "|", fg.ControlTypeName if fg else None)
target = wins[0] if wins else fg
if target is None:
    print("  没有可用窗口"); raise SystemExit
from collections import Counter
types = Counter(); t0 = time.time(); n = 0; maxd = 0
def walk(c, d):
    global n, maxd
    if n > 6000 or time.time() - t0 > 25 or d > 25:
        return
    try:
        kids = c.GetChildren()
    except Exception:
        return
    for k in kids:
        n += 1; maxd = max(maxd, d)
        try: types[k.ControlTypeName] += 1
        except Exception: pass
        walk(k, d + 1)
print("  遍历目标:", (target.Name or "")[:36], "| 节点数 =", n, "| 最大深度 =", maxd,
      "| 用时 %.1fs" % (time.time() - t0))
print("  控件类型直方图（前 12）:", types.most_common(12))
print("  有没有可编辑候选:")
for t in ("EditControl", "DocumentControl", "TextControl", "ComboBoxControl"):
    if types.get(t):
        print("    %-16s x%d" % (t, types[t]))
if not any(types.get(t) for t in ("EditControl", "DocumentControl")):
    print("    ⚠️ 完全没有 Edit/Document 类 -> 该窗口的 UIA 树里没有暴露输入框")
