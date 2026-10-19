import time
from collections import Counter
import uiautomation as auto

hwnd = auto.GetForegroundWindow()
print("  GetForegroundWindow() 返回类型:", type(hwnd).__name__, "值:", hwnd)
root = auto.ControlFromHandle(hwnd)
print("  ControlFromHandle ->", type(root).__name__, "|",
      getattr(root, "ControlTypeName", None), "|", (getattr(root, "Name", "") or "")[:40] if root else None)

hits = []
def cb(c, d):
    try:
        if c.ControlTypeName == "WindowControl" and c.Name:
            hits.append((c.Name[:34], c.NativeWindowHandle))
    except Exception:
        pass
    return True
auto.WalkTree(auto.GetRootControl(), cb, maxDepth=1)
print("  顶层窗口 %d 个:" % len(hits))
for nm, h in hits[:14]:
    print("    ", nm, hex(h) if h else "")
target = root
if target is None:
    print("  ⚠️ 拿不到前台控件")
else:
    types = Counter(); t0 = time.time(); n = 0; maxd = 0; cands = []
    def walk(c, d):
        global n, maxd
        if n > 8000 or time.time() - t0 > 30 or d > 30:
            return
        try: kids = c.GetChildren()
        except Exception: return
        for k in kids:
            n += 1; maxd = max(maxd, d)
            try:
                ct = k.ControlTypeName
                types[ct] += 1
                if ct in ("EditControl", "DocumentControl", "ComboBoxControl"):
                    cands.append((ct, (k.Name or "")[:30], bool(getattr(k, "IsEnabled", True)),
                                  bool(getattr(k, "IsReadOnly", False))))
            except Exception: pass
            walk(k, d + 1)
    walk(target, 0)
    print("  目标窗口:", (getattr(target, 'Name', '') or '')[:36], "| 节点 %d | 深度 %d | %.1fs" % (n, maxd, time.time()-t0))
    print("  类型直方图:", types.most_common(10))
    print("  可编辑候选 %d 个:" % len(cands))
    for ct, nm, en, ro in cands[:8]:
        print("    %-16s enabled=%s readonly=%s 「%s」" % (ct, en, ro, nm))
