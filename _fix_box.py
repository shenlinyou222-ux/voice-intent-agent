from pathlib import Path
p = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\typein.py")
t = p.read_text(encoding="utf-8")
i = t.index("def find_input_box(")
j = t.index("def focus_input(")
new = '''def find_input_box(max_nodes=12000, max_depth=30, budget_s=3.0):
    """在当前前台窗口里找【最可能是输入框】的控件，并返回它。

    ⚠️ 血债（实测）：
      ① 原来深度上限 12 -> 而 Chromium/Electron 窗口的 UIA 树实测【深 23】,
         半路就放弃 -> 永远"没找到输入框"✗（用户："打开deepseek找不到它的输入框"）
      ② 原来无条件返回第一个 EditControl -> 会先撞上「搜索会话名称」这种搜索框 ✗
    现在的策略：遍历（深度 30 / 节点 12000 / 3 秒预算）后【排序挑】：
      · 只要 EditControl（+ 支持 ValuePattern 的 TextControl）；DocumentControl 排除
        （那是聊天记录/文档，往里面打就是"被吞"）
      · 名字里带 搜索/查找/search 的降权
      · 位置越靠下越优先（消息输入框在窗口底部）
    """
    try:
        import uiautomation as auto
        hwnd = auto.GetForegroundWindow()
        root = auto.ControlFromHandle(hwnd) if hwnd else None
        if root is None:
            return None, "拿不到前台窗口"
        t0 = time.time()
        n = 0
        found = []

        def walk(node, depth=0):
            nonlocal n
            if depth > max_depth or n > max_nodes or (time.time() - t0) > budget_s:
                return
            try:
                kids = node.GetChildren()
            except Exception:
                return
            for c in kids:
                n += 1
                try:
                    ct = c.ControlTypeName
                    if ct in ("EditControl", "ComboBoxControl"):
                        found.append(c)
                    elif ct == "TextControl":
                        try:
                            if c.GetValuePattern():
                                found.append(c)
                        except Exception:
                            pass
                except Exception:
                    pass
                walk(c, depth + 1)

        walk(root)
        if not found:
            return None, f"没找到输入框（遍历 {n} 节点 / 深度上限 {max_depth}）"

        def rank(c):
            nm = (getattr(c, "Name", "") or "").lower()
            bad = 1 if any(x in nm for x in ("搜索", "查找", "search", "find")) else 0
            try:
                y = c.BoundingRectangle.top
            except Exception:
                y = 0
            return (bad, -y)          # 非搜索框优先；越靠下优先

        found.sort(key=rank)
        box = found[0]
        info = "%s「%s」候选%d个" % (box.ControlTypeName,
                                    (getattr(box, "Name", "") or "")[:24], len(found))
        return box, info
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


'''
t = t[:i] + new + t[j:]
p.write_text(t, encoding="utf-8")
print("  ✅ find_input_box 已重写（深度30/节点12000/3s + 排序挑）")
