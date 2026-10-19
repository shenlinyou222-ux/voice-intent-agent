from pathlib import Path
v = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\voice_agent.py")
t = v.read_text(encoding="utf-8")
old = '''            _prev = _st.get("text", "")
            if _prev and not _prev.rstrip().endswith(("。", "！", "？", ".", "!", "?", "…")):'''
new = '''            _prev = _st.get("text", "")
            # ⚠️ 标点判据不可靠：模型给【每一段】都自动补句末标点
            #   （实测「它这个过程应该是。」明明没说完也有句号）→
            #   听写类只按【时间间隔】判定是否拼接；命令类仍要求"没说完"，
            #   而且【已执行动作的命令永远不会进到这里】（不记录 -> 不拼接）。
            _unfinished = bool(_prev) and (
                bool(_st.get("dictated")) or
                not _prev.rstrip().endswith(("。", "！", "？", ".", "!", "?", "…")))
            if _unfinished:'''
assert old in t, "锚点未匹配"
v.write_text(t.replace(old, new, 1), encoding="utf-8")
print("  ✅ 拼接条件已放宽（听写类按间隔）")
