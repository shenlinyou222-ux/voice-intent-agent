from pathlib import Path
a = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\actions.py")
t = a.read_text(encoding="utf-8")
if 'triggers.get("dictation_off"' in t:
    print("  已有，跳过")
else:
    anchor = '    # ⭐ 清空输入框（无参数命令：「清空输入」「清空输入框」）'
    blk = '''    # ⭐ 听写模式命令（无参数）：「开始输入」/「终止输入」
    #   ⚠️⚠️ 血债：之前只在配置里加了触发词、也加了 ACTIONS 和规则直通白名单，
    #   但【没接进 parse】-> parse 恒返回 unknown -> 规则直通永不触发 ->
    #   这两条命令 100% 依赖句向量分类器 ✗。后果：分类器一旦判错（或用户说了
    #   没进示例的变体），在强制听写模式下【命令本身会被打进输入框】✗。
    for _k in ("dictation_on", "dictation_off"):
        for _trig in sorted(cfg.triggers.get(_k, []), key=len, reverse=True):
            if _trig in t:
                return _k, ""
''' + anchor
    assert anchor in t, "clear_input 锚点未匹配"
    t = t.replace(anchor, blk, 1)
    a.write_text(t, encoding="utf-8"); print("  ✅ 已接进 parse（在 clear_input 之前）")
