from pathlib import Path
a = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\actions.py")
t = a.read_text(encoding="utf-8")
if "听写模式命令【提前】" in t:
    print("  已有，跳过")
else:
    anchor = "    # ① 精确查找（最高优先，必须含\"精确\"）"
    if anchor not in t:
        anchor = "    # ① 精确查找"
        assert anchor in t, "找不到 ① 精确查找 锚点"
    blk = '''    # ⭐⭐ 听写模式命令【提前】到最前（无歧义的完整短语）
    #   ⚠️ 血债：原来放在 clear_input 之前（靠后），结果被 open 分支里的
    #   「4d 光杆开动词」 `re.match(r"(开|用|上|去|进)(?=.{2,})", t)` 先接走 ——
    #   「**开**始输入」「**开**始听写」「**开**始打字」全都变成 unknown ✗
    #   （同理「用/上/去/进」开头的听写变体也会中招）。提到最前即可。
    for _k in ("dictation_off", "dictation_on"):
        for _trig in sorted(cfg.triggers.get(_k, []), key=len, reverse=True):
            if _trig in t:
                return _k, ""

''' + anchor
    t = t.replace(anchor, blk, 1)
    a.write_text(t, encoding="utf-8"); print("  ✅ 已提前到 _parse_core 最前")
