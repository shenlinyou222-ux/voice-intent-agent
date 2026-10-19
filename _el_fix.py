from pathlib import Path
p = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\entity_link.py")
t = p.read_text(encoding="utf-8")
old = '''        if s < 0.82 and ap and kp and len(kp) >= 3 and ap[0] == kp[0] and _lev(ap, kp) <= 1:
            s, how = 0.82, "pinyin~1"             # ③ 拼音差 1'''
new = '''        # ③ 拼音差 1：⚠️ 必须同时要求【名字】>=3 字 —— 之前判断的是拼音长度
        #    （douban=6），于是「豆瓣」又被吸到「豆包」✗（用户早就骂过这个）。
        #    2 字中文名字只允许拼音【完全相同】（②）。
        if (s < 0.82 and ap and kp and len(k) >= 3 and ap[0] == kp[0]
                and _lev(ap, kp) <= 1):
            s, how = 0.82, "pinyin~1"'''
assert old in t, "拼音锚点未匹配"
t = t.replace(old, new, 1)
old2 = '''            if kn and len(k) >= 3 and _lev(an, kn) <= 1:
                s, how = 0.80, "lev1"
            elif not short_cjk and kn and _lev(an, kn) == 0:
                s, how = 0.90, "exact-norm"'''
new2 = '''            if not kn:
                pass
            # ④a 距离 1 且【两边都 >=3 字】——真正的近似（记事板->记事本）
            elif len(an) >= 3 and len(kn) >= 3 and _lev(an, kn) <= 1:
                s, how = 0.80, "lev1"
            # ④b 输入更长（多出前后缀字）：知乎网->知乎、记事本吧->记事本 —— 安全 ✓
            elif len(an) > len(kn) >= 2 and _lev(an, kn) <= 1:
                s, how = 0.78, "lev1-suffix"
            # ④c 归一化后完全相同（标点/空格差异）
            elif not short_cjk and _lev(an, kn) == 0:
                s, how = 0.90, "exact-norm"'''
assert old2 in t, "编辑距离锚点未匹配"
t = t.replace(old2, new2, 1)
p.write_text(t, encoding="utf-8")
print("  ✅ 两条规则已修")
