from pathlib import Path
a = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\actions.py")
t = a.read_text(encoding="utf-8")
old = """                elif len(sp) >= 3 and len(cp) >= 3 and sp[0] == cp[0] \\
                        and _lev(sp, cp) <= 1:"""
new = """                elif (len(sp) >= 3 and len(cp) >= 3
                      # ⚠️ 必须再要求【名字】>=3 字：sp/cp 是拼音长度（douban=6），
                      #   只看拼音会让「豆瓣」被吸到「豆包」（用户早就骂过这个）。
                      #   2 字中文名字只允许拼音【完全相同】（上一分支）。
                      and len(cl) >= 3 and sp[0] == cp[0]
                      and _lev(sp, cp) <= 1):"""
if old in t:
    a.write_text(t.replace(old, new, 1), encoding="utf-8"); print("  ✅ _homophone_hit 加了名字长度守卫")
else:
    print("  ⚠️ 锚点未匹配，看看实际文本:")
    import re
    for m in re.finditer(r".*len\(sp\) >= 3.*", t):
        print("    ", m.group(0).strip())
