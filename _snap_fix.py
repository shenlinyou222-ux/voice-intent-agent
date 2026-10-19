from pathlib import Path
a = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\actions.py")
t = a.read_text(encoding="utf-8")
# ① 回滚 _homophone_hit 的守卫（它误伤了 6 个搜索用例）
bad = """                elif (len(sp) >= 3 and len(cp) >= 3
                      # ⚠️ 必须再要求【名字】>=3 字：sp/cp 是拼音长度（douban=6），
                      #   只看拼音会让「豆瓣」被吸到「豆包」（用户早就骂过这个）。
                      #   2 字中文名字只允许拼音【完全相同】（上一分支）。
                      and len(cl) >= 3 and sp[0] == cp[0]
                      and _lev(sp, cp) <= 1):"""
good = """                elif len(sp) >= 3 and len(cp) >= 3 and sp[0] == cp[0] \\
                        and _lev(sp, cp) <= 1:"""
if bad in t:
    t = t.replace(bad, good, 1); print("  ✅ _homophone_hit 守卫已回滚（它误伤搜索用例）")
# ② 正主：_snap_score 里对 2 字纯中文名字要求精确（比【清理后】的参数）
old = """            for p in probes:
                d = _lev(p, cl)
                if d < bd:
                    bd, b = d, c"""
new = """            # ⚠️⚠️ 2 个纯汉字的名字只允许【精确】匹配：汉字差一个字 = 完全不同的词
            #   实测血债：「打开豆瓣」被 _snap 吸成「豆包」（距离 1 落在 2*0.5 阈值内）
            #   -> 去开了豆包。历史上同类的还有「帮我打开微信」-> 微博。
            #   注意比的是【清理后】的参数：'知乎。' 这类带标点的正确参数不能被误伤
            #   （上一版写成 p != cl，把 9/24 用例误伤成拒识）。
            _short_cjk = (len(cl) <= 2 and all("\\u4e00" <= ch <= "\\u9fff"
                                               for ch in cl))
            for p in probes:
                if _short_cjk and _clean_arg(p).lower() != cl:
                    continue
                d = _lev(p, cl)
                if d < bd:
                    bd, b = d, c"""
assert old in t, "_snap_score 锚点未匹配"
t = t.replace(old, new, 1)
a.write_text(t, encoding="utf-8"); print("  ✅ _snap_score 加了 2 字中文守卫（比清理后参数）")
