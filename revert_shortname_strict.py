# -*- coding: utf-8 -*-
"""回滚 2 字中文名的严格匹配（净亏损：修掉一个你不会说的豆瓣，弄坏 9 个真实用例）。

保留结论记录：正确做法是①比"清理后的参数"（否则 '知乎。' 这类正确参数误伤）
且②覆盖 resolve() 的别名路径（'doubao' 别名同样会被吸）。这次先回到已知全绿状态。
"""
from pathlib import Path

P = Path(__file__).parent / "actions.py"
t = P.read_text(encoding="utf-8")
orig = t

# ① 移除 _snap_score 里的 2 字中文严格守卫
g1 = '''            # ⭐⭐ 2 个纯汉字的名字只允许【精确】匹配：汉字差一个字 = 完全不同的词
            #   实测血债：「打开豆瓣」被 snap 成「豆包」（距离 1，落在 2×0.5=1 的阈值内）
            #   → 去开了豆包。历史上同类的还有「帮我打开微信」→ 微博。
            _short_cjk = (len(cl) <= 2 and all("\\u4e00" <= ch <= "\\u9fff"
                                               for ch in cl))
            for p in probes:
                if _short_cjk and p != cl:
                    continue
                d = _lev(p, cl)
                if d < bd:
                    bd, b = d, c
        return b, bd'''
g1_new = '''            for p in probes:
                d = _lev(p, cl)
                if d < bd:
                    bd, b = d, c
        return b, bd'''
assert g1 in t, "守卫①的原文没找到（可能已被改）"
t = t.replace(g1, g1_new)

# ② 移除 _homophone_hit 里的 len(cl)>=3 条件
g2 = '''                elif (len(sp) >= 3 and len(cp) >= 3 and sp[0] == cp[0]
                      # ⭐ 2 个汉字的名字只认【拼音完全相同】：豆瓣(douban) 与
                      #   豆包(doubao) 差 1 个字母，但它们是两个不同的词 ——
                      #   近音放宽只对 ≥3 字的名字开放（记事板→记事本 仍生效）
                      and len(cl) >= 3
                      and _lev(sp, cp) <= 1):
                    ok = True'''
g2_new = '''                elif len(sp) >= 3 and len(cp) >= 3 and sp[0] == cp[0] \\
                        and _lev(sp, cp) <= 1:
                    ok = True'''
assert g2 in t, "守卫②的原文没找到（可能已被改）"
t = t.replace(g2, g2_new)

if t != orig:
    P.write_text(t, encoding="utf-8")
    print("已回滚两处守卫 ✓")
else:
    print("没有变化（可能已回滚过）")
