import re
from pathlib import Path
a = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\actions.py")
t = a.read_text(encoding="utf-8")
n0 = len(re.findall(r'return "web_search"', t))
# ① 三处 return "web_search" -> 不再产出该意图（返回 unknown，交给分类器/听写）
t = re.sub(r'return "web_search", (_clean_arg\([^\n]*\))', r'return "unknown", orig.strip()  # ⛔ web_search 已废弃（用户明令删除）', t)
# ② 动作层兜底：即使别的路径产出 web_search，也不许开浏览器
old = 'def act_web_search(arg, cfg, dry=False):'
new = ('def act_web_search(arg, cfg, dry=False):\n'
       '    """⛔ 已废弃：web_search（"搜索X"→浏览器搜索）按用户明令删除。\n'
       '    保留函数只为防御：任何残留路径产出 web_search 时都不许开浏览器。"""\n'
       '    return False, "web_search 意图已删除（这类意图过时）；要站内搜索请说『在X里搜Y』", None\n\n\n'
       'def _act_web_search_disabled(arg, cfg, dry=False):')
if old in t and "意图已删除" not in t:
    t = t.replace(old, new, 1)
a.write_text(t, encoding="utf-8")
print("  ✅ 三处路由已改为 unknown；act_web_search 改为拒绝（防御）")
print("  剩余 return \"web_search\" 次数:", len(re.findall(r'return "web_search"', t)), "（原 %d）" % n0)
