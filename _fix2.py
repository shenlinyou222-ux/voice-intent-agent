import json
from pathlib import Path
root = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
# ① 恢复被我砍错的两条结构性触发词
p = root / "intent_config.json"; d = json.loads(p.read_text(encoding="utf-8")); tr = d["triggers"]
for x in ("搜文件", "找文件"):
    if x not in tr["file_search"]:
        tr["file_search"].append(x)
d["behavior"]["_comment_triggers"] = ("触发词收紧：只砍动作类/泛化词（运行/使用/搜个/查查/搜搜），"
    "结构性组合（搜文件/找文件/搜索一下…）保留——实测砍掉它们会让 file_search->site_search 误判 2->5。")
p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("  file_search =", tr["file_search"])
# ② 热词：只留【纯 ASCII】的英文专有名词
v = root / "voice_agent.py"; t = v.read_text(encoding="utf-8")
old = '''        ctx = hotword_context(
            self.cfg, extra=extra,
            max_chars=int(self.cfg.behavior.get("asr_hotword_chars", 400)),
            ascii_only=bool(self.cfg.behavior.get("asr_hotword_ascii_only", False)))
        return ctx'''
new = '''        ctx = hotword_context(
            self.cfg, extra=extra,
            max_chars=int(self.cfg.behavior.get("asr_hotword_chars", 400)),
            ascii_only=bool(self.cfg.behavior.get("asr_hotword_ascii_only", False)))
        # ⚠️ ascii_only 只判断"含 ASCII"，于是「Edge浏览器」「B站」「qwen办公」也进来了 ✗
        #   用户要求：热词 prompt 只放【少量英文专有名词】-> 这里后置过滤成纯 ASCII。
        if self.cfg.behavior.get("asr_hotword_ascii_only"):
            _parts = [x for x in (ctx or "").split("、") if x and x.isascii()]
            ctx = "、".join(_parts[:40])
            log(f"   ASR 热词（纯英文 {len(_parts[:40])} 个）：{ctx[:70]}", "ASR")
        return ctx'''
if old in t:
    v.write_text(t.replace(old, new, 1), encoding="utf-8"); print("  ✅ 热词已过滤为纯 ASCII")
else:
    print("  ⚠️ _hotwords 锚点未匹配")
