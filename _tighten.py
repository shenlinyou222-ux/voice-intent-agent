import json
from pathlib import Path
p = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\intent_config.json")
d = json.loads(p.read_text(encoding="utf-8")); b = d["behavior"]; tr = d["triggers"]
# ── 热词：只要少量英文专有名词 ──
b["asr_hotword_ascii_only"] = True      # 只放 ASCII（英文专有名词）
b["asr_hotword_chars"] = 160            # 短名单，减少 prompt 干扰
b["_comment_hotword_english_only"] = ("ASR 热词 prompt 只放少量英文专有名词"
    "（DeepSeek/Ollama/Trae/Clash/Steam/Blender/Python/GitHub/HuggingFace/"
    "ModelScope/harness/dsh…），中文名不再进 prompt —— 中文名由 L2/L3 处理。")
# ── 触发词收紧：去掉"动作类/泛化"的词 ──
DROP = {"open": ["运行", "使用", "启动一下"],
        "web_search": ["搜个", "查查", "搜搜"],
        "file_search": ["搜文件", "找文件"]}
for k, vals in DROP.items():
    if k in tr:
        before = list(tr[k]); tr[k] = [x for x in tr[k] if x not in vals]
        if tr[k] != before: print("  %-12s 去掉 %s" % (k, [x for x in before if x not in tr[k]]))
# 明确移除：句中型兜底的泛化动词（保留结构明确的组合）
b["search_mid_verbs"] = ["搜索一下", "搜索", "搜一下", "查询", "检索", "查找", "查一下", "找一下"]
b["_comment_triggers"] = ("触发词收紧：去掉动作类/泛化词（运行/使用/搜个/查查/搜搜/"
    "搜文件/找文件）。句中搜索动词只保留结构明确的组合（search_mid_verbs），"
    "单字动词只允许出现在句首（re.match 锚定）。")
p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("  asr_hotword_ascii_only =", b["asr_hotword_ascii_only"], "| chars =", b["asr_hotword_chars"])
