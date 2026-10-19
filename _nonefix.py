import json
from pathlib import Path
root = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
v = root / "voice_agent.py"; t = v.read_text(encoding="utf-8")
# ⚠️⚠️ 血债：.get(k, default) 只在【键不存在】时用默认值；键存在但值为 None 就中招
#   asr_http_timeout=None -> float(None) -> TypeError
#   asr_http_language=None -> str(None) == "None" -> 非法语言 -> 服务端返回空文本
reps = [
    ('timeout=float(self.cfg.behavior.get(\n                               "asr_http_timeout", 120)),',
     'timeout=float(self.cfg.behavior.get("asr_http_timeout") or 120),'),
    ('language=str(self.cfg.behavior.get("asr_http_language",\n                                                              "zh")),',
     'language=str(self.cfg.behavior.get("asr_http_language") or "zh"),'),
]
n = 0
for old, new in reps:
    if old in t:
        t = t.replace(old, new, 1); n += 1
v.write_text(t, encoding="utf-8")
print(f"  代码修复 {n}/2 处")
p = root / "intent_config.json"; d = json.loads(p.read_text(encoding="utf-8")); b = d["behavior"]
for k in ("asr_http_language", "asr_http_timeout"):
    if b.get(k, "missing") is None:
        del b[k]; print(f"  删除配置里的 {k}=None")
b["asr_http_timeout"] = 120
b["asr_http_language"] = "zh"
p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("  config ->", b["asr_backend"], b["asr_http_url"], b["asr_http_language"], b["asr_http_timeout"])
