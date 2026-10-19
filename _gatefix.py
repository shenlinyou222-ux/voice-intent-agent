from pathlib import Path
p = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\voice_agent.py")
t = p.read_text(encoding="utf-8")
n = t.count('("own", "gguf")')
# ⚠️⚠️ 血债：这三处门控漏了 "http" —— 于是 asr_backend="http" 时
#   _make_asr() 永不被调用，self.asr 恒为 None -> 日志只显示「(无转写)」。
t = t.replace('("own", "gguf")', '("own", "gguf", "http")')
p.write_text(t, encoding="utf-8")
print(f"  替换 {n} 处门控 -> 现在包含 http")
