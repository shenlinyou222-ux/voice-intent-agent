from pathlib import Path
p = Path("asr_http.py"); t = p.read_text(encoding="utf-8")
old = """        self.last_text = (js.get("text") or "").strip()"""
new = """        # ⭐ 解析不到 text 时把【服务端原始响应】打出来（否则静默变空转写，
        #   实测就是这样被吞掉的：agent 显示「(无转写)」而真实原因不明）
        if not (js.get("text") or "").strip():
            raise RuntimeError(f"服务端没返回 text 字段，原始响应: {raw[:400]}")
        self.last_text = (js.get("text") or "").strip()"""
assert old in t, "锚点未匹配"
p.write_text(t.replace(old, new, 1), encoding="utf-8")
print("  ✅ 已让 asr_http 在空文本时抛出服务端原始响应")
