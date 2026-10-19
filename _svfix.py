from pathlib import Path
p = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\ab_sensevoice.py")
t = p.read_text(encoding="utf-8")
old = """            s = rec.create_stream()
            s.accept_wave_file(str(f))
            rec.decode_stream(s)
            sv_txt = (s.result.text or "").strip()"""
new = """            s = rec.create_stream()
            # ⚠️ 1.13.8 里没有 accept_wave_file，正确方法是 accept_waveform(sr, 样本)
            s.accept_waveform(16000, wav_of(f))
            rec.decode_stream(s)
            sv_txt = (s.result.text or "").strip()"""
assert old in t, "锚点未匹配"
p.write_text(t.replace(old, new, 1), encoding="utf-8")
print("  ✅ 已改用 accept_waveform")
