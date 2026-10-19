from pathlib import Path
p = Path("voice_agent.py"); t = p.read_text(encoding="utf-8")
old = '''                _asr_in = raw if self.cfg.behavior.get("asr_use_raw", True) else wav
                txt_own = self.asr.transcribe(_asr_in)'''
new = '''                # ⚠️ 实测血债：文件模式下 raw 可能为 None / 与 mic 模式不同 →
                #    送给 HTTP ASR 的音频坏了 -> 服务端返回空 -> 日志只显示「(无转写)」，
                #    看不出原因（HTTPASR 单测同一文件却正常 293ms）。
                _asr_in = raw if (raw is not None and
                                  self.cfg.behavior.get("asr_use_raw", True)) else wav
                if _asr_in is None:
                    _asr_in = wav
                try:
                    _arr = np.asarray(_asr_in, dtype=np.float32).ravel()
                except Exception:
                    _arr = np.asarray(wav, dtype=np.float32).ravel()
                if _arr.size == 0:
                    _arr = np.asarray(wav, dtype=np.float32).ravel()
                try:
                    log(f"   ASR 输入 {_arr.size} 点  RMS "
                        f"{float(np.sqrt((_arr ** 2).mean())):.5f}  峰值 "
                        f"{float(np.abs(_arr).max()):.5f}  (raw={raw is not None})",
                        "ASR", sub=True)
                except Exception:
                    pass
                txt_own = self.asr.transcribe(_arr)'''
assert old in t, "锚点未匹配"
p.write_text(t.replace(old, new, 1), encoding="utf-8")
print("  ✅ 已加防御 + ASR 输入诊断")
