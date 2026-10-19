from pathlib import Path
root = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
v = root / "voice_agent.py"; t = v.read_text(encoding="utf-8")

NEW = '''        if self.asr_backend == "http":
            # ⭐ ASR 走 HTTP：WSL k3s 里的 llama-asr（GPU 上跑量化 1.7B Q4_K_M, -ngl 99）
            #   实测 82ms/句（CPU 版 702ms），文本与 CPU 逐字一致。
            #   日志【不带 sub=True】—— 之前 sub 日志在普通模式被隐藏，害我查了很久。
            from asr_http import HTTPASR
            ctx = self._hotwords()
            _url = str(self.cfg.behavior.get("asr_http_url")
                       or "http://127.0.0.1:30803/v1/audio/transcriptions")
            if self.cfg.behavior.get("asr_http_resolve_wsl_ip") and "{wsl_ip}" in _url:
                try:
                    import subprocess as _sp
                    _r = _sp.run(["wsl", "-d", str(self.cfg.behavior.get("wsl_distro")
                                                   or "Ubuntu-24.04"),
                                  "-e", "bash", "-c", "hostname -I"],
                                 capture_output=True, text=True, timeout=25)
                    _ip = (_r.stdout or "").split()[0]
                    _url = _url.replace("{wsl_ip}", _ip)
                    log(f"   ASR 端点：WSL IP = {_ip}", "ASR")
                except Exception as _e:
                    log(f"   ⚠️ 解析 WSL IP 失败: {type(_e).__name__}: {_e}", "WARN")
            _a = HTTPASR(url=_url,
                         timeout=float(self.cfg.behavior.get("asr_http_timeout") or 120),
                         language=str(self.cfg.behavior.get("asr_http_language") or "zh"),
                         prompt=ctx)
            log(f"   ASR(http) 端点 {_url}  热词 {len(ctx or '')} 字", "ASR")
            try:
                _a.load()
                log(f"   ASR(http) 健康检查 OK（{_a.load_ms:.0f}ms）", "ASR")
                self.asr = _a
                return _a
            except Exception as _e:
                log(f"   ⚠️ GPU ASR 不可达（{type(_e).__name__}: {_e}）"
                    f"→ 回退进程内 CPU GGUF", "WARN")
                self.asr_backend = "gguf"
                return self._make_asr()'''

i = t.index('        if self.asr_backend == "http":')
tail = '                return self._make_asr()'
j = t.index(tail, i) + len(tail)
t = t[:i] + NEW + t[j:]

# 调用前打出 ASR 类型（可见）
old = "                txt_own = self.asr.transcribe(_arr)"
new = ("                log(f\"   ASR 类型={type(self.asr).__name__} \"\n"
       "                    f\"输入 {_arr.size} 点\", \"ASR\")\n"
       "                txt_own = self.asr.transcribe(_arr)")
assert old in t, "transcribe 锚点未匹配"
t = t.replace(old, new, 1)
v.write_text(t, encoding="utf-8")
print("  ✅ http 分支已干净重写 + 类型诊断已加")
