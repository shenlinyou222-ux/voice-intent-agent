import json
from pathlib import Path
root = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
# ① 配置：用 {wsl_ip} 占位 + 开启动态解析
p = root / "intent_config.json"
d = json.loads(p.read_text(encoding="utf-8")); b = d["behavior"]
b["asr_backend"] = "http"
b["asr_http_url"] = "http://{wsl_ip}:30803/v1/audio/transcriptions"
b["asr_http_model"] = "qwen3-asr"
b["asr_http_resolve_wsl_ip"] = True
b["wsl_distro"] = "Ubuntu-24.04"
b["_comment_gpu_asr"] = ("ASR = WSL k3s 里 llama-asr（GPU, 量化 1.7B Q4_K_M, -ngl 99）。"
    "⚠️ Windows localhost 连不到 k3s NodePort（.wslconfig 默认 NAT 模式）→ "
    "必须用 WSL IP，且 WSL 重启后 IP 会变 → {wsl_ip} 占位 + 启动时 wsl hostname -I 动态解析；"
    "解析/健康检查失败自动回退进程内 CPU GGUF。")
p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("  url =", b["asr_http_url"], "| resolve =", b["asr_http_resolve_wsl_ip"])
# ② 代码：动态解析 WSL IP + 健康检查失败回退 CPU
v = root / "voice_agent.py"; t = v.read_text(encoding="utf-8")
old1 = '''            return HTTPASR(url=self.cfg.behavior.get(
                               "asr_http_url",
                               "http://127.0.0.1:8124/v1/audio/transcriptions"),'''
new1 = '''            _url = str(self.cfg.behavior.get(
                "asr_http_url", "http://127.0.0.1:8124/v1/audio/transcriptions"))
            # ⭐⭐ WSL/k3s 的 NodePort：Windows 的 localhost【连不上】
            #   （.wslconfig 没开 networkingMode=mirrored → 默认 NAT 不转发 NodePort），
            #   实测：127.0.0.1:30803 失败、172.20.71.101:30803 HTTP 200 ✓
            #   而 WSL 重启后 IP 会变 → 每次启动动态解析（wsl hostname -I 第一个地址）。
            if self.cfg.behavior.get("asr_http_resolve_wsl_ip") and "{wsl_ip}" in _url:
                try:
                    import subprocess as _sp
                    _r = _sp.run(["wsl", "-d", str(self.cfg.behavior.get(
                                      "wsl_distro", "Ubuntu-24.04")),
                                  "-e", "bash", "-c", "hostname -I"],
                                 capture_output=True, text=True, timeout=25)
                    _ip = (_r.stdout or "").split()[0]
                    _url = _url.replace("{wsl_ip}", _ip)
                    log(f"   ASR 端点：WSL IP = {_ip}", "ASR", sub=True)
                except Exception as _e:
                    log(f"   ⚠️ 解析 WSL IP 失败: {_e}", "WARN")
            _a = HTTPASR(url=_url,'''
t2 = t.replace(old1, new1, 1)
assert t2 != t, "锚点1未匹配"
old2 = "                           prompt=ctx)"
new2 = '''                           prompt=ctx)
            try:                      # 健康检查失败 → 回退进程内 CPU GGUF
                _a.load()
                return _a
            except Exception as _e:
                log(f"   ⚠️ GPU ASR 不可达（{_e}）→ 回退进程内 CPU GGUF", "WARN")
                self.asr_backend = "gguf"
                return self._make_asr()'''
t3 = t2.replace(old2, new2, 1)
assert t3 != t2, "锚点2未匹配"
v.write_text(t3, encoding="utf-8")
print("  ✅ 代码补丁已应用（动态解析 + 回退）")
