import json
from pathlib import Path
p = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\intent_config.json")
d = json.loads(p.read_text(encoding="utf-8")); b = d["behavior"]
b["asr_backend"] = "http"
b["asr_http_url"] = "http://127.0.0.1:30803/v1/audio/transcriptions"
b["asr_http_timeout"] = 120
b["asr_http_language"] = "zh"
b["asr_http_resolve_wsl_ip"] = False
b["_comment_gpu_asr"] = ("ASR = WSL k3s llama-asr (GPU, 1.7B Q4_K_M, -ngl 99) via "
                         "netsh portproxy 127.0.0.1:30803. 82ms/句 vs CPU 702ms。"
                         "三个坑：① _make_asr 门控只认 own/gguf -> http 永不生效；"
                         "② .get(k,default) 对值为 None 的键失效；③ sub=True 日志被隐藏。")
p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("  asr_backend =", b["asr_backend"], "|", b["asr_http_url"])
