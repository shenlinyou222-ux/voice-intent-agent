import json
from pathlib import Path
root = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
p = root / "intent_config.json"
d = json.loads(p.read_text(encoding="utf-8"))
b = d["behavior"]
b["asr_backend"] = "http"
b["asr_http_url"] = "http://127.0.0.1:30803/v1/audio/transcriptions"
b["asr_http_model"] = "qwen3-asr"
b["_comment_gpu_asr"] = ("ASR 切到 WSL k3s 里的 llama-asr：GPU 上跑量化 1.7B "
                         "Q4_K_M(-ngl 99)，与 CPU 版同一份权重 -> 质量一致、速度约 8x。"
                         "端点 NodePort 30803。热词仍走 prompt 字段。")
p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("  asr_backend =", b["asr_backend"])
print("  asr_http_url =", b["asr_http_url"])
