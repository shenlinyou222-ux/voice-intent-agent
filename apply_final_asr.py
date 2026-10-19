# -*- coding: utf-8 -*-
"""最终配置：ASR 用 1.7B 在本进程 CPU 上跑（替换原模型）；送 ASR 的音频不做 AGC；热词保留。
不碰 app 的 8124 服务（用户明确：不要停）。"""
import json
from pathlib import Path

P = Path(__file__).parent / "intent_config.json"
d = json.loads(P.read_text(encoding="utf-8"))
b = d["behavior"]

b["asr_backend"] = "gguf"          # 进程内 llama.cpp，CPU（不再打 8124/8019）
b["asr_n_ctx"] = 4096              # ⭐ 对标 app 的 llama-server -c 4096（我原来 512）
b["asr_threads"] = 8               # 1.7B 比 0.6B 重，给 8 线程
b["asr_use_raw"] = True            # ⭐ 不要 AGC：送 ASR 的是 AGC 之前的原始音频
b["asr_hotwords"] = True           # ⭐ 热词保留
b["_comment_asr_final"] = (
    "最终形态（用户指令）：ASR = Qwen3-ASR-1.7B-Q4_K_M 在【本进程 CPU】跑"
    "（asr_backend=gguf，替换掉原来的模型）；参数对标 app 的 llama-server"
    "（-c 4096）；送 ASR 的音频【不做 AGC】（asr_use_raw=true，AGC 只留给 VAD）；"
    "热词上下文保留（asr_hotwords=true）。app 的 8124 服务保持不动。")

P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("配置完成：")
for k in ("asr_backend", "asr_n_ctx", "asr_threads", "asr_use_raw", "asr_hotwords",
          "asr_gguf", "asr_mmproj"):
    v = b.get(k)
    print(f"  {k:16s} = {Path(v).name if isinstance(v, str) and '\\\\' in v else v}")
