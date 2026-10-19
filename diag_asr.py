# -*- coding: utf-8 -*-
"""决定性诊断：原始完整 Qwen3-ASR 对同一批音频听到什么？

我们的管线 = A(audio_tower 冻结) + 自训 ctc_head
原始模型   = A + 官方 lm_head（原封不动）

如果原始模型也听成「打开lender」，说明音频里就没有清晰的 B 音
（TTS 拼读问题）→ 不是我们 ctc_head 的锅。
"""
import json
import sys
import time
from pathlib import Path

import torch

A_DIR = Path(r"E:\models\qwen3-asr-0.6b-official")
HERE = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
HOT = HERE / "_hot_wavs"

print("=" * 84)
print("  加载原始完整 Qwen3-ASR（带官方 lm_head）")
print("=" * 84)
t0 = time.time()
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

proc = AutoProcessor.from_pretrained(str(A_DIR))
model = Qwen3ASRForConditionalGeneration.from_pretrained(
    str(A_DIR), dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()
print(f"  载入 {time.time()-t0:.0f}s   参数 {sum(p.numel() for p in model.parameters())/1e6:.1f} M")

CASES = [
    (0, "打开Blender"), (4, "启动Ollama"), (6, "启动PowerShell"),
    (8, "启动Python"), (9, "打开Steam"), (11, "打开Trae"),
    (12, "启动Clash"), (13, "打开Node"), (16, "打开GitHub"),
    (17, "打开HuggingFace"), (18, "打开ModelScope"), (19, "上CSDN"),
]

print()
print("=" * 84)
print("  原始 Qwen3-ASR 的转写")
print("=" * 84)
import soundfile as sf
results = []
with torch.no_grad():
    for idx, text in CASES:
        p = HOT / f"h{idx}.wav"
        if not p.exists():
            print(f"  [!] 缺 {p.name}")
            continue
        wav, sr = sf.read(str(p), dtype="float32")
        dur = len(wav) / sr
        inputs = proc(audio=wav, sampling_rate=sr, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
        hyp = proc.batch_decode(out, skip_special_tokens=True)[0]
        results.append((text, hyp, dur))
        print(f"  说 {text:22s} ({dur:.2f}s)")
        print(f"  原始模型听 → {hyp!r}")

print()
print("=" * 84)
print("  我们的管线听 →（对照）")
print("=" * 84)
sys.path.insert(0, str(HERE))
from model_runner import IntentModel, read_wav
m = IntentModel(model_dir=HERE / "model", device="cpu")
with torch.no_grad():
    for text, orig_hyp, dur in results:
        # 找到对应的 wav
        for idx, t in CASES:
            if t == text:
                r = m.predict(read_wav(str(HOT / f"h{idx}.wav")))
                same = "✅ 一致" if r["text"] == orig_hyp else "❌ 不一致"
                print(f"  说 {text:22s}")
                print(f"      原始 {orig_hyp!r}")
                print(f"      我们 {r['text']!r}   {same}")
                break

json.dump([{"said": t, "orig": o, "dur": d} for t, o, d in results],
          open(HERE / "asr_diag.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"\n-> {HERE/'asr_diag.json'}")
