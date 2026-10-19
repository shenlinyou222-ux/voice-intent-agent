# -*- coding: utf-8 -*-
r"""定位 Qwen3-ASR 处理器到底吃什么形式的音频"""
import sys
import traceback
import wave
from pathlib import Path

import numpy as np

D = r"E:\models\qwen3-asr-0.6b-official"
WAV = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\wavs\train_launch_app_0182322b5f.wav")

with wave.open(str(WAV), "rb") as w:
    sr = w.getframerate()
    raw = w.readframes(w.getnframes())
wav = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
print(f"wav: {len(wav)} 采样 {sr} Hz  dtype={wav.dtype} ndim={wav.ndim}")

import transformers
from transformers import AutoProcessor
print("transformers", transformers.__version__)

try:
    from transformers.audio_utils import is_valid_audio, make_list_of_audio, make_audio_chat_template_content
    print("is_valid_audio(np)      =", is_valid_audio(wav))
    print("is_valid_audio(list)    =", is_valid_audio(list(wav[:100])))
    print("make_list_of_audio ok   =", type(make_list_of_audio(wav)), len(make_list_of_audio(wav)))
except Exception:
    traceback.print_exc()

proc = AutoProcessor.from_pretrained(D)
print("processor:", type(proc).__name__)
print("audio_token:", getattr(proc, "audio_token", None),
      "id:", getattr(proc, "audio_token_id", None))

for label, kw in (("裸 numpy", dict(audio=wav)),
                  ("dict+采样率", dict(audio={"array": wav, "sampling_rate": 16000})),
                  ("本地路径str", dict(audio=str(WAV)))):
    print(f"\n---------- 试 {label} ----------")
    try:
        out = proc.apply_transcription_request(language="Chinese", return_tensors="pt", **kw)
        print("  ✅ OK  keys =", list(out.keys()))
        for k, v in out.items():
            try:
                print(f"     {k}: shape={tuple(v.shape)} dtype={v.dtype}")
            except Exception:
                print(f"     {k}: {type(v)}")
        break
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {e}")
        traceback.print_exc(limit=6)
