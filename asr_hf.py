# -*- coding: utf-8 -*-
"""进程内 GPU ASR 后端（Qwen3-ASR + transformers + device_map="cuda"）

为什么需要它（用户的判断，完全正确）：
  · 之前走 HTTP（llama-server:8124）是【多此一举】✗ —— 实测 agent 侧 408ms，
    而服务端纯模型只要 160~290ms → **HTTP 编码/上传/解析白吃 ~120~250ms** ✗
  · 而且那个 llama-server 是【纯 CPU 构建】（目录里没有 ggml-cuda*.dll ✗），
    `-ngl 99` 被忽略 ✗ —— 它压根没用 GPU

做法：直接在本进程用 transformers 把模型加载到 CUDA（框架里已证明可行 ✓）：
    AutoProcessor.from_pretrained(path)
    AutoModelForMultimodalLM.from_pretrained(path, dtype="auto", device_map="cuda")
    proc.apply_transcription_request(audio=pcm_float32_16k, language="zh")
    model.generate(**inputs, max_new_tokens=128)
    proc.decode(..., return_format="transcription_only")

接口与 asr_gguf.GGUFASR 一致：load() / transcribe(np.float32 16k) / last_ms
"""
from __future__ import annotations

import time

import numpy as np


class HFASR:
    """进程内 GPU 转写。懒加载：第一次 transcribe 才装模型。"""

    def __init__(self, model_path: str, language: str = "zh", max_new_tokens: int = 128,
                 device: str = "cuda"):
        self.model_path = model_path
        self.language = language
        self.max_new_tokens = int(max_new_tokens)
        self.device = device
        self.last_ms = 0.0
        self.last_tokens = 0
        self._proc = None
        self._model = None

    # ── 加载 ──
    def load(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoProcessor, AutoModelForMultimodalLM

        t0 = time.time()
        self._proc = AutoProcessor.from_pretrained(self.model_path)
        # ★ 不用 device_map（那需要 accelerate ✗，agent 环境里没装）——
        #   先按默认加载再手动 .to("cuda")，效果一样且零依赖 ✓
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.model_path, dtype="auto")
        self._model = self._model.to(self.device)
        self._model.eval()
        self.load_ms = (time.time() - t0) * 1000

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        self._proc = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ── 推理 ──
    def transcribe(self, pcm) -> str:
        """pcm: np.float32 单声道 16kHz（[-1,1]）。返回纯转写文本。"""
        self.load()
        import torch

        arr = np.asarray(pcm, dtype=np.float32).ravel()
        t0 = time.perf_counter()
        inputs = self._proc.apply_transcription_request(
            audio=arr, language=self.language).to(self._model.device, self._model.dtype)
        with torch.inference_mode():
            ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = str(self._proc.decode(ids[:, inputs["input_ids"].shape[1]:],
                                     return_format="transcription_only")[0]).strip()
        self.last_ms = (time.perf_counter() - t0) * 1000
        self.last_tokens = int(ids.shape[1] - inputs["input_ids"].shape[1])
        return text


if __name__ == "__main__":
    import sys
    import wave

    wav = sys.argv[1] if len(sys.argv) > 1 else r"E:\tmp\sannong_16k.wav"
    model = sys.argv[2] if len(sys.argv) > 2 else \
        r"C:\Users\user\Desktop\ds harness\aigc_demo_framework\models\qwen3-asr-0.6b"
    with wave.open(wav, "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    print(f"   音频 {len(a)/16000:.2f}s  峰值 {np.abs(a).max():.5f}")
    eng = HFASR(model)
    print(f"   加载模型 {model} ...")
    eng.load()
    print(f"   加载耗时 {getattr(eng, 'load_ms', 0):.0f}ms  device={eng._model.device}")
    for i in range(3):
        t = eng.transcribe(a)
        print(f"   第{i+1}次 → {eng.last_ms:.0f}ms  {eng.last_tokens}tok  「{t}」")
