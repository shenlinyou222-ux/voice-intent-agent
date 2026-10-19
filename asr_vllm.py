# -*- coding: utf-8 -*-
"""vLLM 后端（GPU，Qwen3-ASR）—— 集群里 vllm-asr:30805

⚠️ 关键：vLLM **没有** OpenAI 的 /v1/audio/transcriptions 端点 ✗
   （85-vllm-asr.yaml 里明确警告："这与 llama.cpp 的 -F file=@x.wav 完全不同 —— 调用方要改"）
   ASR 走 chat/completions 的多模态输入：
     {"model":"qwen3-asr","messages":[{"role":"user","content":[
        {"type":"input_audio","input_audio":{"data":"<base64 wav>","format":"wav"}}]}]}

实测（0.6B，GPU + CUDA Graph）：1.05s 音频 → ~105ms（稳态）✓
对比：CPU 进程内 llama.cpp 1.7B = 430ms；transformers GPU 逐token = 227ms（无图加速）
"""
from __future__ import annotations

import base64
import json
import os
import re
import struct
import time
import urllib.request

import numpy as np

DEFAULT_URL = "http://127.0.0.1:30805/v1/chat/completions"
DEFAULT_MODEL = "qwen3-asr"

_PREFIX = re.compile(r"language\s+[A-Za-z]+(?:<asr_text>)?", re.I)


def pcm_to_wav_b64(pcm, sr: int = 16000) -> str:
    a = np.clip(np.asarray(pcm, dtype=np.float32).ravel(), -1.0, 1.0)
    b = (a * 32767).astype("<i2").tobytes()
    wav = (b"RIFF" + struct.pack("<I", 36 + len(b)) + b"WAVEfmt " +
           struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) +
           b"data" + struct.pack("<I", len(b)) + b)
    return base64.b64encode(wav).decode()


class VLLMASR:
    """进程外 GPU 转写（HTTP，chat/completions）。接口与 GGUFASR 一致。

    ★★ 热词偏置 = 给热词 token 的 logits 【加分】（logit_bias），不是硬约束 ✓
        · 软偏置：模型【更可能】吐热词，但【仍然可以】吐别的 → 拒识逻辑照旧有效 ✓
        · 硬约束（structured_outputs）会把每句话都变成命令 ✗，不要用
        · 实测（12s 闲聊长句）：α≤8 完全正常 ✓；α=12 开始污染（「乱七八九」✗）
          → 取 α=6 ✓
        · token id 由服务端 /tokenize 现取（「三农」是单 token 113091 ✓）

    ⚠️ vLLM 0.30 删掉了 guided_regex/guided_choice ✗，新 API 是 structured_outputs
       （我们不用它 —— 那是硬约束 ✓ 不符合"只提升概率"的需求）
    """

    # ★★ 热词偏置已【停用】（用户最终决定：去除热词偏置，只保留谐音匹配）
    #    实测依据（42 条中英混合，α=6）：
    #      不加热词      完全一致 42.9%  拉丁保住 56.0%
    #      只偏首 token  完全一致 42.9%  拉丁保住 46.4%  ← 比不加热词还差 9.6 点
    #      偏置全部 token 完全一致 52.4%  拉丁保住 56.0%
    #    但即使修好（偏置全部 token），相对不加热词也只是【持平拉丁保住、+9.5 点完全一致】，
    #    而偏置会把模型往热词上推，反而【压低其他词的命中】—— 与"保证不漏"的目标冲突。
    #    → 最终方案：ASR 不做任何偏置，召回全部交给谐音匹配层（phon_match.py）。
    DEFAULT_HOTWORDS: list = []
    HOTWORD_ENABLED = (os.environ.get("OPENCODE_ASR_BIAS") == "1")   # [OPENCODE-H] 默认 False(原样保留); set OPENCODE_ASR_BIAS=1 即用配置里的 hotwords/bias

    # ★★ token 层屏蔽默认表 = 繁体字形 + 拉丁碎片（用户要求"只输出简体中文" ✓）
    #   繁体 → 强制简体；拉丁 → 模型只能音译（"DeepSeek"→"迪普西克"），
    #   实测真机 ASR 吐过的拉丁 token 仅 27 个（_collect_latin_tokens.py），
    #   全是拆分碎片（Bl+ender / Deep+Seek / H+ugging+Face），屏蔽后拼音口径统一
    DEFAULT_BAD_WORDS = ["農", "濃", "龍", "儂", "膿", "籠", "隴", "壟", "聾", "噥",
                         "兩", "補", "參", "壹", "貳", "肆", "伍", "陸", "柒", "捌",
                         "玖", "拾", "實", "醫", "爾", "兒", "農民",
                         # ── 拉丁碎片（实测危险子集，2026-10-18）──
                         "Bl", "ender", "Steam", "Power", "Shell", "Deep", "Seek",
                         "bl", "tr", "ay", "Cl", "ash", "Note", "F", "alcon",
                         "Edge", "GitHub", "H", "ugging", "Face", "Model", "Scope",
                         "O", "lim", "p", "ia", "Python"]

    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL,
                 max_tokens: int = 64, timeout: float = 30.0,
                 hotwords=None, hotword_bias: float = 0.0,
                 language: str = "Chinese", bad_words=None):
        self.url = url
        self.model = model
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        # ★ 热词偏置停用：即使传了 hotwords，也只在 HOTWORD_ENABLED 为真时才生效
        self.hotwords = list(hotwords) if (hotwords and self.HOTWORD_ENABLED) else []
        self.hotword_bias = float(hotword_bias or 0.0) if self.HOTWORD_ENABLED else 0.0
        self.language = str(language or "Chinese")
        # ★ 官方强制语言前缀（模型就是这样训练的 ✓）
        self.lang_prefix = f"language {self.language}<asr_text>"
        # ★★ token 层屏蔽（用户指定：要在【token 层】屏蔽，不靠闭集 ✗）
        #   模型在 token 层面就吐不出这些字 ✓✓
        #   默认 = 繁体字形 + 拉丁碎片（强制简体+音译，只输出汉字）；
        #   可由配置 asr_vllm_bad_words 覆盖 ✓
        self.bad_words = list(bad_words) if bad_words else self.DEFAULT_BAD_WORDS
        self.logit_bias = {}          # {token_id(str): α}
        self.last_ms = 0.0
        self.last_tokens = 0
        self.loaded = True            # 服务端常驻

    def _base_url(self) -> str:
        return self.url.split("/v1/")[0]

    def load(self) -> None:
        """把热词转成 token id（用服务端 /tokenize），构建 logit_bias。"""
        if not self.hotword_bias or not self.hotwords or self.logit_bias:
            return
        ids = set()
        for w in self.hotwords:
            try:
                req = urllib.request.Request(
                    self._base_url() + "/tokenize",
                    data=json.dumps({"model": self.model, "prompt": w}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    d = json.loads(r.read())
                toks = d.get("tokens") or d.get("token_ids") or []
                # ★ 偏置【全部】token，不是只偏第一个
                #   实测（42 条中英混合，α=6）：
                #     只偏首 token → 完全一致 42.9%  拉丁保住 46.4%（比不加热词还差！）
                #     偏置全部    → 完全一致 52.4%  拉丁保住 56.0%
                #   原因：deepseek / huggingface / hackernews 都是 2~3 个 token，
                #        只偏第一个 → 模型被推着"开始拼"但后面没偏置 → 拼不完整或放弃
                for t in toks:
                    ids.add(int(t))
            except Exception:
                pass
        self.logit_bias = {str(i): self.hotword_bias for i in sorted(ids)}

    def transcribe(self, pcm) -> str:
        self.load()
        # ★★ 强制简体中文（用户要求"只输出简体中文"）三手：
        #   ① system 消息给语言提示（官方训练格式：language <NAME>）✓
        #   ② bad_words 屏蔽【繁体字形 + 拉丁碎片】→ 模型在 token 层面就吐不出
        #      繁体和拉丁（拉丁只能音译，如 "DeepSeek"→"迪普西克"）✓✓
        #   ③ 闭集对齐层再做一次 繁→简 归一（最终保险 ✓）
        #   ⚠️ continue_final_message 用不了 ✗：模型的 chat 模板会丢掉 assistant
        #      的内容（服务端报错 "final message does not appear in the chat"）
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": f"language {self.language}"},
                {"role": "user", "content": [
                    {"type": "input_audio",
                     "input_audio": {"data": pcm_to_wav_b64(pcm), "format": "wav"}}]},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        if self.bad_words:
            payload["bad_words"] = self.bad_words      # ★ 屏蔽繁体+拉丁 ✓
        if self.logit_bias:
            payload["logit_bias"] = self.logit_bias    # ★ 软偏置：提升热词概率
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.url, data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.loads(r.read())
        self.last_ms = (time.perf_counter() - t0) * 1000
        txt = d["choices"][0]["message"]["content"] or ""
        self.last_tokens = int(d.get("usage", {}).get("completion_tokens", 0) or 0)
        return _PREFIX.sub("", txt).replace("<asr_text>", "").strip()


if __name__ == "__main__":
    import sys
    import wave

    wav = sys.argv[1] if len(sys.argv) > 1 else r"E:\tmp\sannong_16k.wav"
    with wave.open(wav, "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    eng = VLLMASR()
    print(f"   音频 {len(a)/16000:.2f}s  端点 {eng.url}")
    for i in range(4):
        t = eng.transcribe(a)
        print(f"   第{i+1}次 → {eng.last_ms:6.0f}ms  {eng.last_tokens}tok  「{t}」")
