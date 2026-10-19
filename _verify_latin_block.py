# -*- coding: utf-8 -*-
r"""验证：屏蔽拉丁 token 后，ASR 音译 + 路由命中 全链路仍工作

用本地 OwnASR + transformers LogitsProcessor 模拟 vLLM bad_words 效果
（vLLM 服务端在远程，本地无法直接测；logits_processor 机制等价）。

对 _hot_wavs 含拉丁的几条 + utt 抽样，对比：
  A. 原始转写（无屏蔽）
  B. 屏蔽拉丁后转写（音译）
  C. B 的路由命中
"""
import sys
import wave
from pathlib import Path

sys.path.insert(0, r"C:\Users\user\Desktop\ds harness\voice_agent")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from asr_own import OwnASR  # noqa: E402
from actions import Config, resolve_by_rank  # noqa: E402

HERE = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
A_DIR = r"E:\models\qwen3-asr-0.6b-official"

# 实测危险拉丁 token（_collect_latin_tokens.py 结果）+ 闭集 53 词
LATIN_TOKENS = [
    "Bl", "ender", "Steam", "Power", "Shell", "Deep", "Seek",
    "bl", "tr", "ay", "Cl", "ash", "Note", "F", "alcon",
    "Edge", "GitHub", "H", "ugging", "Face", "Model", "Scope",
    "O", "lim", "p", "ia", "Python",
    # 闭集 53 词的拉丁形态（防大小写变体绕过）
    "deepseek", "DeepSeek", "blender", "Blender", "ollama", "Ollama",
    "clash", "Clash", "github", "GitHub", "huggingface", "HuggingFace",
    "modelscope", "ModelScope", "python", "Python", "steam", "Steam",
    "powershell", "PowerShell", "trae", "Trae", "node", "Node",
    "edge", "Edge", "csdn", "CSDN", "falcon", "Falcon",
    "notebook", "Notebook", "dvc", "DVC", "harness", "Harness",
]


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(path)
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def main():
    asr = OwnASR(model_dir=A_DIR, device="cpu", dtype="float32",
                 language="Chinese", max_new_tokens=64)
    asr.load()
    tokz = asr.proc.tokenizer

    # 收集拉丁 token id
    latin_ids = set()
    for t in LATIN_TOKENS:
        ids = tokz.encode(t, add_special_tokens=False)
        latin_ids.update(ids)
    print(f"拉丁 token id 数: {len(latin_ids)}")

    from transformers import LogitsProcessor

    class BlockLatin(LogitsProcessor):
        def __call__(self, input_ids, scores):
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[:, list(latin_ids)] = True
            return scores.masked_fill(mask, float("-inf"))

    block = BlockLatin()

    def transcribe_blocked(wav):
        """绕开 OwnASR.transcribe()，直接调 model.generate 注入 logits_processor"""
        pad = (-len(wav)) % 16000
        if pad:
            wav = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])
        msgs = [{"role": "user",
                 "content": [{"type": "audio", "audio": wav}]}]
        inputs = asr.proc.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        inputs = {k: (v.to("cpu") if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        pre = tokz.encode("language Chinese<asr_text>", add_special_tokens=False)
        pre_t = torch.tensor([pre], dtype=inputs["input_ids"].dtype)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], pre_t], dim=1)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], torch.ones_like(pre_t)], dim=1)
        with torch.no_grad():
            out = asr.model.generate(**inputs, max_new_tokens=64,
                                     do_sample=False, logits_processor=[block])
        gen = out[0, int(inputs["input_ids"].shape[1]):]
        txt = asr.proc.decode(gen, return_format="transcription_only")
        if isinstance(txt, list):
            txt = txt[0] if txt else ""
        return str(txt).strip()

    # 含拉丁的 wav
    wavs = [
        (HERE / "_hot_wavs" / "h0.wav", "打开Blender"),
        (HERE / "_hot_wavs" / "h4.wav", "启动Ollama"),
        (HERE / "_hot_wavs" / "h16.wav", "打开GitHub"),
        (HERE / "_hot_wavs" / "h19.wav", "上CSDN"),
        (HERE / "_hot_wavs" / "h12.wav", "启动Clash"),
        (HERE / "_hot_wavs" / "h18.wav", "打开ModelScope"),
    ]

    CFG = Config(HERE / "intent_config.json")

    print(f"\n{'wav':<12} {'原始转写':<20} {'屏蔽后转写':<20} {'路由命中':<15}")
    print("-" * 75)
    for p, label in wavs:
        wav, _ = read_wav(p)
        # A. 原始
        txt_a = asr.transcribe(wav)
        # B. 屏蔽拉丁
        txt_b = transcribe_blocked(wav)
        # C. 路由
        it, ar, s1, _ = resolve_by_rank(txt_b, CFG)
        hit = f"{it}/{ar}" if it else "拒判"
        print(f"{p.stem:<12} {txt_a[:18]:<20} {txt_b[:18]:<20} {hit:<15}")


if __name__ == "__main__":
    main()
