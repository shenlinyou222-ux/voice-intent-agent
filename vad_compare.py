# -*- coding: utf-8 -*-
r"""Silero VAD（神经网络，业界标准） vs 我们现在的手写能量阈值

要回答的是：环境噪声 vs 人声，到底是不是"难"问题。
用四类输入各测一遍，看 Silero 的概率输出是否干净可分：
  ① 正常电平的 TTS 人声
  ② **缩放到本机麦克风实际电平的人声**（帧 RMS ~0.0015）← 关键：它扛不扛低电平
  ③ 本机噪声底电平的纯噪声（帧 RMS ~0.0005）
  ④ 数字静音（精确 0，本机静音时就是这样）

用法: E:\py312\python.exe vad_compare.py
"""
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
MODELS = [Path(r"C:\Users\user\Desktop\ds harness\open-llm-vtuber\frontend\libs\silero_vad_v5.onnx"),
          Path(r"C:\Users\user\Desktop\ds harness\minimind-o\model\vad\silero_vad.onnx")]


def load_wav(p):
    with wave.open(str(p), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return a, sr


class Silero:
    """Silero VAD v5 onnx：16kHz / 512 样本一块（32ms），带 LSTM state"""

    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self.s = ort.InferenceSession(str(path), sess_options=so,
                                      providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.sr = np.array(16000, dtype=np.int64)

    def probs(self, x):
        """把整段音频切成 512 样本喂进去，返回每块的语音概率"""
        out = []
        self.reset()
        for i in range(0, len(x) - 511, 512):
            chunk = x[i:i + 512].reshape(1, -1).astype(np.float32)
            p, self.state = self.s.run(None, {"input": chunk, "state": self.state,
                                              "sr": self.sr})
            out.append(float(p[0][0]))
        return np.array(out)


def stats(tag, x):
    if len(x) == 0:
        print(f"  {tag:38s} （无数据）")
        return
    frames = x[:len(x) // 480 * 480].reshape(-1, 480)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    print(f"  {tag:38s} 帧RMS 中位 {np.median(rms):.5f}  峰值 {rms.max():.5f}"
          f"  峰值帧 {rms.max():.5f}")


speech, sr = load_wav(sorted((HERE / "wavs").glob("train_exact*.wav"))[0])
rng = np.random.default_rng(0)
noise_mic = rng.normal(0, 0.0005, 16000 * 3).astype(np.float32)
zeros = np.zeros(16000 * 3, dtype=np.float32)
speech_norm = speech / (np.sqrt((speech ** 2).mean()) + 1e-12) * 0.0015   # 本机实际电平

cases = [("① TTS人声 正常电平", speech),
         ("② 人声 缩放到本机电平", speech_norm),
         ("③ 纯噪声 本机噪声底电平", noise_mic),
         ("④ 数字静音（精确0）", zeros)]

for mp in MODELS:
    if not mp.exists():
        continue
    print(f"\n{'='*78}\n  {mp.name}   ({mp.stat().st_size/1e6:.1f} MB)\n{'='*78}")
    v = Silero(mp)
    for tag, x in cases:
        p = v.probs(x)
        if len(p) == 0:
            print(f"  {tag:38s} 太短")
            continue
        stats(tag, x)
        print(f"  {'':38s} Silero 概率: 中位 {np.median(p):.3f}  最大 {p.max():.3f}"
              f"  >0.5 占比 {100*(p>0.5).mean():5.1f}%   → "
              f"{'判为语音' if (p>0.5).mean() > 0.3 else '判为静音/噪声'}")
