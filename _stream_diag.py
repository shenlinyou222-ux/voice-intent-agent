# -*- coding: utf-8 -*-
"""AgentStream 最小诊断：把 append / 自动 commit / delta 三件事分开看。

用法：python -X utf8 _stream_diag.py
"""
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from agent_stream import AgentStream, pcm16  # noqa: E402

SR = 16000
WAV = HERE / "_dual" / "211710_dual.wav"
with wave.open(str(WAV), "rb") as w:
    x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
url = "ws://172.20.71.101:30805/v1/realtime"

got = []
s = AgentStream(url, model="qwen3-asr", commit_ms=500,
                on_text=lambda t, fin: got.append((t, fin)),
                on_log=lambda m: print(f"   {m}", flush=True)).start()
for _ in range(60):
    if s.state == "ready":
        break
    time.sleep(0.1)
print(f"  state={s.state}  last_err={s.last_err!r}")

print("\n  ── 送 1 秒音频，然后等 2 秒（自动 commit 应该在这期间触发）──")
for i in range(0, SR, 480):
    s.push(pcm16(x[i:i + 480]))
time.sleep(2.0)
print(f"   n_audio_sent={s.n_audio_sent}  n_commit={s.n_commit}  "
      f"n_delta={s.n_delta}  收到 {len(got)} 条文本")

print("\n  ── 再送 2 秒，并手动 commit ──")
for i in range(SR, SR * 3, 480):
    s.push(pcm16(x[i:i + 480]))
time.sleep(0.3)
s.commit()
for _ in range(50):
    time.sleep(0.1)
    if s.n_delta:
        break
print(f"   n_audio_sent={s.n_audio_sent}  n_commit={s.n_commit}  "
      f"n_delta={s.n_delta}  收到 {len(got)} 条文本  last_err={s.last_err!r}")
print(f"   文本拼接 = {''.join(t for t, _ in got)[:80]!r}")

print("\n  ── final commit + 收尾 ──")
s.commit(final=True)
for _ in range(80):
    time.sleep(0.1)
    if got and got[-1][1]:
        break
print(f"   n_commit={s.n_commit}  n_delta={s.n_delta}  收到 {len(got)} 条")
print(f"   全文 = {''.join(t for t, _ in got)[:160]!r}")
s.stop()
print(f"   最终 state={s.state}  last_err={s.last_err!r}")
