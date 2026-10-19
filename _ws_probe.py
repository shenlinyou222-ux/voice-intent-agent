# -*- coding: utf-8 -*-
"""vLLM realtime WS 探针 —— 一次性回答三个未知项（不用写正式客户端）。

未知项：
  ① 握手顺序：官方 docstring 写的是 commit 在 append【之前】（与 OpenAI realtime 习惯相反）
  ② 事件名/字段：session.created / session.update / input_audio_buffer.append /
     input_audio_buffer.commit / transcription.delta / transcription.done
  ③ **delta 会不会改写前缀**（这决定切分器能不能简单做）

用法（在 WSL 里用 vllm 的 venv 跑，它有 websockets）：
  /home/user/vllm-venv/bin/python _ws_probe.py <wav> [ws_url]
"""
import asyncio
import base64
import json
import sys
import wave
from pathlib import Path

import websockets

URL = sys.argv[2] if len(sys.argv) > 2 else "ws://172.20.71.101:30805/v1/realtime"
WAV = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/c/Users/user/Desktop/ds harness/voice_agent/_dual/211710_dual.wav"
SR = 16000
CHUNK_MS = 100


def load_pcm(path, seconds=None):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR, f"采样率 {w.getframerate()} != {SR}"
        assert w.getnchannels() == 1, f"声道 {w.getnchannels()} != 1"
        raw = w.readframes(w.getnframes())
    if seconds:
        raw = raw[:int(seconds * SR) * 2]
    return raw


async def recv_until(ws, tag, seconds, expect=()):
    """收 seconds 秒，把收到的事件打出来"""
    got = []
    end = asyncio.get_event_loop().time() + seconds
    while True:
        left = end - asyncio.get_event_loop().time()
        if left <= 0:
            break
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=left)
        except asyncio.TimeoutError:
            break
        try:
            d = json.loads(msg)
        except Exception:
            print(f"     [{tag}] 非 JSON: {str(msg)[:120]}")
            continue
        got.append(d)
        t = d.get("type")
        extra = ""
        if t == "transcription.delta":
            extra = f"  delta={d.get('delta')!r}"
        elif t == "transcription.done":
            extra = f"  text={d.get('text')!r}  usage={d.get('usage')}"
        elif t == "error":
            extra = f"  ❌ error={d.get('error')!r}  code={d.get('code')!r}"
        elif t == "session.created":
            extra = f"  { {k: v for k, v in d.items() if k != 'type'} }"
        print(f"     [{tag}] {t}{extra}")
    return got


async def main():
    pcm = load_pcm(WAV)
    print(f"  音频 {WAV}\n  时长 {len(pcm)/2/SR:.2f}s   目标 {URL}\n")
    async with websockets.connect(URL, max_size=None) as ws:
        print("  ── 阶段 0：连接后等 session.created ──")
        await recv_until(ws, "0", 3)

        print("\n  ── 阶段 1：session.update（协议实测：扁平 {type, model}，不是嵌套）──")
        await ws.send(json.dumps({"type": "session.update", "model": "qwen3-asr"}))
        await recv_until(ws, "1", 2)

        step = CHUNK_MS * 2 * (SR // 1000)
        n = 0
        print(f"\n  ── 阶段 2：直接 append（自然顺序），每片 {CHUNK_MS}ms ──")
        for i in range(0, min(len(pcm), SR * 2 * 6), step):     # 先送前 6 秒
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[i:i + step]).decode()}))
            n += 1
            await asyncio.sleep(0.05)
        print(f"     已 append {n} 片")
        await recv_until(ws, "2", 4)

        print("\n  ── 阶段 3：commit(final=False) 后继续 append ──")
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await recv_until(ws, "3", 2)
        for i in range(SR * 2 * 6, min(len(pcm), SR * 2 * 12), step):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[i:i + step]).decode()}))
            await asyncio.sleep(0.05)
        await recv_until(ws, "3", 4)

        print("\n  ── 阶段 4：commit(final=True) 收尾 ──")
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        await recv_until(ws, "4", 8)
        print("\n  ── 阶段 5：试 docstring 的顺序（commit 在 append 之前）──")
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await recv_until(ws, "5", 1)
        for i in range(0, min(len(pcm), SR * 2 * 3), step):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[i:i + step]).decode()}))
            await asyncio.sleep(0.05)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        await recv_until(ws, "5", 6)
    print("\n  探针结束")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
