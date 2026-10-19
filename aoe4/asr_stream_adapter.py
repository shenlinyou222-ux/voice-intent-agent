# -*- coding: utf-8 -*-
"""SSE → WS 适配器：让"本机 llama.cpp 的 HTTP-SSE 流式 ASR"冒充
vLLM 的 `ws://…/v1/realtime`，从而**原样复用**已有的流式抢跑链路：

    agent_stream.AgentStream（WS 客户端，协议见 STREAMING_ENGINE_NOTES.md）
      + stream_consume.StreamConsumer（A 路增量消费者：段去重 + 只放行可撤销动作）

为什么需要它：
  · 本机 `127.0.0.1:8124` 是 llama.cpp，只提供 **HTTP + SSE**（`stream=true`）✓
  · 已有链路只认 **WebSocket** `/v1/realtime`（vLLM 的协议）✗
  · 集群里那个提供 WS 的 `vllm-asr`（NodePort 30805）目前起不来 ✗

协议对齐（全部来自 agent_stream.py 的真机实测注释）：
  客户端 → 服务端：
    {"type":"session.update","model":"qwen3-asr"}            # 扁平，必须
    {"type":"input_audio_buffer.append","audio":"<b64 PCM16 16k mono>"}
    {"type":"input_audio_buffer.commit","final":bool}        # 解码触发点
  服务端 → 客户端：
    {"type":"session.created"}
    {"type":"transcription.delta","delta":"<新增文本>"}
    {"type":"transcription.done","text":"<累计文本>"}

实现策略（因为 llama.cpp 只能"整段解码"）：
  · 说话期间每 REDECODE_MS 把【累积音频】重解一次 → 与上次结果做**后缀差分** → 吐 delta
  · 收到 commit → 立刻重解一次 + 发 transcription.done
  · final=True → 清空缓冲，准备下一句

用法：
    python aoe4/asr_stream_adapter.py                 # 起服务（默认 127.0.0.1:8788）
    python aoe4/asr_stream_adapter.py --selftest      # 自测：拿 wav 走一遍完整协议
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import struct
import time
import urllib.request

ASR_URL = "http://127.0.0.1:8124/v1/audio/transcriptions"
MODEL = "qwen3-asr"
SR = 16000
REDECODE_MS = 2000         # 说话期间的重解间隔
# ★★ 全局单并发：整段重解很贵（6s 缓冲 ≈ 2~4s CPU）。多会话/与 B 路叠加会
#    让 llama-server 并发解码 → 实测 6 核跑满（599%）。全局只允许一个解码在飞。
_DECODE_LOCK = None


def _lock():
    global _DECODE_LOCK
    if _DECODE_LOCK is None:
        import threading
        _DECODE_LOCK = threading.Lock()
    return _DECODE_LOCK
MIN_AUDIO_S = 0.30         # 少于这么久不解码（避免噪声出字）


def pcm16_to_wav(pcm: bytes, sr: int = SR) -> bytes:
    """给裸 PCM16 套一个 WAV 头（llama.cpp 只吃 RIFF/WAV）。"""
    n = len(pcm)
    return (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) +
            b"data" + struct.pack("<I", n) + pcm)


def _decode_blocking(pcm: bytes) -> str:
    """POST 到本机 ASR（stream=true 走 SSE），累积出完整文本。"""
    if len(pcm) < int(SR * 2 * 0.05):
        return ""
    boundary = "----vbstream"
    wav = pcm16_to_wav(pcm)
    body = b""
    for name, val in (("model", MODEL), ("stream", "true")):
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{val}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode() + wav + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(ASR_URL, data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}"})
    out = []
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                t = d.get("type") or ""
                if t.endswith("delta") and d.get("delta"):
                    out.append(d["delta"])
                elif t.endswith("done") and d.get("text"):
                    return d["text"]
    except Exception:
        pass
    text = "".join(out)
    # 剥掉模型每次都带的 language 前缀
    import re
    text = re.sub(r"language\s+[A-Za-z]+(?:<asr_text>)?", "", text, flags=re.I)
    return text.replace("<asr_text>", "").strip()


async def _decode(pcm: bytes) -> str:
    """全局单并发解码（防止多会话/与 B 路叠加把 ASR 服务器打满）。"""
    import asyncio as _aio

    def _job():
        with _lock():
            return _decode_blocking(pcm)

    return await _aio.to_thread(_job)


class Session:
    def __init__(self, ws):
        self.ws = ws
        self.pcm = bytearray()
        self.sent_text = ""          # 已经吐给客户端的累计文本
        self.last_decode = 0.0
        self.busy = False            # ★ 上一次重解还没回来就别再发（否则挤占最终 ASR 算力）

    async def _emit_new(self, full: str) -> None:
        full = (full or "").strip()
        if not full:
            return
        if full.startswith(self.sent_text):
            new = full[len(self.sent_text):]
        else:                        # 模型改写/纠正了前缀 → 只吐新增的尾巴
            new = full[len(self.sent_text):] if len(full) > len(self.sent_text) else ""
        if new:
            self.sent_text = full
            await self.ws.send(json.dumps({"type": "transcription.delta", "delta": new},
                                          ensure_ascii=False))

    async def append(self, audio_b64: str) -> None:
        try:
            self.pcm += base64.b64decode(audio_b64 or "")
        except Exception:
            return
        now = time.time()
        if (not self.busy
                and len(self.pcm) >= int(SR * 2 * MIN_AUDIO_S)
                and (now - self.last_decode) * 1000 >= REDECODE_MS):
            self.last_decode = now
            self.busy = True
            try:
                await self._emit_new(await _decode(bytes(self.pcm)))
            finally:
                self.busy = False

    async def commit(self, final: bool) -> None:
        if len(self.pcm) >= int(SR * 2 * 0.05):
            await self._emit_new(await _decode(bytes(self.pcm)))
        await self.ws.send(json.dumps({"type": "transcription.done",
                                       "text": self.sent_text}, ensure_ascii=False))
        if final:
            self.pcm.clear()
            self.sent_text = ""
            self.last_decode = 0.0


async def handler(ws):
    await ws.send(json.dumps({"type": "session.created"}))
    sess = Session(ws)
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            t = msg.get("type")
            if t == "input_audio_buffer.append":
                await sess.append(msg.get("audio"))
            elif t in ("input_audio_buffer.commit", "commit"):
                await sess.commit(bool(msg.get("final")))
            elif t == "session.update":
                await ws.send(json.dumps({"type": "session.updated"}))
    except Exception:
        pass


async def serve(host: str, port: int) -> None:
    import websockets
    async with websockets.serve(handler, host, port, max_size=None):
        print(f"   SSE→WS 适配器已启动： ws://{host}:{port}/v1/realtime")
        print(f"   后端 ASR： {ASR_URL}（stream=true SSE）")
        print(f"   重解间隔： {REDECODE_MS} ms")
        await asyncio.Future()


async def selftest(wav_path: str, url: str) -> int:
    import websockets
    import wave
    with wave.open(wav_path, "rb") as w:
        pcm = w.readframes(w.getnframes())
        sr = w.getframerate()
    print(f"   自测音频: {wav_path}  {len(pcm)/2/sr:.2f}s @ {sr}Hz")
    async with websockets.connect(url, max_size=None) as ws:
        print("   收到:", (await ws.recv())[:80])
        await ws.send(json.dumps({"type": "session.update", "model": MODEL}))
        chunk = int(sr * 0.03) * 2
        t0 = time.time()
        first = None
        for i in range(0, len(pcm), chunk):
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(pcm[i:i + chunk]).decode()}))
            await asyncio.sleep(0.03)
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.001))
                if m.get("type") == "transcription.delta" and first is None:
                    first = (time.time() - t0) * 1000
                    print(f"   首个 delta @ {first:.0f} ms: {m.get('delta')!r}")
            except asyncio.TimeoutError:
                pass
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        while True:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            except asyncio.TimeoutError:
                break
            if m.get("type") == "transcription.done":
                print(f"   最终: {m.get('text')!r}   总耗时 {(time.time()-t0)*1000:.0f} ms")
                break
    return 0


def main() -> int:
    global ASR_URL, REDECODE_MS
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--asr-url", default=ASR_URL)
    ap.add_argument("--redecode-ms", type=int, default=REDECODE_MS)
    ap.add_argument("--selftest", default=None, metavar="WAV")
    a = ap.parse_args()
    ASR_URL, REDECODE_MS = a.asr_url, a.redecode_ms
    if a.selftest:
        return asyncio.run(selftest(a.selftest, f"ws://{a.host}:{a.port}/v1/realtime"))
    asyncio.run(serve(a.host, a.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
