# -*- coding: utf-8 -*-
"""HTTP STT 后端 v2：**完全对标 app 的 ASR 链路，只额外带热词上下文**。

事实链（实测）：
    app 麦克风 → 8019/stt → voice_intent.transcribe →
        POST http://127.0.0.1:8124/v1/audio/transcriptions
        llama-server -m Qwen3-ASR-1.7B-Q4_K_M.gguf
                     --mmproj mmproj-1.7B-Q4_K_M.gguf -c 4096 --port 8124
    （RSS 1519MB / 私有 2286MB，**零显存**，纯 CPU）

所以我们对齐的方式：**不自己跑模型**，直接调这个 8124 服务（同一个进程、同一份
权重、同一个 -c 4096），只在请求里多带一个 `prompt` = 热词上下文。
—— 用户要求："参数全部对标，仅热词上下文保留"。

⚠️ 不走 8019：那个壳的服务端调用是写死的（`transcribe(wav)`），带不进 prompt。
"""
import io
import json
import time
import urllib.error
import urllib.request
import uuid
import wave

import numpy as np

DEFAULT_URL = "http://127.0.0.1:8124/v1/audio/transcriptions"


def _wav_bytes(x, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    return buf.getvalue()


class HTTPASR:
    """调 app 那条 llama-server 转录接口（OpenAI 兼容），额外带热词 prompt。"""

    def __init__(self, url=DEFAULT_URL, timeout=120.0, language="zh",
                 prompt=None, temperature=0.0, send_prompt=False, **kw):
        self.url = url or DEFAULT_URL
        self.timeout = float(timeout)
        self.language = language
        # ⚠️ 2026-10-21 血债：prompt（热词上下文）在 /v1/audio/transcriptions 上
        #   会让 Qwen3-ASR 解码退化 —— 同一段 1.79s 音频实测：
        #     带 prompt：6.7s、154 字「ه류류류…и типать типать…」，撞满 max_tokens
        #     不带    ：0.35s、「喂喂，现在呢？」
        #   所以默认【不发】；确需实验时显式 send_prompt=True。
        self.prompt = prompt or ""      # 保留字段，便于日志/调试
        self.send_prompt = bool(send_prompt)
        self.temperature = float(temperature)
        self.load_ms = 0.0
        self.last_ms = 0.0
        self.last_tokens = 0
        self.last_text = ""

    def load(self):
        """探活：/health（llama-server 自带），失败则直接报错，别静默"""
        t0 = time.perf_counter()
        health = self.url.split("/v1/")[0] + "/health"
        try:
            with urllib.request.urlopen(health, timeout=10) as r:
                r.read()
        except Exception as e:
            raise RuntimeError(
                f"llama-server 不可达：{health}（{type(e).__name__}: {e}）"
                f" —— 它应该由 app 的语音链路在跑；没有就手动起："
                f"llama-server -m Qwen3-ASR-1.7B-Q4_K_M.gguf "
                f"--mmproj mmproj-1.7B-Q4_K_M.gguf -c 4096 --port 8124")
        self.load_ms = (time.perf_counter() - t0) * 1000

    def unload(self):
        return None

    def transcribe(self, wav):
        x = np.asarray(wav, dtype=np.float32).ravel()
        wavb = _wav_bytes(x)
        B = "----dshvoice" + uuid.uuid4().hex
        parts = []

        def field(k, v):
            parts.append(f'--{B}\r\nContent-Disposition: form-data; '
                         f'name="{k}"\r\n\r\n{v}\r\n'.encode("utf-8"))

        field("model", "qwen3-asr")
        field("language", self.language)
        field("response_format", "json")
        if self.prompt and self.send_prompt:
            field("prompt", self.prompt)          # ⚠️ 默认不发（会退化，见 __init__ 注释）
        if self.temperature:
            field("temperature", str(self.temperature))
        parts.append(f'--{B}\r\nContent-Disposition: form-data; name="file"; '
                     f'filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n'
                     .encode("utf-8"))
        parts.append(wavb)
        parts.append(b"\r\n")
        parts.append(f"--{B}--\r\n".encode("utf-8"))
        body = b"".join(parts)

        req = urllib.request.Request(
            self.url, data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={B}",
                     "Content-Length": str(len(body))})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"转录 HTTP {e.code}: "
                               f"{e.read()[:200].decode('utf-8', 'ignore')}")
        self.last_ms = (time.perf_counter() - t0) * 1000
        try:
            js = json.loads(raw)
        except Exception:
            self.last_text = raw.strip()
            return self.last_text
        # ⭐ 解析不到 text 时把【服务端原始响应】打出来（否则静默变空转写，
        #   实测就是这样被吞掉的：agent 显示「(无转写)」而真实原因不明）
        if not (js.get("text") or "").strip():
            raise RuntimeError(f"服务端没返回 text 字段，原始响应: {raw[:400]}")
        self.last_text = (js.get("text") or "").strip()
        # 实测这个服务会把提示词回显在结果里：'language Chinese<asr_text>打开录音。'
        # 不剥掉就会污染意图判定 → 只取 <asr_text> 之后的部分
        if "<asr_text>" in self.last_text:
            self.last_text = self.last_text.split("<asr_text>", 1)[1].strip()
        elif self.last_text.startswith("language "):
            self.last_text = ""
        self.last_tokens = int(js.get("tokens") or 0)
        return self.last_text


if __name__ == "__main__":
    import sys
    a = HTTPASR(prompt=sys.argv[1] if len(sys.argv) > 1 else "")
    a.load()
    print(f"llama-server 可达（探活 {a.load_ms:.0f}ms）  prompt={a.prompt[:60]!r}")
    for p in sys.argv[2:]:
        with wave.open(p, "rb") as w:
            raw = w.readframes(w.getnframes())
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        print(f"  {p.split(chr(92))[-1][:40]}: {a.transcribe(x)!r}  ({a.last_ms:.0f}ms)")
