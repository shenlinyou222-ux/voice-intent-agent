# -*- coding: utf-8 -*-
"""vLLM /v1/realtime 客户端（同步接口，内部跑 asyncio 线程）

协议全部来自 2026-10-14 真机实测（见 STREAMING_ENGINE_NOTES.md 第三节）：
  · 连上收 `session.created`；`session.update` 是**扁平** {"type","model"}（嵌套会 error，可选）
  · 连续 `input_audio_buffer.append` {"audio": base64(PCM16 16k mono)} —— **append 不触发解码**
  · **`commit` 才是解码触发点** → 收 `transcription.delta`（字段名是 `delta`）/ `transcription.done`
  · 每次 commit 的输出都带 `language X<asr_text>` 前缀 → 必须剥
  · delta 实测为【只追加】（不改写前缀）

为什么是同步接口：app 的主循环是同步的（读麦克风→推 VAD），不能塞个事件循环进去。
所以内部起一个线程跑 asyncio；`push()` 只往队列里塞字节，回调 `on_text` 在**内部线程**里
被调用 —— 调用方请把文本丢进自己的线程安全队列，别在回调里做重活。

用法：
    s = AgentStream(url, model="qwen3-asr", commit_ms=500, on_text=cb)
    s.start()                       # 连接（不阻塞）
    s.push(pcm16_bytes)             # 按住期间每 30ms 一块
    s.commit(final=True)            # 松手：让引擎把最后一段也吐出来
    s.stop()
    # s.state == "ready" | "connecting" | "error" | "closed" → 调用方据此决定回退
"""
import asyncio
import base64
import queue
import re
import threading
import time

_PREFIX = re.compile(r"language\s+[A-Za-z]+(?:<asr_text>)?", re.I)
# 结尾可能是"没写完的前缀标记"（模型会把它拆在多个 delta 里）：
#   'language' / 'language Chi' / '<asr' / '<asr_text' … 这些先不发，等它完整
_PARTIAL_TAIL = re.compile(r"(?:language(?:\s+[A-Za-z]*)?|<asr_?(?:t(?:e(?:x(?:t)?)?)?)?)$",
                           re.I)


def strip_prefix(t: str) -> str:
    """剥掉每次窗口（commit）都会带的 `language X` / `language X<asr_text>` 前缀。

    ⚠️ 实测两种形态都有：带 `<asr_text>` 和不带。只匹配带标签的那种会剥不干净。
    """
    return _PREFIX.sub("", t or "").replace("\r", "")


class AgentStream:
    def __init__(self, url, model="qwen3-asr", commit_ms=500, on_text=None,
                 on_log=None, max_backoff=8.0):
        self.url = url
        self.model = model
        self.commit_ms = max(100, int(commit_ms))
        self.on_text = on_text or (lambda text, final: None)
        self.on_log = on_log or (lambda msg: None)
        self.max_backoff = max_backoff
        self.state = "connecting"
        self.n_fail = 0          # 连续失败次数（只为"降级只说一次"服务）
        self.n_delta = 0
        self.n_commit = 0
        self.n_audio_sent = 0        # 实发出去的 append 条数（诊断用）
        self.last_err = ""
        self._q = queue.Queue()
        self._th = None
        self._loop = None
        self._stop = threading.Event()
        self._raw = ""       # 累积的原始文本（含 language 标记）
        self._clean = ""     # 剥掉标记后的文本
        self._sent = 0       # 已发出的 clean 字符数（指针只前进不回退）

    # ── 同步侧接口 ──
    def start(self):
        self._th = threading.Thread(target=self._thread_main, daemon=True,
                                    name="agent-stream")
        self._th.start()
        return self

    def push(self, pcm16: bytes):
        """追加音频（PCM16 小端、16k、单声道）。不阻塞、不等待结果。"""
        if pcm16 and not self._stop.is_set():
            self._q.put(("audio", pcm16))

    def commit(self, final: bool = False):
        """让引擎立刻解码已缓冲的音频。final=True 表示这一段说完了。"""
        if not self._stop.is_set():
            self._q.put(("commit", final))

    def stop(self, timeout=3.0):
        self._stop.set()
        self._q.put(("__stop__", None))
        if self._th:
            self._th.join(timeout=timeout)
        self.state = "closed"

    # ── 内部：线程 + 事件循环 ──
    def _thread_main(self):
        try:
            import websockets  # 没装就抛，调用方自行回退
        except Exception as e:
            self.state = "error"
            self.last_err = f"no websockets: {e}"
            self.on_log(f"⚠️ AgentStream 不可用（{e}）→ 应回退离线路径")
            return
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run(websockets))
        except Exception as e:
            self.state = "error"
            self.last_err = f"{type(e).__name__}: {e}"
            self.on_log(f"⚠️ AgentStream 退出：{self.last_err}")
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    async def _run(self, websockets):
        backoff = 0.5
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.url, max_size=None,
                                              open_timeout=5, close_timeout=2) as ws:
                    # ⚠️ 必须发 session.update 校验模型，否则后续每条消息都回
                    #    error: "Model not validated ... (model_not_validated)"
                    #    —— 实测（自测第一次跑就撞上）。字段是【扁平】{type, model}。
                    await ws.send(_json({"type": "session.update", "model": self.model}))
                    self.state = "ready"
                    backoff = 0.5
                    self.n_fail = 0
                    self.on_log(f"🔗 流式已连上 {self.url}（已校验 model={self.model}）")
                    await asyncio.gather(self._sender(ws), self._receiver(ws))
            except Exception as e:
                self.state = "error"
                self.last_err = f"{type(e).__name__}: {e}"
                self.n_fail = getattr(self, "n_fail", 0) + 1
                if self.n_fail == 1:
                    # ★ 双线设计：A 路（短切分抢跑）只是【加速】，B 路（整段 ASR）
                    #   才是【权威】。A 路连不上 = 正常降级（退回 B-only），不是警告
                    #   —— 只说一次，之后静默自愈（每 30s 试一次，不再刷日志）。
                    self.on_log("流式不可用 → A 路（抢跑）关闭，B 路（整段 ASR）继续工作")
                backoff = 30.0
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            self.state = "connecting"

    async def _sender(self, ws):
        started = False
        pending = False
        first_audio_t = 0.0
        while not self._stop.is_set():
            sent = 0
            # 一次多处理几条（早先每条 sleep 10ms，518 块要 5.2 秒才发完 → 自测里 delta 0 条）
            for _ in range(64):
                try:
                    kind, val = self._q.get_nowait()
                except queue.Empty:
                    break
                if kind == "__stop__":
                    return
                if kind == "audio":
                    await ws.send(_json({"type": "input_audio_buffer.append",
                                         "audio": base64.b64encode(val).decode()}))
                    self.n_audio_sent += 1
                    if not pending:
                        first_audio_t = time.time()
                    pending = True
                elif kind == "commit":
                    await ws.send(_json({"type": "input_audio_buffer.commit",
                                         "final": bool(val)}))
                    self.n_commit += 1
                    if val:
                        # 一句结束 → 下一句必须【重新】发启动 commit，
                        # 否则第二句永远不会开始解码（长连接场景实测必需）
                        started = False
                    else:
                        started = True
                sent += 1
            # ⚠️⚠️ 服务端语义（源码 connection.py:154-175 实测）：
            #   · 非 final commit 只是【启动一次生成】，生成会持续消费之后 append 的音频
            #   · commit 在生成进行中会被【忽略】（"Generation already in progress"）
            #   · 模型侧按【固定 5 秒】一段解码（qwen3_asr_realtime.segment_duration_s=5.0），
            #     所以 delta 的节奏是 5 秒，不是几百毫秒
            #   因此：攒够 commit_ms 音频后只发【一次】启动 commit，之后不要再发。
            if pending and not started and \
                    (time.time() - first_audio_t) * 1000 >= self.commit_ms:
                await ws.send(_json({"type": "input_audio_buffer.commit",
                                     "final": False}))
                self.n_commit += 1
                started = True
            if sent == 0:
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0)

    async def _receiver(self, ws):
        async for raw in ws:
            try:
                d = _loads(raw)
            except Exception:
                continue
            t = d.get("type")
            if t == "transcription.delta":
                self._emit(d.get("delta") or "")
            elif t == "transcription.done":
                # ⚠️ done.text 是【本次连接的全量文本】，直接再喂一遍会让整句被重复切
                #    （实测：段数 19 而不是 8）。这里只补"delta 没给到的那点差集"。
                dt = strip_prefix(d.get("text") or "")
                if dt and not self._clean.startswith(dt) and dt != self._clean:
                    extra = dt[len(self._clean):] if dt.startswith(self._clean) else ""
                    if extra:
                        self._clean += extra
                        self.on_text(extra, False)
                self.on_text("", True)          # 只是"这一段结束了"的信号
            elif t == "error":
                self.last_err = f"{d.get('error')} ({d.get('code')})"
                self.on_log(f"   ⚠️ 流式回 error：{self.last_err}")
            elif t == "session.created":
                self.on_log(f"   · session {d.get('id')}")

    def _emit(self, delta_raw: str):
        """⚠️⚠️ 两个实测血债（都在这里修）：
        ① 前缀必须在【拼接后的文本】上剥，不能按单个 delta 剥 ——
           模型把 'language' / ' Chinese' / '<asr_text>' 拆在多个 delta 里，
           逐条剥谁都匹配不上 → 标记混进正文。
        ② 结尾"没写完的标记"先不发 —— 否则半截 'language' 已经发出去，
           等标记补全被剥掉时 clean 文本会【变短】，若再兜底重发整段，
           就会把整句重复切一遍（实测：段数 19 而不是 8，10.3s/15.4s 处出现重复段）。
        所以：只发【已成形文本】的增量，且指针只前进不回退。
        """
        if not delta_raw:
            return
        self._raw += delta_raw
        m = _PARTIAL_TAIL.search(self._raw)
        safe = self._raw[:m.start()] if m else self._raw
        clean = strip_prefix(safe)
        if len(clean) < self._sent:          # 防御：万一还是缩了，别重发
            self.on_log(f"   ⚠️ 流式文本回缩 {self._sent}→{len(clean)}（已忽略，不重发）")
            self._sent = len(clean)
            return
        new = clean[self._sent:]
        self._sent = len(clean)
        self._clean = clean
        if new:
            self.n_delta += 1
            self.on_text(new, False)


def _json(o):
    import json
    return json.dumps(o)


def _loads(s):
    import json
    return json.loads(s)


def pcm16(block_floats):
    """float32 [-1,1] → PCM16 little-endian bytes"""
    import numpy as np
    x = np.clip(np.asarray(block_floats, dtype=np.float32), -1.0, 1.0)
    return (x * 32767.0).astype("<i2").tobytes()
