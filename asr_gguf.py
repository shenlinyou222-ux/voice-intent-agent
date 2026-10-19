# -*- coding: utf-8 -*-
r"""单进程 · 模型以 C++（llama.cpp）形式加载 —— 替代 torch fp32 解码器

    Qwen3-ASR-0.6B-Q4_K_M.gguf   462 MB   ← LLM 部分（量化）
    mmproj-Qwen3-ASR-0.6B-Q8_0.gguf 204 MB ← 音频塔 + 投影（量化）
    ─────────────────────────────────────
    合计 666 MB，对比 torch fp32 官方权重 3,130 MB（4.7×）

推理全程在 llama.dll / mtmd.dll 里（C++），不走 torch、不起外部服务。
音频路径用的就是 mtmd 的 Qwen3-ASR 预处理器（dll 里可见
`mtmd_audio_preprocessor_qwen3a` / `mtmd_bitmap_init_from_audio`）。

用法: E:\py312\python.exe asr_gguf.py [wav ...]
"""
import ctypes
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

try:
    from dlog import log as _log
except Exception:
    def _log(msg="", tag=None, sub=None):
        print(f"{('[' + tag + '] ') if tag else ''}{msg}", flush=True)

MODELS = Path(r"C:\Users\user\Desktop\ds harness\aigc_demo_framework\models")
GGUF = MODELS / "qwen3-asr-gguf" / "Qwen3-ASR-0.6B-Q4_K_M.gguf"
MMPROJ = MODELS / "qwen3-asr-gguf" / "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"
LLAMA_DIR = MODELS / "llamacpp"


class GGUFASR:
    """进程内 llama.cpp 音频 ASR（Qwen3-ASR GGUF + mmproj）"""

    def __init__(self, gguf=GGUF, mmproj=MMPROJ, n_threads=4, n_ctx=512,
                 language="Chinese", max_new_tokens=192, verbose=False,
                 context=None, use_audio_tags=False):
        # ⚠️ n_ctx 决定 KV cache 预分配量：KV ≈ 112 KB/token × n_ctx（28 层 ×
        #    8 kv_heads × 128 head_dim × 2(K/V) × 2B）。抄 server 的 4096 会白占
        #    470 MB；实测一句话只要 ~100 token（12 秒语音 = 65 audio token），
        #    512 就够（还剩 4 倍余量），内存差 394 MB 而速度不变。
        #
        # ⭐ context = 热词/上下文偏置。官方 chat_template.json 里有个
        #    【system 消息】(ns.system_text)，那就是 Qwen3-ASR 的上下文注入通道：
        #    不给它，专有名词只能硬猜汉字 —— 实测现场「启动DeepSeek」被转成
        #    「启动DVC Harness的。」「G.M的。」「骑士版。」这类。
        # ⭐ use_audio_tags：官方模板的音频标记是
        #    <|audio_start|><|audio_pad|><|audio_end|> 三个，
        #    而 mtmd 的 default_marker 只给中间那个。
        self.gguf = str(gguf)
        self.mmproj = str(mmproj)
        self.n_threads = n_threads
        self.n_ctx = n_ctx
        self.language = language
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.context = context
        self.use_audio_tags = use_audio_tags
        self.llm = None
        self.load_ms = 0.0
        self.last_ms = 0.0
        self.last_tokens = 0
        self.last_prompt = ""

    # ── 静音 llama.cpp / mtmd 的 C 层日志 ──
    def _silence_c_logs(self):
        """C 层日志默认直接打 stderr（clip_model_loader / add_text /
        audio_tokens-> / clip_encode …）—— 实测占日志行数的 78%，
        把自己应用层的日志冲得没法看。这里只放行 ERROR 级别。
        """
        try:
            import ctypes
            from llama_cpp import llama_cpp as LC
            CB = getattr(LC, "llama_log_callback", None)
            if CB is None:
                CB = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p,
                                      ctypes.c_void_p)

            def _cb(level, text, user_data):
                try:
                    # ⚠️ llama.h 的枚举是 ERROR=2, WARN=3, INFO=4, DEBUG=5
                    #    （不是 ggml 那套 INFO=2/ERROR=4）—— 一开始按 ggml 写
                    #    `>=4` 恰好把 INFO 全放行了，噪音照打。
                    if level is not None and int(level) <= 3:
                        sys.stderr.write(text.decode("utf-8", "ignore"))
                except Exception:
                    pass

            self._log_cb = CB(_cb)          # 必须持有引用，否则被 GC → 崩
            LC.llama_log_set(self._log_cb, None)
            # 有些版本 ggml 侧还有独立 logger
            try:
                import llama_cpp.ggml as _g   # noqa: F401
            except Exception:
                pass
        except Exception as e:
            _log(f"C 层日志静音失败: {type(e).__name__}: {e}", "WARN")

    # ── 载入（C++ 侧）──
    def load(self):
        if self.llm is not None:
            return
        if os.path.isdir(str(LLAMA_DIR)):
            try:
                os.add_dll_directory(str(LLAMA_DIR))
            except Exception:
                pass
        import llama_cpp
        import llama_cpp.mtmd_cpp as mtmd
        from llama_cpp import Llama
        self._llama_cpp = llama_cpp
        self._mtmd = mtmd
        self._silence_c_logs()

        t0 = time.perf_counter()
        # mmap 加载：GGUF 只映射进地址空间，页按需调入
        self.llm = Llama(model_path=self.gguf, n_ctx=self.n_ctx,
                         n_threads=self.n_threads, n_batch=512,
                         logits_all=False, embedding=False, use_mmap=True,
                         verbose=self.verbose)
        p = mtmd.mtmd_context_params_default()
        p.use_gpu = False
        p.print_timings = False
        p.n_threads = self.n_threads
        self.mtmd_ctx = mtmd.mtmd_init_from_file(self.mmproj.encode(),
                                                 self.llm.model, p)
        if not self.mtmd_ctx:
            raise RuntimeError("mtmd_init_from_file 失败")
        if not bool(mtmd.mtmd_support_audio(self.mtmd_ctx)):
            raise RuntimeError("该 mmproj 不支持音频")
        self.sr = int(mtmd.mtmd_get_audio_sample_rate(self.mtmd_ctx))
        self.marker = mtmd.mtmd_default_marker().decode("utf-8")
        self.load_ms = (time.perf_counter() - t0) * 1000
        n = os.path.getsize(self.gguf) + os.path.getsize(self.mmproj)
        _log(f"llama.cpp 进程内载入 {n/1e6:.0f} MB GGUF（C++）"
             f"  音频采样率 {self.sr}  {self.load_ms/1000:.1f}s", "ASR")

    # ── 卸载（动态加载用：闲久了把内存还回去）──
    def unload(self):
        try:
            if getattr(self, "mtmd_ctx", None):
                self._mtmd.mtmd_free(self.mtmd_ctx)
        except Exception:
            pass
        self.mtmd_ctx = None
        self.llm = None
        import gc
        gc.collect()

    # ── 转写 ──
    def transcribe(self, wav):
        self.load()
        import llama_cpp
        m = self._mtmd
        wav = np.ascontiguousarray(np.asarray(wav, dtype=np.float32).ravel())
        if self.sr and self.sr != 16000:      # 模型要求的采样率不是 16k 时线性重采样
            t = np.linspace(0, len(wav) / 16000, int(len(wav) * self.sr / 16000),
                            endpoint=False)
            wav = np.interp(t, np.arange(len(wav)) / 16000, wav).astype(np.float32)
        t0 = time.perf_counter()

        # 官方 chat_template.json 的形状（见 E:\models\qwen3-asr-0.6b-official）：
        #   <|im_start|>system\n{system_text}<|im_end|>\n
        #   <|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n
        #   <|im_start|>assistant\n[language X]<asr_text>
        sys_part = ""
        if self.context:
            sys_part = f"<|im_start|>system\n{self.context}<|im_end|>\n"
        marker = self.marker
        if self.use_audio_tags:
            marker = f"<|audio_start|>{marker}<|audio_end|>"
        lang_part = f"language {self.language}" if self.language else ""
        text = (f"{sys_part}<|im_start|>user\n{marker}<|im_end|>\n"
                f"<|im_start|>assistant\n{lang_part}<asr_text>")
        self.last_prompt = text
        buf = (ctypes.c_float * len(wav)).from_buffer_copy(wav)
        bmp = m.mtmd_bitmap_init_from_audio(len(wav), buf)
        if not bmp:
            raise RuntimeError("mtmd_bitmap_init_from_audio 失败")
        bitmaps = (m.mtmd_bitmap_p_ctypes * 1)(bmp)

        it = m.mtmd_input_text()
        tb = text.encode("utf-8")
        it.text = tb
        it.text_len = len(tb)
        it.add_special = True
        it.parse_special = True

        chunks = m.mtmd_input_chunks_init()
        rc = m.mtmd_tokenize(self.mtmd_ctx, chunks, ctypes.byref(it), bitmaps, 1)
        if rc != 0:
            raise RuntimeError(f"mtmd_tokenize 失败 rc={rc}")

        self.llm.reset()
        try:
            self.llm._ctx.kv_cache_clear()
        except Exception:
            pass
        n_past = llama_cpp.llama_pos(0)
        rc = m.mtmd_helper_eval_chunks(self.mtmd_ctx, self.llm._ctx.ctx, chunks,
                                       0, 0, 512, True, ctypes.byref(n_past))
        if rc != 0:
            raise RuntimeError(f"mtmd_helper_eval_chunks 失败 rc={rc}")
        # ⚠️ mtmd 是直接写 llama 上下文的 KV cache，高层 Llama 的记账（n_tokens）
        #    不会自动更新 —— 不手动同步，llm.sample() 会断言 n_tokens>0 失败，
        #    而且后续 eval 的 position 也会从头开始（错位）。
        self.llm.n_tokens = int(n_past.value)

        vocab = llama_cpp.llama_cpp.llama_model_get_vocab(self.llm.model)
        L = llama_cpp.llama_cpp
        ctx_p = self.llm._ctx.ctx
        n_vocab = int(L.llama_vocab_n_tokens(vocab))

        # ⭐⭐⭐ 热词【token 前缀树】加分（= NeMo boosting_tree_alpha /
        #      sherpa-onnx hotwords-score 的等价实现）
        #  · 只放名词/专有名词/昵称（用户明确要求：动词、泛词一律不许进）
        #  · 只在"前缀已匹配、下一个 token 还没出"时，给【活跃节点的子节点】加分；
        #    不做静态全局加分（那会乱诱导；NeMo 用树就是为了避免这个）
        #  · 开销：每步只碰几个 id + 一次 numpy argmax（151936 维 ≈ 0.05ms）
        # ⭐⭐⭐ 热词【token 前缀树】+【逐词权重】（= NeMo boosting_tree 的等价物）
        #  热词写法：["ollama:6", "DeepSeek:3", "Trae"]（冒号后是"轻度"，缺省用全局 α）
        #  三条实测得来的护栏：
        #   ① 树根阻尼 root_scale：从根直接命中词首时只给一小部分分，
        #      否则模型会被硬塞热词（α=20 实测：「开，对不对？」→「Ollama×12」）
        #   ② 重复惩罚：同一个 token 连续吐 ≥3 次就把它的加分清零
        #      （α=20 实测：「…」→「Ollama」×60 死循环，永远不出 EOG）
        #   ③ 只在"前缀已匹配"时给【活跃节点的子节点】加分，不做全局静态加分
        _hw = getattr(self, "hotwords", None) or []
        _alpha = float(getattr(self, "hotwords_score", 5.0) or 0.0)
        _root_scale = float(getattr(self, "hotwords_root_scale", 0.5))
        if _hw and getattr(self, "_trie", None) is None:
            self._trie = {}
            for _item in _hw:
                _w, _wt = str(_item), None
                if ":" in _w:                      # 逐词权重
                    _w, _, _s = _w.rpartition(":")
                    try:
                        _wt = float(_s)
                    except ValueError:
                        _w, _wt = str(_item), None
                _wt = _alpha if _wt is None else _wt
                _ids = [t for t in self.llm.tokenize(_w.encode("utf-8"))
                        if t != self.llm.token_bos()]
                if not _ids:
                    continue
                _n = self._trie
                for _t in _ids:
                    _n = _n.setdefault(_t, {})
                    _n["w"] = _wt               # 该路径的权重（逐词覆盖全局 α）
                _n["$"] = True
        _trie = getattr(self, "_trie", None)

        toks = []
        active = [_trie] if _trie else []
        _last_tok, _rep = -1, 0
        for _ in range(self.max_new_tokens):
            lp = ctypes.cast(L.llama_get_logits_ith(ctx_p, -1),
                             ctypes.POINTER(ctypes.c_float))
            arr = np.ctypeslib.as_array(lp, shape=(n_vocab,))
            touched = []
            if active:
                for _node in active:
                    _wt = float(_node.get("w", _alpha))
                    _is_root = (_node is _trie)
                    _boost = _wt * (_root_scale if _is_root else 1.0)
                    for _tid in _node:
                        if _tid in ("$", "w"):
                            continue
                        if _tid == _last_tok and _rep >= 3:
                            continue               # ② 重复惩罚
                        touched.append((_tid, float(arr[_tid])))
                        arr[_tid] += _boost
            tok = int(arr.argmax())
            for _tid, _old in touched:
                arr[_tid] = _old
            if tok < 0 or L.llama_vocab_is_eog(vocab, tok):
                break
            toks.append(tok)
            _rep = _rep + 1 if tok == _last_tok else 0
            _last_tok = tok
            if _trie:
                _nxt = []
                for _node in [_trie] + active:
                    _c = _node.get(tok)
                    if _c is not None:
                        _nxt.append(_c)
                active = _nxt or [_trie]
            self.llm.eval([tok])
        m.mtmd_bitmap_free(bmp)
        raw = self.llm.detokenize(toks).decode("utf-8", "ignore")
        self.last_tokens = len(toks)
        self.last_ms = (time.perf_counter() - t0) * 1000
        # 输出形如 language Chinese<asr_text>xxx
        if "<asr_text>" in raw:
            raw = raw.split("<asr_text>", 1)[1]
        return raw.strip()


def _load_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


if __name__ == "__main__":
    asr = GGUFASR()
    asr.load()
    args = sys.argv[1:]
    if args:
        files = [Path(p) for p in args]
    else:
        files = sorted(Path(__file__).resolve().parent.joinpath("wavs").glob(
            "train_exact*.wav"))[:6]
    for f in files:
        w, sr = _load_wav(f)
        assert sr == 16000, f"{f.name} sr={sr}"
        t = asr.transcribe(w)
        print(f"  {f.name[:34]:36s} {len(w)/16000:5.2f}s → 「{t}」"
              f"  ({asr.last_ms:.0f} ms, {asr.last_tokens} tok)")
