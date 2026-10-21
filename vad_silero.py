# -*- coding: utf-8 -*-
r"""标准 VAD 分句：Silero（神经网络 VAD）+ 前置 AGC

为什么重做：原来的分句是"能量阈值 + 噪声跟踪 + 连续发声"一堆手调参数，
在"麦克风电平 −70 dBFS"这种输入上无论怎么调都只能靠运气（实测 78% 的段是噪声）。

业界标准链路是：
    采集 → AGC/归一化（把电平拉到正常工作区）→ 神经网络 VAD → 分句 → ASR
本文件就是这条链路的实现：
  · PreGain   —— 前置 AGC（目标 RMS 0.03 ≈ −30 dBFS，即标准 VAD 的工作区）
  · SileroVAD —— Silero VAD 模型（onnxruntime，本机已有 v5/v4 的 onnx）
  · StateMachine —— **逐行复用你们 git 仓库里的标准实现**
       open-llm-vtuber:src/open_llm_vtuber/vad/silero.py
       滞回判据：平滑后的 prob ≥ 0.4 且 dB ≥ 60 才命中；
                 命中 3 块（3×32ms≈0.1s）起段；连续 24 块（≈0.77s）不命中收段；
                 预滚动 20 块（≈0.64s）避免吃掉起音
  · VADFactory —— 与那份实现同形状的工厂，方便换后端
"""
import collections
import ctypes
import os
import time
from enum import Enum
from pathlib import Path

import numpy as np

SR = 16000
WINDOW = 512                      # Silero 固定要求：16k 下 512 样本（32ms）

# 本机已有的 Silero 权重（git 仓库 open-llm-vtuber 的前端 libs 里也有一份）
_CANDIDATES = [
    Path(r"C:\Users\user\Desktop\ds harness\open-llm-vtuber\frontend\libs\silero_vad_v5.onnx"),
    Path(r"C:\Users\user\Desktop\ds harness\minimind-o\model\vad\silero_vad.onnx"),
    Path(r"C:\Users\user\Desktop\ds harness\open-llm-vtuber\frontend\libs\silero_vad_legacy.onnx"),
]


def find_model():
    for p in _CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("找不到 silero onnx 权重")


# ══════════════════════════════════════════════════════════════
#  前置 AGC：把电平拉进标准 VAD 的工作区
# ══════════════════════════════════════════════════════════════
class PreGain:
    """百分位 AGC —— 用最近 window_s 内帧 RMS 的 p95 估"语音电平"，目标 target_rms。

    为什么敢用这么激进的增益：Silero 对增益不敏感（实测 1×~200× 下噪声概率始终
    <0.05），所以增益偏大不会把噪声变成语音；反过来增益太小则一切都像静音。
    """

    def __init__(self, target_rms=0.05, window_s=None, pct=None,
                 min_gain=0.1, max_gain=80.0, smooth=0.3, ceiling=0.95):
        # [OPENCODE] AGC 百分位/窗口可配（默认改为【跟随典型语音电平】）
        #   实测（2026-10-19 用户反馈『对着麦说话录不到』）：silero 要 post-gain
        #   RMS >= ~0.03 才判得出人声（0.005→prob 0.017 / 0.05→0.868）。
        #   旧 pct=95/10s：房间几下响动就把增益压下去 → 正常说话落在 0.02~0.04
        #   → prob 中位 0.2~0.4，卡在 0.30 门槛上下 → 时好时坏 ✗
        #   改 pct=75/8s：跟着典型说话电平抬；突变仍由 smooth/ceiling 兜住。
        import os as _os
        if pct is None:
            pct = float(_os.environ.get("VOICE_AGC_PCT", "75") or 75)
        if window_s is None:
            window_s = float(_os.environ.get("VOICE_AGC_WINDOW", "8") or 8)
        self.pct_cfg = float(pct)
        self.window_s_cfg = float(window_s)
        pct = self.pct_cfg
        window_s = self.window_s_cfg
        self.target = float(target_rms)
        self.maxlen = int(window_s * SR / 480)
        self.pct = pct
        self.min_gain, self.max_gain = min_gain, max_gain
        self.smooth = smooth
        # 抗削波上限：本块峰值×增益不许越过它（标准 limiter 做法）
        self.ceiling = float(ceiling)
        self._hist = collections.deque(maxlen=max(10, self.maxlen))
        self.gain = 1.0
        self.last_g = 1.0        # 本块实际用的增益（含 limiter 让步）
        self._last = 0.0

    def apply(self, block):
        rms = float(np.sqrt(np.mean(block ** 2)))
        self._hist.append(rms)
        pk = float(np.max(np.abs(block))) if block.size else 0.0
        now = time.time()
        if now - self._last > 0.5 and len(self._hist) > 10:
            ref = float(np.percentile(np.fromiter(self._hist, dtype=np.float64),
                                      self.pct))
            want = self.target / max(ref, 1e-6)
            want = float(min(max(want, self.min_gain), self.max_gain))
            self.gain += (want - self.gain) * self.smooth       # 慢平滑，防泵动
            self._last = now
        g = self.gain
        # ⭐ 抗削波（limiter）+ 上限收到 30 —— 注意：**这不是**"听不到说话"的真凶。
        #   真凶是状态机里的硬电平门限（见 intent_config 的 vad_db_threshold 注释）：
        #   门限 60 要求块 RMS>=0.0305，把 prob=0.999 的语音段整个拦掉。
        #   这一段的实测依据是：本机说话约 0.0015 RMS，AGC 想要 0.03/0.0003=×100
        #   去够目标电平，增益会顶到上限；×100 时电平漂移过大（日志见 0.13~0.22），
        #   而离线实测 ×10~30 才是该麦克风落在 Silero 语音区间（prob 0.956~0.999）
        #   的增益。削波率实测 0.00%（曾怀疑是削波，已用 test_agc_clip.py 否定），
        #   所以 limiter 在这里是**防御性**的：一旦真实语音更响（0.01~0.05 峰值），
        #   它保证不硬削，而不是用来修当前故障。
        if pk > 1e-9 and pk * g > self.ceiling:
            g = self.ceiling / pk
        self.last_g = g
        return np.clip(block * g, -1.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════
#  Silero 模型（onnxruntime；v4 用 h/c，v5 用 state）
# ══════════════════════════════════════════════════════════════
class SileroVAD:
    def __init__(self, path=None, threads=1):
        import onnxruntime as ort
        self.path = str(path or find_model())
        so = ort.SessionOptions()
        so.inter_op_num_threads = threads
        so.intra_op_num_threads = threads
        so.log_severity_level = 4
        self.sess = ort.InferenceSession(self.path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.sr = np.array(SR, dtype=np.int64)
        self.reset()

    def reset(self):
        if "state" in self.inputs:                    # v5
            self.state = np.zeros((2, 1, 128), dtype=np.float32)
        else:                                         # v4: h/c
            self.h = np.zeros((2, 1, 64), dtype=np.float32)
            self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def prob(self, chunk512):
        x = chunk512.reshape(1, -1).astype(np.float32)
        if "state" in self.inputs:
            out, self.state = self.sess.run(None, {"input": x, "state": self.state,
                                                   "sr": self.sr})
        else:
            out, self.h, self.c = self.sess.run(None, {"input": x, "h": self.h,
                                                       "c": self.c, "sr": self.sr})
        return float(out[0][0])


# ══════════════════════════════════════════════════════════════
#  StateMachine —— 逐行复用 git 仓库里的标准实现
#    open-llm-vtuber:src/open_llm_vtuber/vad/silero.py
# ══════════════════════════════════════════════════════════════
class State(Enum):
    IDLE = 1
    ACTIVE = 2
    INACTIVE = 3


class StateMachine:
    # ★ 2026-10-21 默认值对齐线上实测口径（原 0.4/60/3/24/5 是"常开听不到人"的旧值：
    #   db 60 要求块 RMS>=0.0305，会把 prob=0.999 的轻语音整段拦掉）
    def __init__(self, prob_threshold=0.3, db_threshold=40, required_hits=1,
                 required_misses=16, smoothing_window=2, pre_buffer_chunks=20):
        self.state = State.IDLE
        self.prob_threshold = prob_threshold
        self.db_threshold = db_threshold
        self.required_hits = required_hits
        self.required_misses = required_misses
        self.smoothing_window = smoothing_window
        self.probs, self.dbs, self.chunks = [], [], []
        self.miss_count = 0
        self.hit_count = 0
        self.prob_window = collections.deque(maxlen=smoothing_window)
        self.db_window = collections.deque(maxlen=smoothing_window)
        self.pre_buffer = collections.deque(maxlen=pre_buffer_chunks)
        self._pending = []

    @classmethod
    def calculate_db(cls, audio_data):
        rms = np.sqrt(np.mean(np.square(audio_data)))
        return 20 * np.log10(rms + 1e-7) if rms > 0 else -np.inf

    def reset_buffers(self):
        self.probs.clear()
        self.dbs.clear()
        self.chunks.clear()

    def _smoothed(self, prob, db):
        self.prob_window.append(prob)
        self.db_window.append(db)
        return float(np.mean(self.prob_window)), float(np.mean(self.db_window))

    def process(self, prob, chunk):
        """喂一块（512 样本 float32），返回切出来的段（可能 0 或 1 个）"""
        int_chunk = chunk * 32767                       # 与标准实现一致：dB 按 int16 量纲
        db = self.calculate_db(int_chunk)
        sp, sd = self._smoothed(prob, db)
        hit = (sp >= self.prob_threshold and sd >= self.db_threshold)
        seg = None

        # ⭐⭐ 预滚动必须是一个【一直在滚】的环形缓冲（血债，2026-10-14 实测）
        #   旧版只在 IDLE 里 append，收段时 pre_buffer.clear() 再补 1 块 →
        #   上一段刚收完、紧接着又说话时，预滚动远不足 640ms，
        #   而触发还要等「5 块平滑 + hits」≈200~300ms → 段起点落到起音【之后】
        #   = 从说话中间切进来。实测证据（同一句话两条路）：
        #     utt/162207_mic_dictation_就好像可以了.wav 段头 = 段内 RMS×2.35
        #     （整段最响就在开头），转写吃掉「感」；
        #     同一句 162212 转写完整「感觉好像可以了吧。」= 两条拼接、还重复一条。
        #   改法：无论什么状态，每块都进 pre_buffer；收段时【不清空】。
        #   为什么不会因此把上一段的尾巴重复带进新段：VAD 要连丢 22 块（≈704ms）
        #   才收段，所以新段触发时 pre_buffer 的 640ms 窗口里只剩那段停顿。
        self.pre_buffer.append(chunk)

        if self.state == State.IDLE:
            if hit:
                self.hit_count += 1
                if self.hit_count >= self.required_hits:
                    self.state = State.ACTIVE
                    self.reset_buffers()
                    for c in self.pre_buffer:            # 预滚动补回起音
                        self.chunks.append(c)
                    self.hit_count = 0
            else:
                self.hit_count = 0
        elif self.state == State.ACTIVE:
            self.chunks.append(chunk)
            if hit:
                self.miss_count = 0
            else:
                self.miss_count += 1
                if self.miss_count >= self.required_misses:
                    self.state = State.INACTIVE
                    self.miss_count = 0
        elif self.state == State.INACTIVE:
            seg = np.concatenate(self.chunks) if self.chunks else None
            self.reset_buffers()
            self.state = State.IDLE
            # 刻意【不】 clear pre_buffer：当前块在上面的 append 里已经进去了
        return seg


# ══════════════════════════════════════════════════════════════
#  引擎：把 480/任意长度的流按 512 对齐喂给 Silero
# ══════════════════════════════════════════════════════════════
class SileroVADEngine:
    # ★ 默认值与 intent_config.json / opencode_fastpath/vad_profiles.json 对齐
    #   （未显式传参时也必须能正常起判，不能回落到旧的 0.4/60/3/24/5）
    def __init__(self, model_path=None, pre_gain=True, target_rms=0.05,
                 prob_threshold=0.3, db_threshold=40, required_hits=1,
                 required_misses=16, smoothing_window=2, max_speech_ms=12000,
                 max_gain=80.0):
        self.vad = SileroVAD(model_path)
        self.sm = StateMachine(prob_threshold, db_threshold, required_hits,
                               required_misses, smoothing_window)
        # max_gain 默认 30：本机麦克风说话约 0.0015 RMS，×30 → 0.045，
        # 正好落在 Silero 判"语音"的区间（离线实测 prob 0.956）；
        # 顶到 ×100 会过冲+削波，prob 反而掉到 0.000（见 PreGain 注释）。
        self.gain = (PreGain(target_rms=target_rms, max_gain=max_gain)
                     if pre_gain else None)
        self.max_speech_frames = int(max_speech_ms / (1000 * WINDOW / SR))
        self._buf = np.zeros(0, dtype=np.float32)
        self._rms_hist = collections.deque(maxlen=100)
        self._frame_count = 0
        self.last_prob = 0.0
        self.last_db = -np.inf
        # 本段语音里见过的最大的 VAD 概率 + 最小 SNR 余量（用于日志诊断）
        self.last_seg_prob = 0.0
        self._pmax = 0.0

    # 与现有 Segmenter 兼容的接口
    @property
    def state(self):
        return "speech" if self.sm.state == State.ACTIVE else "idle"

    # 兼容旧接口：标定那套只用于显示噪声底，Silero 侧不需要（AGC 前置）
    def set_noise_floor(self, floor):
        self.noise_floor = float(floor)

    rms_floor = 0.0
    rms_ceiling = 0.0
    voice_margin = 0.0
    energy_threshold = 0.0
    # ⚠️ 兼容字段：旧代码里有几处用 self.seg.frame_len 做"峰值帧 RMS"分帧。
    #    Silero 的真实窗长是 WINDOW(512)，这里给同值，别再用 480 硬编码。
    frame_len = WINDOW
    # 注意：**故意不提供** last_run_ms / last_voiced_ms / 能量型 threshold
    #   —— 调用方用 hasattr(self.seg, "last_run_ms") 判断走哪套诊断，
    #   提供了就会打印出"门槛 0.40000"这种把概率当能量门槛的误导输出。

    @property
    def level(self):
        return max(self._rms_hist) if self._rms_hist else 0.0

    @property
    def peak_prob(self):
        """最近一次收段的【峰值 VAD 概率】—— 日志里报"VAD峰"必须用它。

        血债（2026-10-19）：原来没有这个属性 → 调用方 `getattr(seg,"peak_prob",0.0)`
        永远拿到 0.000，日志里那行「VAD峰0.000」是假的（真峰值在 last_seg_prob 上）。
        """
        try:
            return float(self.last_seg_prob or self._pmax or 0.0)
        except Exception:
            return 0.0

    @property
    def threshold(self):
        return self.sm.prob_threshold

    @property
    def gain_now(self):
        # 本块【实际】施加的增益：limiter 让步时它小于平滑出来的目标增益，
        # 日志里看到的就是真正进模型的那个电平。
        return self.gain.last_g if self.gain else 1.0

    def push(self, block):
        x = np.asarray(block, dtype=np.float32).ravel()
        if self.gain is not None:
            x = self.gain.apply(x)
        self._rms_hist.append(float(np.sqrt(np.mean(x ** 2))))
        self._buf = np.concatenate([self._buf, x])
        out = []
        while len(self._buf) >= WINDOW:
            chunk = self._buf[:WINDOW]
            self._buf = self._buf[WINDOW:]
            self._frame_count += 1
            p = self.vad.prob(chunk)
            self.last_prob = p
            self.last_db = StateMachine.calculate_db(chunk * 32767)
            seg = self.sm.process(p, chunk)
            if seg is not None and len(seg) > 0:
                # 段结束时记下这段的峰概率（last_prob 此刻已是静音帧，会误导日志）
                self.last_seg_prob = self._pmax
                self._pmax = 0.0
                out.append(seg)
            else:
                self._pmax = max(self._pmax, p)
                if self.sm.state == State.ACTIVE and \
                        len(self.sm.chunks) >= self.max_speech_frames:
                    # 撞上限：按标准做法强制收段（但已由 VAD 判过是语音，不会收垃圾）
                    self.last_seg_prob = self._pmax
                    self._pmax = 0.0
                    self.sm.state = State.INACTIVE
                    seg = self.sm.process(0.0, chunk)
                    if seg is not None:
                        out.append(seg)
        return out

    def flush(self):
        if self.sm.state == State.ACTIVE and self.sm.chunks:
            seg = np.concatenate(self.sm.chunks)
            self.last_seg_prob = self._pmax
            self._pmax = 0.0
            self.sm.reset_buffers()
            self.sm.state = State.IDLE
            return seg if len(seg) > SR * 0.2 else None
        return None

    def reset(self):
        self.vad.reset()
        self.sm.state = State.IDLE
        self.sm.reset_buffers()
        self.sm.pre_buffer.clear()
        self._buf = np.zeros(0, dtype=np.float32)


class VADFactory:
    """与 git 仓库里同形状的工厂（open-llm-vtuber/src/open_llm_vtuber/vad/vad_factory.py）"""

    @staticmethod
    def get_vad_engine(engine_type="silero_vad", **kw):
        if engine_type != "silero_vad":
            return None
        return SileroVADEngine(model_path=kw.get("model_path"),
                               pre_gain=kw.get("pre_gain", True),
                               target_rms=kw.get("target_rms", 0.05),
                               max_gain=kw.get("max_gain", 80.0),
                               prob_threshold=kw.get("prob_threshold", 0.3),
                               db_threshold=kw.get("db_threshold", 40),
                               required_hits=kw.get("required_hits", 1),
                               required_misses=kw.get("required_misses", 16),
                               smoothing_window=kw.get("smoothing_window", 2),   # ★ 之前没转发 → 传了也不生效 ✗
                               max_speech_ms=kw.get("max_speech_ms", 12000))
