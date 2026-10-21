# -*- coding: utf-8 -*-
"""VAD 分句 —— 把连续麦克风流切成一句一句

═══ 2026-10-12 重做（依据 run_live.log / run_src.log 实测）═══

现场实测：
  · 房间噪声 ≤0.0006（设备在静音时给精确 0，等于带噪声门）
  · 你的语音帧 RMS 约 0.0015~0.004，峰值帧 0.012~0.022
  · 旧门槛 = max(0.008, 噪声底×3) = 0.008 → 比语音高 5~10 倍，能量判据从不因说话触发
  · 于是唯一触发者是 webrtcvad(level 1)，它在数字静音上也会把约 16% 的帧判成语音
    → 切出 63 段、76% 转写为空；撞满 6s 上限的复读乱码【段平均 RMS 0.0014~0.0058】
    反而比两句正确的（0.00055/0.00039）还高 —— 段平均能量对好坏没有可分性。

判据（结构照抄 DSH 语音插件 client.js:32-43，那是在这支麦克风上已经调好的）：

    noise_ref : 慢跟踪的"真实环境底"（快降 ref_fall / 极慢升 ref_rise）
    noise     : 更快的噪声跟踪（noise_down / noise_up）
    rms_floor   = max(noise_ref × floor_mult, ABS_MIN)
    rms_ceiling = max(noise_ref × ceil_mult, rms_floor)
    thr         = clamp(max(rms_floor, noise × voice_margin), rms_floor, rms_ceiling)

  ① 夹紧区间【运行中自校准】，不再锚死在开机那 1.5 秒：
     实测踩中过两种塌法 ——
       · 标定被污染（说话/风扇/系统音）：中位算出 0.00224 → 区间抬到 [0.0034, 0.0179]，
         而语音帧才 0.0012 → VAD 直接聋掉，一句都切不出来；
       · 夹紧值写死成常数：0.0008 比本机语音帧还高 → 连续发声凑不满 240ms → 同样聋。
     现在 noise_ref 用 1%/帧 的下落速度（时间常数 ~3 秒）自动收敛到真实环境底。
     实测：落得太快（试过 5%）会把 ref 拉到环境底以下 → 门槛跟着掉 → 连底噪都被判成
     发声（四个场景里多切出一个噪声段）。所以 1% 是"能自愈、又不塌"的折中：
     被污染的标定约 5~10 秒内自愈（头几秒可能漏掉一句话）。
  ② 连续发声长度 min_run_ms：一段里必须有一串【连续】达标帧。
     随机误判近似按帧独立，命中率 p 时连续 n 帧概率 ≈ p^n（p=0.16、n=8 → 4e-7），
     真实说话轻松给几百毫秒。这一步把"6 秒噪声段"和"被噪声框住的一句话"分开。
  ③ 预滚动 pre_roll_ms：段起点前多带一小段缓冲，避免首音素被削掉
     （旧版从触发的那一帧才开始，起音会被吃掉一截）。

webrtcvad 默认关闭：它的判决基于绝对能量（面向 -30~-20 dBFS 的正常输入），
在本机这种极低电平上会在静音里随机触发。
"""
import collections
import time

import numpy as np

SAMPLE_RATE = 16000

# 绝对下限（≈1.5 个 16bit 计数）。低于它一律不算语音，防全零流把门槛拉到 0。
ABS_MIN_RMS = 1.5 / 32768


class Segmenter:
    """输入 16 kHz float32 小块，输出一句一句的音频"""

    def __init__(self, silence_ms=500, min_speech_ms=200, energy_threshold=0.0025,
                 max_speech_ms=12000, frame_ms=30, use_webrtcvad=False,
                 aggressiveness=1, adaptive_floor=True,
                 floor_mult=1.5, ceil_mult=8.0, voice_margin=2.0,
                 ref_fall=0.01, ref_rise=0.0005,
                 noise_floor=0.0008, noise_down=0.25, noise_up=0.002,
                 min_run_ms=150, pre_roll_ms=300):
        self.frame_ms = frame_ms
        self.frame_len = int(SAMPLE_RATE * frame_ms / 1000)      # 480 @30ms
        self.silence_frames = max(1, int(silence_ms / frame_ms))
        self.min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        self.max_speech_frames = max(1, int(max_speech_ms / frame_ms))
        self.min_run_frames = max(1, int(min_run_ms / frame_ms))

        self.energy_threshold = float(energy_threshold)   # adaptive_floor=False 时的固定门槛
        self.adaptive_floor = bool(adaptive_floor)
        self.floor_mult = float(floor_mult)
        self.ceil_mult = float(ceil_mult)
        self.voice_margin = float(voice_margin)
        self.ref_fall = float(ref_fall)
        self.ref_rise = float(ref_rise)
        self.noise_down = float(noise_down)
        self.noise_up = float(noise_up)
        self.noise_floor = float(noise_floor)     # 开机标定值，仅作种子
        self.noise_ref = None                     # 慢跟踪环境底（夹紧区间基准）
        self.noise = None                         # 快跟踪噪声
        # 先把区间按种子摆出来（供开机日志显示），运行中由 _track_floor 持续更新
        self.rms_floor = max(self.noise_floor * self.floor_mult, ABS_MIN_RMS)
        self.rms_ceiling = max(self.noise_floor * self.ceil_mult, self.rms_floor)

        self._vad = None
        if use_webrtcvad:
            try:
                import webrtcvad
                self._vad = webrtcvad.Vad(int(aggressiveness))
            except Exception:
                self._vad = None

        self._buf = np.zeros(0, dtype=np.float32)
        self._state = "idle"          # idle | speech
        self._speech = []
        self._sil = 0
        self._voiced = 0
        self._run = 0                 # 当前连续发声帧数
        self._best_run = 0            # 本段最长连续发声帧数
        self._rms_hist = []           # 最近 100 帧（3 秒）的 RMS，用来看电平
        self._pre = collections.deque(maxlen=max(0, int(pre_roll_ms / frame_ms)))

        # 给日志用：最近一段的证据
        self.last_floor = 0.0
        self.last_threshold = self.energy_threshold
        self.last_run_ms = 0.0
        self.last_voiced_ms = 0.0

    # ── 开机标定：只作种子，区间由 _track_floor 动态维护 ──
    def set_noise_floor(self, floor):
        self.noise_floor = float(floor)
        self.noise_ref = None
        self.noise = None

    # ── 门槛：慢跟踪环境底 + 快跟踪噪声 + 双向夹紧 ──
    def _track_floor(self, rms):
        if self.noise_ref is None:
            self.noise_ref = max(min(self.noise_floor, 1.0), ABS_MIN_RMS)
        if rms < self.noise_ref:
            self.noise_ref += (rms - self.noise_ref) * self.ref_fall
        else:
            self.noise_ref += (rms - self.noise_ref) * self.ref_rise
        if self.noise is None:
            self.noise = self.noise_ref
        self.noise += (rms - self.noise) * (
            self.noise_down if rms < self.noise else self.noise_up)
        self.last_floor = self.noise
        self.rms_floor = max(self.noise_ref * self.floor_mult, ABS_MIN_RMS)
        self.rms_ceiling = max(self.noise_ref * self.ceil_mult, self.rms_floor)
        if not self.adaptive_floor:
            thr = max(self.energy_threshold, ABS_MIN_RMS)
        else:
            thr = min(max(self.rms_floor, self.noise * self.voice_margin),
                      self.rms_ceiling)
        self.last_threshold = thr
        return thr

    # ── 单帧是不是语音 ──
    def _is_speech(self, frame):
        rms = float(np.sqrt(np.mean(frame ** 2)))
        thr = self._track_floor(rms)
        if rms > thr:
            return True
        if self._vad is not None:
            try:
                pcm = (np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes()
                return bool(self._vad.is_speech(pcm, SAMPLE_RATE))
            except Exception:
                pass
        return False

    # ── 这一段收不收：累计发声 + 连续发声，两个都过 ──
    def _accept(self):
        self.last_run_ms = self._best_run * self.frame_ms
        self.last_voiced_ms = self._voiced * self.frame_ms
        return (self._voiced >= self.min_speech_frames
                and self._best_run >= self.min_run_frames)

    def push(self, chunk):
        """喂一块音频（任意长度 float32），返回 [完整句子, ...]（可能 0 或多句）"""
        self._buf = np.concatenate([self._buf, np.asarray(chunk, dtype=np.float32)])
        out = []
        while len(self._buf) >= self.frame_len:
            frame = self._buf[:self.frame_len]
            self._buf = self._buf[self.frame_len:]
            self._rms_hist.append(float(np.sqrt(np.mean(frame ** 2))))
            if len(self._rms_hist) > 100:
                self._rms_hist.pop(0)
            sp = self._is_speech(frame)
            if self._state == "idle":
                if sp:
                    self._state = "speech"
                    # ⭐ 预滚动：把起音前缓冲的帧接到段首（不计入发声证据）
                    self._speech = list(self._pre) + [frame]
                    self._pre.clear()
                    self._sil = 0
                    self._voiced = 1
                    self._run = 1
                    self._best_run = 1
                else:
                    self._pre.append(frame)
            else:
                self._speech.append(frame)
                if sp:
                    self._voiced += 1
                    self._run += 1
                    if self._run > self._best_run:
                        self._best_run = self._run
                    self._sil = 0
                else:
                    self._run = 0
                    self._sil += 1
                # 判停：静音够久
                if self._sil >= self.silence_frames:
                    if self._accept():
                        out.append(np.concatenate(self._speech))
                    self._reset()
                elif len(self._speech) >= self.max_speech_frames:
                    # ⚠️ 撞上限的段也必须过同一套判据 —— 旧版在这里无条件收，
                    #    正是 6 秒复读乱码的来源。
                    if self._accept():
                        out.append(np.concatenate(self._speech))
                    self._reset()
        return out

    def flush(self):
        """强制结束当前句（关闭时用）"""
        got = None
        if self._state == "speech" and self._accept():
            got = np.concatenate(self._speech)
        self._reset()
        return got

    def _reset(self):
        self._state = "idle"
        self._speech = []
        self._sil = 0
        self._voiced = 0
        self._run = 0
        self._best_run = 0
        self._pre.clear()

    @property
    def level(self):
        """最近 3 秒的峰值帧 RMS —— 看麦克风有没有收到东西"""
        return max(self._rms_hist) if self._rms_hist else 0.0

    @property
    def level_now(self):
        return self._rms_hist[-1] if self._rms_hist else 0.0

    @property
    def floor(self):
        """当前快跟踪噪声值"""
        return self.last_floor

    @property
    def env_floor(self):
        """当前慢跟踪环境底（夹紧区间的基准）"""
        return self.noise_ref or 0.0

    @property
    def threshold(self):
        """当前生效门槛"""
        return self.last_threshold

    @property
    def state(self):
        return self._state


# ══════════════════════════════════════════════════════════════
#  自测：拿一批 wav 拼成连续流，看能不能正确切句
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    import wave
    from pathlib import Path

    def _load_wav(path):
        """标准库读 16bit PCM wav（构建环境没有 soundfile）"""
        with wave.open(str(path), "rb") as w:
            sr, ch = w.getframerate(), w.getnchannels()
            raw = w.readframes(w.getnframes())
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if ch > 1:
            a = a.reshape(-1, ch).mean(axis=1)
        return a, sr

    wavdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "wavs"
    files = sorted(wavdir.glob("*.wav"))[:8]
    if not files:
        print("没有 wav 可测")
        sys.exit(0)

    seg = Segmenter()
    total = 0
    t0 = time.time()
    for f in files:
        wav, sr = _load_wav(f)
        assert sr == SAMPLE_RATE, f"{f.name} sr={sr}"
        stream = np.concatenate([np.zeros(8000, np.float32), wav, np.zeros(16000, np.float32)])
        got = []
        for i in range(0, len(stream), 480):
            got += seg.push(stream[i:i + 480])
        print(f"  {f.name:44s} {len(wav)/16000:.2f}s → 切出 {len(got)} 句 "
              f"{[round(len(g)/16000,2) for g in got]}  门槛 {seg.threshold:.5f}")
        total += len(got)
    print(f"\n  共 {len(files)} 个文件 → 切出 {total} 句  ({time.time()-t0:.2f}s)")
    print(f"  VAD 后端: {'webrtcvad' if seg._vad else '能量阈值'}  "
          f"环境底 {seg.env_floor:.5f}  门槛 {seg.threshold:.5f}")
