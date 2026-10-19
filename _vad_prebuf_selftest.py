# -*- coding: utf-8 -*-
"""VAD 预滚动修复的前后对照 —— 用【真实音频】复现「吃掉首字」并证明修好了。

素材：utt/162212_mic_dictation_感觉好像可以了吧.wav（真人语音，就是那句「感」被吃掉的）
流：  [1s 静音][前半句][0.72s 停顿][后半句][1s 静音]
      —— 0.72s > 22 块判停门限（704ms），VAD 必然在这一刀切断，
         而"后半句"距上一段收段只有几块 → 老代码的预滚动被 clear 过、来不及攒回来。

对照：
  OLD = 收段时 pre_buffer.clear() 后再补 1 块（修复前的 vad_silero.StateMachine）
  NEW = pre_buffer 一直在滚、收段不清空（现在的 vad_silero.StateMachine）
判据：后半段的段起点相对"真实起音"是【之前】还是【之后】——
      在之后就是切进来了（会吃掉前一两个字），以及两段的转写差在哪。
"""
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from vad_silero import PreGain, SileroVAD, StateMachine, WINDOW, SR  # noqa: E402
from _cut_align import asr  # noqa: E402

GAP_S = 0.72
SRC = HERE / "utt" / "162212_mic_dictation_感觉好像可以了吧.wav"


class OldSM(StateMachine):
    """修复前的行为：收段时清空 pre_buffer，之后每块只补 1 块；IDLE 里才 append"""

    def process(self, prob, chunk):
        int_chunk = chunk * 32767
        db = self.calculate_db(int_chunk)
        sp, sd = self._smoothed(prob, db)
        hit = (sp >= self.prob_threshold and sd >= self.db_threshold)
        seg = None
        if self.state == State.IDLE:
            self.pre_buffer.append(chunk)
            if hit:
                self.hit_count += 1
                if self.hit_count >= self.required_hits:
                    self.state = State.ACTIVE
                    self.reset_buffers()
                    for c in self.pre_buffer:
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
            self.pre_buffer.clear()
            self.state = State.IDLE
            self.pre_buffer.append(chunk)
        return seg


# State 是 vad_silero 内部的枚举，OldSM 里要用到
from vad_silero import State  # noqa: E402


def load(p):
    with wave.open(str(p), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def locate(seg, x):
    """归一化互相关定位：返回段的起点秒数"""
    a = np.asarray(seg, np.float64)
    a = a - a.mean()
    na = np.linalg.norm(a)
    best, bl = -1e9, 0
    for lag in range(0, len(x) - len(seg) + 1, 160):
        b = x[lag:lag + len(seg)].astype(np.float64)
        b = b - b.mean()
        d = na * np.linalg.norm(b)
        if d <= 0:
            continue
        c = float(np.dot(a, b) / d)
        if c > best:
            best, bl = c, lag
    return bl / SR, best


def main():
    sp = load(SRC)
    half = len(sp) // 2
    sil = lambda s: np.zeros(int(s * SR), np.float32)          # noqa: E731
    stream = np.concatenate([sil(1.0), sp[:half], sil(GAP_S), sp[half:], sil(1.0)])
    print(f"\n  素材 {SRC.name}  {len(sp)/SR:.2f}s → 流 {len(stream)/SR:.2f}s"
          f"（静音 1.0 + 前半 {half/SR:.2f} + 停顿 {GAP_S} + 后半 "
          f"{(len(sp)-half)/SR:.2f} + 静音 1.0）")

    # 概率时间线只算一次（两个状态机吃同一串 prob，差别只在状态机本身）
    g = PreGain(target_rms=0.03, max_gain=8.0)
    g.gain = 8.0
    vad = SileroVAD()
    chunks, probs = [], []
    for i in range(0, len(stream) - WINDOW + 1, WINDOW):
        c = g.apply(stream[i:i + WINDOW])
        chunks.append(c)
        probs.append(vad.prob(c))
    print(f"  块数 {len(probs)}   prob max {max(probs):.3f}  "
          f"≥0.4 的块数 {sum(1 for p in probs if p >= 0.4)}")

    # 真实起音（不依赖状态机）：安静后 prob 首次 ≥0.5
    onsets, run = [], 0
    for i, p in enumerate(probs):
        if p < 0.2:
            run += 1
        else:
            if p >= 0.5 and run >= 10:
                onsets.append(i)
            run = 0
    onset_s = [round(i * WINDOW / SR, 3) for i in onsets]
    print(f"  独立起音（prob≥0.5）时刻：{onset_s}")

    # 用【生产参数】而不是类默认值（默认是 db=60/hits=3/misses=24，配置里是 40/2/22）
    P = dict(prob_threshold=0.4, db_threshold=40, required_hits=2, required_misses=22)
    for name, sm in (("OLD（清空预滚动）", OldSM(**P)),
                     ("NEW（预滚动一直在滚）", StateMachine(**P))):
        # 精确记账：段 = chunks 里最后 n 块（预滚动 + ACTIVE 块，天然连续），
        # 收段发生在 INACTIVE 那一次调用，此时 len(sm.chunks) 还没被清。
        segs = []
        for i, p in enumerate(probs):
            before = len(sm.chunks)
            s = sm.process(p, chunks[i])
            if s is not None and len(s):
                segs.append((i - before, i - 1, s))
        print(f"\n  ── {name} ──  切出 {len(segs)} 段")
        for k, (a, b, s) in enumerate(segs):
            t0, t1 = a * WINDOW / SR, (b + 1) * WINDOW / SR
            pre = stream[max(0, a * WINDOW - 3200):a * WINDOW]
            h = 3200
            nw = max(1, len(s) // h)
            pk = max(float(np.sqrt((s[i * h:(i + 1) * h] ** 2).mean())) for i in range(nw))
            pr = float(np.sqrt((pre ** 2).mean())) if len(pre) else 0.0
            head50 = float(np.sqrt((s[:800] ** 2).mean())) if len(s) >= 800 else 0.0
            rms = float(np.sqrt((s ** 2).mean()))
            cand = [o for o in onset_s if t0 - 2.5 <= o <= t0 + 0.1]
            o = max(cand) if cand else None
            d = (t0 - o) * 1000 if o is not None else None
            verdict = ("?" if d is None else
                       (f"✓ 段前还留 {-d:.0f}ms" if d <= 0 else f"✗ 切进来 {d:.0f}ms"))
            try:
                txt = asr(s, gain=1.0)
            except Exception as e:
                txt = f"❌{type(e).__name__}"
            print(f"     段{k}: 块{a}~{b} = {t0:.2f}~{t1:.2f}s（{len(s)/SR:.2f}s）  "
                  f"首50ms/段RMS={head50/max(rms,1e-9):5.2f}  "
                  f"段前200/段最响={pr/max(pk,1e-9):5.2f}  {verdict}")
            print(f"            转写：{txt!r}")
        if len(segs) >= 2:
            a0, b0, _ = segs[0]
            a1, b1, _ = segs[1]
            print(f"     → 第 2 段起点块 {a1}（{a1*WINDOW/SR:.2f}s）"
                  f"，比第 1 段收段块({b0}) 晚 {a1-b0} 块")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
