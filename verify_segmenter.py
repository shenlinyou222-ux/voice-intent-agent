# -*- coding: utf-8 -*-
r"""离线复现 run_live.log 的现场，验证新分段判据（不需要麦克风）。

现场实测（2026-10-12 run_live.log）：
  · 房间噪声底 0.0006，帧 RMS 中位数 0.0001（大量精确 0）
  · 你的语音帧 RMS 约 0.0015~0.004
  · 旧配置：9 分钟切 63 段，76% 转写为空

三个场景（都按真实电平合成）：
  S1 安静房间   噪声 0.0002              + 语音缩放到 0.002
  S2 数字静音   60% 精确 0 + 40%×0.0003  + 语音缩放到 0.002   ← 门槛塌陷的坑（真机特征）
  S3 吵房间     噪声 0.006               + 语音缩放到 0.05    ← 检验会不会变迟钝

期望：每个场景只切出语音那一段。

用法: E:\voice_agent_build\Scripts\python.exe verify_segmenter.py
"""
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vad import Segmenter  # noqa: E402

SR = 16000


def load_wav(path):
    """标准库读 16bit PCM wav（构建环境没有 soundfile）"""
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2, f"只支持 16bit wav，实际 {w.getsampwidth()*8}bit"
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def build_stream(speech, kind):
    rng = np.random.default_rng(0)

    def zeros_noise(n):
        a = rng.normal(0, 0.0003, n).astype(np.float32)
        a[rng.random(n) < 0.6] = 0.0          # 60% 精确 0 —— 真机就是这样
        return a

    cfg = {
        "quiet": (lambda n: rng.normal(0, 0.0002, n).astype(np.float32), 0.002),
        "zeros": (zeros_noise, 0.002),
        "noisy": (lambda n: rng.normal(0, 0.006, n).astype(np.float32), 0.05),
    }[kind]
    noise, target = cfg
    s = speech / (np.sqrt((speech ** 2).mean()) + 1e-12) * target
    return np.concatenate([noise(5 * SR), s.astype(np.float32), noise(5 * SR)])


def feed(seg, stream):
    got = []
    for i in range(0, len(stream), 480):
        got += seg.push(stream[i:i + 480])
    tail = seg.flush()
    if tail is not None:
        got.append(tail)
    return got


def frame_peak(g):
    n = len(g) // 480 * 480
    if n == 0:
        return 0.0
    return float(np.max(np.sqrt((g[:n].reshape(-1, 480) ** 2).mean(axis=1))))


def run_case(name, stream, note, cal_noise):
    print("=" * 78)
    print(f"{name}   {note}")
    print("=" * 78)
    print("  ① 旧配置: webrtcvad(level 1) 开、门槛 max(0.008, floor*3)")
    old = Segmenter(min_speech_ms=400, energy_threshold=0.008, max_speech_ms=6000,
                    use_webrtcvad=True, aggressiveness=1, adaptive_floor=False)
    go = feed(old, stream)
    for g in go:
        rms = float(np.sqrt((g ** 2).mean()))
        print(f"      段 {len(g)/SR:5.2f}s  段平均RMS {rms:.5f}  峰值帧 {frame_peak(g):.5f}"
              f"{'   ← 纯噪声段' if rms < 0.001 else ''}")
    print(f"    → 旧配置切出 {len(go)} 段")

    print("  ② 新判据: 噪声跟踪(快降慢升) + 夹紧[噪声底×1.5, ×8] + 连续发声 240ms + 关 webrtcvad")
    new = Segmenter(min_speech_ms=400, max_speech_ms=6000, use_webrtcvad=False,
                    floor_mult=1.5, ceil_mult=8.0, voice_margin=2.0,
                    noise_floor=cal_noise, noise_down=0.25, noise_up=0.002,
                    min_run_ms=240)
    gn = feed(new, stream)
    for g in gn:
        rms = float(np.sqrt((g ** 2).mean()))
        print(f"      段 {len(g)/SR:5.2f}s  段平均RMS {rms:.5f}  峰值帧 {frame_peak(g):.5f}")
    print(f"    → 新判据切出 {len(gn)} 段"
          f"   噪声跟踪 {new.floor:.5f}  门槛 {new.threshold:.5f}"
          f"   连续发声 {new.last_run_ms:.0f}/{new.last_voiced_ms:.0f}ms")
    ok = len(gn) == 1
    print(f"    判定: {'✅ 只切出语音那一段' if ok else '❌ 切出 ' + str(len(gn)) + ' 段'}")
    print()
    return ok


def main():
    wavs = sorted((HERE / "wavs").glob("*launch*.wav"))[:1]
    if not wavs:
        print("没找到测试 wav")
        return 1
    speech, sr = load_wav(wavs[0])
    assert sr == SR, f"{wavs[0].name} sr={sr}"
    print(f"语音样本: {wavs[0].name}  {len(speech)/SR:.2f}s  "
          f"RMS {np.sqrt((speech**2).mean()):.4f}")
    print(f"合成流: 5s 噪声 + {len(speech)/SR:.2f}s 语音(按现场电平缩放) + 5s 噪声\n")

    results = []
    for kind, name, note, cal in (
        ("quiet", "S1 安静房间", "噪声 0.0002 / 语音 0.002", 0.0002),
        ("zeros", "S2 数字静音(真机特征)", "60% 精确 0 + 40%×0.0003 / 语音 0.002", 0.0002),
        ("noisy", "S3 吵房间", "噪声 0.006 / 语音 0.05", 0.006),
        ("quiet", "S4 标定被污染", "噪声 0.0002 / 语音 0.002，但开机标定被污染成 0.0024", 0.0024),
    ):
        results.append(run_case(name, build_stream(speech, kind), note, cal))

    print("=" * 78)
    print(f"汇总: {sum(results)}/{len(results)} 个场景通过"
          + ("   ✅ 全部通过" if all(results) else "   ❌ 有失败"))
    print("=" * 78)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
