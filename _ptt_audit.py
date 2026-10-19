# -*- coding: utf-8 -*-
"""PTT 段审计：把每一段【原始 / 处理后】摊开看，并用 ASR 做"挖掉头/挖掉尾"对照，
   定位到底是哪一头、哪一环把话吞了。

用法：python -X utf8 _ptt_audit.py [_ptt]
"""
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _cut_align import asr  # noqa: E402

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "_ptt"
BARS = " .:-=+*#%@"


def load(p):
    with wave.open(str(p), "rb") as w:
        return (np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
                / 32768.0), w.getframerate()


def env(x, h=320):                                  # 20ms
    n = len(x) // h
    return np.sqrt((x[:n * h].reshape(n, h) ** 2).mean(axis=1)) if n else np.zeros(0)


def spark(e, cols=64):
    if not len(e):
        return ""
    idx = np.linspace(0, len(e), cols + 1).astype(int)
    v = np.array([e[a:b].max() if b > a else 0.0 for a, b in zip(idx[:-1], idx[1:])])
    m = float(v.max()) or 1e-9
    return "".join(BARS[min(9, int(q / m * 9.999))] for q in v)


def speech_span(e, ratio=6.0, floor=0.0008):
    """返回 (起, 止) 的 20ms 帧下标。

    环境底取整段帧能量的 p10 —— **不能**假设"段头就是静音"：
    按下就开口时段头就是语音，用段头估环境底会把阈值抬到语音之上、把话裁掉。
    """
    if not len(e):
        return -1, -1
    amb = float(np.percentile(e, 10))
    thr = max(amb * ratio, floor)
    idx = np.where(e >= thr)[0]
    return (int(idx[0]), int(idx[-1])) if len(idx) else (-1, -1)


files = sorted(DIR.glob("*.wav"))
if not files:
    print(f"  {DIR} 里没有 wav")
    sys.exit(0)
print(f"\n  {DIR}   共 {len(files)} 个")
for f in files:
    x, sr = load(f)
    e = env(x)
    a, b = speech_span(e)
    print(f"\n  ══ {f.name}  {len(x)/sr:.2f}s  峰值 {np.abs(x).max():.5f}  "
          f"RMS {np.sqrt((x**2).mean()):.5f}  帧数 {len(e)}(20ms)")
    print(f"     |{spark(e)}|   (按整段峰值归一)")
    if a >= 0:
        print(f"     首个像说话的帧 {a*20}ms   末个 {b*20}ms   "
              f"→ 段头静音 {a*20}ms   段尾静音 {(len(e)-1-b)*20}ms")
        print(f"     头 20 帧包络(20ms/格)：{' '.join(f'{v:.4f}' for v in e[:20])}")
    else:
        print("     整段都没超过阈值（全是静音/极轻）")
    # ASR 对照：整段 / 挖头500ms / 挖尾500ms / 换增益
    variants = {"整段": x, "挖头500ms": x[8000:], "挖尾500ms": x[:-8000] if len(x) > 8000 else x}
    for name, seg in variants.items():
        if len(seg) < sr // 4:
            continue
        try:
            t1 = asr(seg, sr=sr, gain=1.0)
        except Exception as ex:
            t1 = f"❌{type(ex).__name__}"
        try:
            t8 = asr(seg, sr=sr, gain=8.0)
        except Exception as ex:
            t8 = f"❌{type(ex).__name__}"
        print(f"     {name:<10} ×1: {t1!r}")
        print(f"     {'':<10} ×8: {t8!r}")
print()
