# -*- coding: utf-8 -*-
"""麦克风读取速率探针（不需要人说话）—— 定位"按住 10 秒只攒到 0.99s"。

背景：`_raw_rec.py` / `_ptt_demo.py` 用同样的
      sd.InputStream(16000, 1ch, blocksize=480, device=1) 是正常的，
      而 `_dual_live.py`（多了一个 worker 线程 + deque）拿到 ~3.3 块/秒（应 33 块/秒，差 10 倍）。
      跑这个探针就能知道：是设备/参数问题，还是脚本结构问题。

对每个候选配置跑 TEST_S 秒，数块数、算速率、看电平（有声/全零）。
"""
import sys
import time
from pathlib import Path

import numpy as np

SR, BLOCK, TEST_S = 16000, 480, 5.0


def probe(sd, dev, sr, block, tag, extra=None):
    kw = dict(samplerate=sr, channels=1, dtype="float32", blocksize=block, device=dev)
    if extra:
        kw.update(extra)
    try:
        with sd.InputStream(**kw) as st:
            n, peak, t0 = 0, 0.0, time.time()
            while time.time() - t0 < TEST_S:
                blk, ovf = st.read(block)
                n += 1
                peak = max(peak, float(np.abs(blk).max()))
            el = time.time() - t0
        rate = n * (block / (sr or SR)) / el
        print(f"  {tag:<46} 块 {n:>5}  实测 {n/el:6.1f} 块/s  "
              f"音频 {rate:5.2f}x 实时  峰值 {peak:.4f}")
        return n / el
    except Exception as e:
        print(f"  {tag:<46} ❌ {type(e).__name__}: {str(e)[:60]}")
        return -1


def main():
    import sounddevice as sd
    print("\n  === 输入设备清单（名字 / 声道 / 默认采样率）===")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"    [{i:>2}] {d['name'][:44]:<44} {d['max_input_channels']}ch  "
                  f"默认 {d['default_samplerate']:.0f}Hz")
    print(f"\n  默认 host API: {sd.query_hostapis(sd.default.hostapi)['name']}")
    print(f"\n  === 逐配置测 {TEST_S:.0f} 秒（应约 {SR/BLOCK:.1f} 块/s = 1.00x 实时）===")
    probe(sd, 1, SR, BLOCK, "device=1  16000  block=480（现在用的）")
    probe(sd, 1, SR, BLOCK, "device=1  16000  block=480  latency=low", {"latency": "low"})
    probe(sd, 1, SR, BLOCK, "device=1  16000  block=480  latency=high", {"latency": "high"})
    probe(sd, 1, SR, 0, "device=1  16000  block=0（让 PortAudio 自己定）")
    probe(sd, 1, 48000, 1440, "device=1  48000  block=1440（原生 48k）")
    # 按名字找同名设备，逐个试
    names = {}
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and "USBAudio2.0" in d["name"]:
            names[i] = d
    print(f"\n  同名设备（USBAudio2.0）共 {len(names)} 个：{sorted(names)}")
    for i in sorted(names):
        probe(sd, i, SR, BLOCK, f"device={i} 16000 block=480")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
