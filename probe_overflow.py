# -*- coding: utf-8 -*-
r"""量清麦克风缓冲有多深：停止读取多久才会 overflow（丢音频）

这决定了「加载期间阻塞主循环」到底会不会丢声音。
用法: E:\voice_agent_build\Scripts\python.exe probe_overflow.py
"""
import sys
import time

import numpy as np
import sounddevice as sd

SR, BLK = 16000, 480          # 30ms 一块，跟 voice_agent 一致
dev = sd.default.device[0]
print(f"设备 {dev}：{sd.query_devices(dev)['name']}  {SR}Hz  block={BLK}(30ms)")

# ① 先找"多久不读就溢出"
with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                    blocksize=BLK, device=dev) as st:
    t0 = time.time()
    while time.time() - t0 < 2.0:          # 先正常抽 2 秒
        st.read(BLK)
    print("\n① 停止读取，逐档试探何时开始 overflow：")
    for pause in (0.2, 0.5, 1.0, 2.0, 3.0, 5.0):
        time.sleep(pause)
        ovf = 0
        t1 = time.time()
        n = 0
        while time.time() - t1 < 1.0:
            d, o = st.read(BLK)
            n += 1
            ovf += bool(o)
        print(f"   停读 {pause:>4.1f}s → 随后 1 秒里读到 {n:3d} 块"
              f"（理论 33 块），overflow 标记 {ovf} 次")

# ② 连续读时会不会溢出（对照组）
with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                    blocksize=BLK, device=dev) as st:
    t0 = time.time()
    n = ovf = 0
    while time.time() - t0 < 3.0:
        d, o = st.read(BLK)
        n += 1
        ovf += bool(o)
    print(f"\n② 连续读 3 秒：读到 {n} 块（理论 100），overflow {ovf} 次")
