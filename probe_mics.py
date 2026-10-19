# -*- coding: utf-8 -*-
r"""逐设备实测麦克风电平：区分「设备哑了」还是「只是音量低」。

用法:
  E:\voice_agent_build\Scripts\python.exe probe_mics.py [每设备秒数]
  E:\voice_agent_build\Scripts\python.exe probe_mics.py --dev 23 --rate 16000 --secs 2

注意: 这里打印的 idx 是 sounddevice 真实索引，可直接给 voice_agent.exe --device N。
      voice_agent 自己打印的设备列表是「只留输入设备后重新编号」的，和真实索引不一致。
跑扫描时请正常说话，否则只能看到环境底噪。
"""
import sys

import numpy as np
import sounddevice as sd

MAX_PLAUSIBLE = 1.0  # 超过 1.0 视为 PortAudio 未初始化缓冲的垃圾值


def parse_args(argv):
    dev, rate, secs = None, None, 1.5
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--dev":
            dev = int(argv[i + 1]); i += 2
        elif a == "--rate":
            rate = int(argv[i + 1]); i += 2
        elif a == "--secs":
            secs = float(argv[i + 1]); i += 2
        else:
            secs = float(a); i += 1
    return dev, rate, secs


def grab(dev, rate, secs):
    rec = sd.rec(int(secs * rate), samplerate=rate, channels=1,
                 dtype="float32", device=dev, blocking=True)
    peak = float(np.max(np.abs(rec)))
    rms = float(np.sqrt(np.mean(rec.astype(np.float64) ** 2)))
    return peak, rms


def main():
    dev, rate, secs = parse_args(sys.argv[1:])
    print(f"sounddevice default (in,out) = {sd.default.device}")

    if dev is not None:
        d = sd.query_devices(dev)
        rate = rate or int(d["default_samplerate"]) or 16000
        print(f"单设备测试: [{dev}] {d['name']}  {rate}Hz  {secs}s")
        try:
            peak, rms = grab(dev, rate, secs)
        except Exception as e:  # noqa: BLE001
            print(f"  打开失败 {type(e).__name__}: {e}")
            return 1
        ok = "可用" if peak <= MAX_PLAUSIBLE and peak > 0.005 else "信号不足/异常"
        print(f"  peak={peak:.4f}  rms={rms:.5f}  -> {ok}")
        return 0

    devs = sd.query_devices()
    hostapis = sd.query_hostapis()
    in_idx = [i for i, d in enumerate(devs) if d["max_input_channels"] > 0]
    fixed = rate  # --rate 单独给出 => 所有设备都用该采样率(默认 16000 即 app 的采样率)
    if fixed:
        print(f"输入设备 {len(in_idx)} 个，统一用 {fixed}Hz 打开（app 固定 16000Hz），每设备 {secs}s。请说话。\n")
    else:
        print(f"输入设备 {len(in_idx)} 个，每设备 {secs}s（用各自原生采样率）。请说话。\n")

    rows = []
    for i in in_idx:
        d = devs[i]
        r = fixed or int(d["default_samplerate"]) or 16000
        ha = hostapis[d["hostapi"]]["name"]
        try:
            peak, rms = grab(i, r, secs)
        except Exception as e:  # noqa: BLE001
            print(f"[{i:2d}] {str(d['name'])[:38]:38s} {ha[:10]:10s} @{r:<6d} 打不开: {type(e).__name__}")
            continue
        rows.append((i, str(d["name"]), r, peak, rms, ha))

    print(f"\n{'idx':>3}  {'name':38s} {'hostapi':10s} {'rate':>6} {'peak':>9} {'rms':>9}  meter")
    good = []
    for i, name, r, peak, rms, ha in rows:
        bar = "#" * min(36, int(peak * 300)) if peak <= MAX_PLAUSIBLE else "!!"
        if peak > MAX_PLAUSIBLE:
            tag = "  (垃圾值/驱动不支持, 忽略)"
        elif peak >= 0.02:
            tag = "  <== 有信号"
            good.append((i, name, r, peak))
        elif peak >= 0.005:
            tag = "  (弱但可用)"
            good.append((i, name, r, peak))
        else:
            tag = "  (近乎静音)"
        print(f"{i:>3}  {name[:38]:38s} {ha[:10]:10s} {r:>6} {peak:>9.4f} {rms:>9.5f}  {bar}{tag}")

    if good:
        b = max(good, key=lambda x: x[3])
        print(f"\n最强真实信号: [{b[0]}] {b[1]} ({b[2]}Hz)  peak={b[3]:.4f}")
        print(f"建议启动: voice_agent.exe --device {b[0]}")
    else:
        print("\n没有任何设备有信号 —— 检查麦克风是否被系统静音/禁用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
