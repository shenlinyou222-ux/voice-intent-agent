# -*- coding: utf-8 -*-
"""长时间原始录音（只采不判）—— 放在后台，用户照常用 app 说话，
   事后再拿 utt/index.jsonl 的时间戳跟这段原始音频对齐（不依赖"掐点说话"）。

用法：python -X utf8 _raw_rec.py --sec 240 --device 1
输出：_cutprobe/raw.wav + _cutprobe/meta.json（含 start_epoch / start_iso，用于对齐）
"""
import argparse
import datetime as _dt
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", type=float, default=240)
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "_cutprobe"))
    ap.add_argument("--blocksize", type=int, default=480)
    a = ap.parse_args()

    import sounddevice as sd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = a.device
    print(f"  设备 {dev}：{sd.query_devices(dev)['name']}   录 {a.sec:.0f}s", flush=True)
    print("  ⏺ 开始（照常用 app 说话即可，不用管这个窗口）\n", flush=True)

    t_start = time.time()
    start_iso = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chunks, loud = [], []
    last = 0.0
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=a.blocksize, device=dev) as st:
        while time.time() - t_start < a.sec:
            blk, _ = st.read(a.blocksize)
            x = np.asarray(blk[:, 0], dtype=np.float32).copy()
            chunks.append(x)
            loud.append(float(np.sqrt((x ** 2).mean())))
            now = time.time()
            if now - last > 2.0:
                last = now
                q = loud[-67:]                       # 最近 ~2s
                lv = max(q) if q else 0.0
                print(f"     {now-t_start:6.1f}s  峰值块RMS {lv:.5f} "
                      f"{'█' * min(30, int(lv * 3000))}", flush=True)
    x = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    with wave.open(str(out / "raw.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    meta = {"device": dev, "device_name": sd.query_devices(dev)["name"],
            "start_epoch": t_start, "start_iso": start_iso,
            "sec": len(x) / SR, "peak": float(np.abs(x).max()),
            "rms": float(np.sqrt((x ** 2).mean())),
            "loud_block_rms_max": float(np.max(loud)) if loud else 0.0}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"\n  💾 {out/'raw.wav'}  {meta['sec']:.1f}s  峰值 {meta['peak']:.5f}  "
          f"最响块 RMS {meta['loud_block_rms_max']:.5f}")
    print(f"  💾 起始时刻 {start_iso}（对齐 utt/index.jsonl 用）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
