# -*- coding: utf-8 -*-
"""切断定位 · 第一半：同采一条流
  · 一边把【原始音频】（不做任何 AGC/增益）原封不动写盘
  · 一边喂【生产同款】Silero 分句引擎（vad_silero.SileroVADEngine，参数取自 intent_config.json）
这样"真实起音"和"段起点"来自同一段音频，可以逐样本对齐；事后由 _cut_report.py 出结论。

用法：
  E:\\voice_agent_build\\Scripts\\python.exe -X utf8 _cut_probe.py --sec 25 --device 1
停止条件：录满 --sec 秒，或（切出 ≥3 段 且 最后一段结束后又过了 2 秒）。
"""
import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from vad_silero import SileroVADEngine, SR, WINDOW  # noqa: E402

CFG = json.loads((HERE / "intent_config.json").read_text(encoding="utf-8"))
B = CFG["behavior"]


def make_engine():
    """生产同款（voice_agent.py:150-161 的那一组参数）"""
    return SileroVADEngine(
        target_rms=float(B.get("vad_target_rms", 0.03)),
        max_gain=float(B.get("vad_max_gain", 30.0)),
        prob_threshold=float(B.get("vad_prob_threshold", 0.4)),
        db_threshold=float(B.get("vad_db_threshold", 60)),
        required_hits=int(B.get("vad_required_hits", 3)),
        required_misses=int(B.get("vad_required_misses", 24)),
        max_speech_ms=int(B.get("vad_max_speech_ms", 12000)),
    )


def bar(v, span=30, scale=600.0):
    return "█" * min(span, int(v * scale))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", type=float, default=25.0, help="最长录制秒数")
    ap.add_argument("--device", type=int, default=1, help="输入设备号（默认 1 = app 在用的那个）")
    ap.add_argument("--out", default=str(HERE / "_cutprobe"))
    ap.add_argument("--blocksize", type=int, default=480, help="必须和 app 一致（30ms）")
    a = ap.parse_args()

    import sounddevice as sd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wav_path = out / "raw.wav"

    print("  可用输入设备：")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"    [{i}] {d['name']}  ({d['max_input_channels']}ch)")
    dev = a.device
    print(f"\n  使用设备 {a.device}：{sd.query_devices(dev)['name']}")
    print(f"  生产 VAD 参数：prob≥{B.get('vad_prob_threshold')} db≥{B.get('vad_db_threshold')} "
          f"hits={B.get('vad_required_hits')} misses={B.get('vad_required_misses')} "
          f"AGC target={B.get('vad_target_rms')} max_gain={B.get('vad_max_gain')}")
    print(f"  预滚动 pre_buffer = 20 块 × 512 点 = 640ms（vad_silero.StateMachine 默认值）")

    eng = make_engine()
    raw = []
    segs = []            # [(n_samples_before_seg_ended, seg_len)]
    fed = 0
    seg_open = None

    print(f"\n  ⏺ 录 {a.sec:.0f}s（录满或切出 3 段后 2s 自动停）")
    for k in (3, 2, 1):
        print(f"     {k} ...", flush=True)
        time.sleep(0.7)
    print("     🎤 请开始说话（正常语速说两句就行）\n", flush=True)

    t0 = time.time()
    last_seg_t = None
    last_print = 0.0
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=a.blocksize, device=dev) as st:
        while time.time() - t0 < a.sec:
            blk, ovf = st.read(a.blocksize)
            x = np.asarray(blk[:, 0], dtype=np.float32).copy()
            raw.append(x)
            fed += len(x)
            try:
                for s in eng.push(x):
                    segs.append((fed, len(s), float(np.sqrt((np.asarray(s) ** 2).mean()))))
                    last_seg_t = time.time()
                    print(f"     📥 切出第 {len(segs)} 段：{len(s)/SR:.2f}s  "
                          f"引擎输出 RMS {float(np.sqrt((np.asarray(s)**2).mean())):.5f}", flush=True)
            except Exception as e:
                print(f"     ⚠️ 分句异常: {type(e).__name__}: {e}", flush=True)
            now = time.time()
            if now - last_print > 0.5:
                last_print = now
                raw_lv = float(np.sqrt((x ** 2).mean()))
                print(f"     {now-t0:5.1f}s  原始RMS {raw_lv:.5f} {bar(raw_lv, 24, 400):<24s} "
                      f"AGC×{eng.gain_now:.1f}  段={len(segs)}", flush=True)
            if len(segs) >= 3 and last_seg_t and time.time() - last_seg_t > 2.0:
                print("     ✅ 已够（3 段 + 2s 尾巴），提前结束", flush=True)
                break

    tail = np.concatenate(raw) if raw else np.zeros(0, np.float32)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(tail, -1, 1) * 32767).astype("<i2").tobytes())
    meta = {"device": dev, "device_name": sd.query_devices(dev)["name"],
            "sec": len(tail) / SR, "blocksize": a.blocksize,
            "n_segments": len(segs),
            "seg_ends_samples": [int(s[0]) for s in segs],
            "seg_lens": [int(s[1]) for s in segs],
            "seg_rms": [round(s[2], 6) for s in segs],
            "raw_peak": float(np.abs(tail).max()),
            "raw_rms": float(np.sqrt((tail ** 2).mean())),
            "cfg": {k: B.get(k) for k in
                    ("vad_prob_threshold", "vad_db_threshold", "vad_required_hits",
                     "vad_required_misses", "vad_target_rms", "vad_max_gain",
                     "vad_max_speech_ms")}}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"\n  💾 {wav_path}  {len(tail)/SR:.2f}s  峰值 {meta['raw_peak']:.5f}  "
          f"RMS {meta['raw_rms']:.5f}")
    print(f"  💾 {out/'meta.json'}  切出 {len(segs)} 段")
    print(f"\n  下一步：跑 _cut_report.py 出「段头丢了多少毫秒语音」的结论")
    return 0


if __name__ == "__main__":
    sys.exit(main())
