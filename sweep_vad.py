# -*- coding: utf-8 -*-
r"""扫 floor_mult × min_run_ms，挑一组三个场景都过的值（数据定参，不拍脑袋）。

夹紧区间现在是 [噪声底×floor_mult, 噪声底×ceil_mult]，噪声底由"开机标定"给
（每个场景传自己的真实噪声，模拟标定）。ceil_mult 固定 8（只防门槛被噪声顶飞）。

用法: E:\voice_agent_build\Scripts\python.exe sweep_vad.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vad import Segmenter            # noqa: E402
from verify_segmenter import load_wav, build_stream, feed  # noqa: E402

SR = 16000

SCENES = [("quiet", "S1 安静0.0002", 0.0002),
          ("zeros", "S2 数字静音", 0.0002),
          ("noisy", "S3 吵0.006", 0.006)]

FLOOR_MULTS = [1.2, 1.5, 2.0, 3.0]
RUNS = [120, 180, 240]
MARGIN = 2.0
CEIL_MULT = 8.0


def main():
    wavs = sorted((HERE / "wavs").glob("*launch*.wav"))[:1]
    speech, sr = load_wav(wavs[0])
    assert sr == SR
    streams = {k: build_stream(speech, k) for k, _, _ in SCENES}

    print(f"{'floormult':>10} {'run_ms':>7} | "
          + " | ".join(f"{n:>18}" for _, n, _ in SCENES) + " | 结论")
    print("-" * 112)
    best = None
    for fm in FLOOR_MULTS:
        for run in RUNS:
            cells, okall = [], True
            for kind, name, nf in SCENES:
                seg = Segmenter(min_speech_ms=400, max_speech_ms=6000, use_webrtcvad=False,
                                floor_mult=fm, ceil_mult=CEIL_MULT, voice_margin=MARGIN,
                                noise_floor=nf, noise_down=0.25, noise_up=0.002,
                                min_run_ms=run)
                got = feed(seg, streams[kind])
                n = len(got)
                ok = (n == 1)
                okall &= ok
                cells.append(f"{'✅' if ok else '❌'} {n}段 跑{seg.last_run_ms:>4.0f}ms")
            mark = "  ⭐ 全过" if okall else ""
            print(f"{fm:>10} {run:>7} | " + " | ".join(cells) + f" |{mark}")
            if okall and best is None:
                best = (fm, run)
    print()
    if best:
        print(f"→ 建议 floor_mult={best[0]}  min_run_ms={best[1]}（第一组全过的）")
    else:
        print("→ 没有一组全过：需要进一步放宽（看各列失败原因）")
    return 0 if best else 1


if __name__ == "__main__":
    sys.exit(main())
