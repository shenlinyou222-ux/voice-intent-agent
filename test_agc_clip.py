# -*- coding: utf-8 -*-
"""复现并验证"常开监听听不到人说话"的真凶：AGC 顶到 ×100 → 削波 → Silero 判 0。

现场数据（12:41 运行日志）：
    电平 0.1302  AGC×99.3  VAD概率 0.000
    电平 0.2179  AGC×100.0 VAD概率 0.000
即：增益顶到上限、输入被削平，VAD 概率是 0 —— 40 秒一句都没切出来。

本脚本把一句 **真实语音** 衰减到本机麦克风的真实量级，再叠上实测噪声底，
分别用 旧上限(100) / 新上限(30)+limiter 跑同一个 Silero 引擎，对比：
  · 有没有切出语音段
  · 段的峰概率
  · 有没有削波（|x| 撞 1.0 的比例）
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vad_silero import SileroVADEngine          # noqa: E402
from model_runner import read_wav               # noqa: E402

SR = 16000
BLOCK = 480
RAW_SPEECH_RMS = 0.0015     # 本机麦克风说话时的实测 RMS
RAW_NOISE_RMS = 0.0003      # 本机麦克风噪声底实测 RMS


def make_scene(wav):
    """把语音缩到本机实测电平 + 前置 1s 静音（让 AGC 先收敛到稳态）"""
    x = np.asarray(wav, dtype=np.float32)
    x = x / (np.sqrt((x ** 2).mean()) + 1e-12) * RAW_SPEECH_RMS
    rng = np.random.default_rng(0)
    lead = rng.normal(0, RAW_NOISE_RMS, SR).astype(np.float32)
    tail = rng.normal(0, RAW_NOISE_RMS, int(SR * 0.8)).astype(np.float32)
    return np.concatenate([lead, x, tail])


def run(wav, max_gain, db_threshold=60):
    eng = SileroVADEngine(prob_threshold=0.4, db_threshold=db_threshold,
                          required_hits=3, required_misses=24,
                          target_rms=0.03, max_gain=max_gain)
    segs, probs, clipped, gains, dbs = [], [], 0, [], []
    n = 0
    for i in range(0, len(wav) - BLOCK + 1, BLOCK):
        blk = wav[i:i + BLOCK]
        segs += eng.push(blk)
        probs.append(eng.last_prob)
        dbs.append(eng.last_db)
        g = eng.gain_now
        gains.append(g)
        clipped += int(np.sum(np.abs(blk * g) > 1.0))
        n += blk.size
    tail = eng.flush()
    if tail is not None:
        segs.append(tail)
    return {
        "max_gain": max_gain,
        "db_threshold": db_threshold,
        "segs": len(segs),
        "seg_sec": [round(len(s) / SR, 2) for s in segs],
        "prob_max": float(np.max(probs)),
        "db_speech": float(np.percentile(dbs, 90)),
        "gain_final": round(float(gains[-1]), 1),
        "clip_pct": 100.0 * clipped / max(n, 1),
    }


def main():
    wavs = sorted(Path("wavs").glob("*.wav"))
    if not wavs:
        print("❌ wavs/ 下没有 wav")
        return 1
    print(f"语音素材: {wavs[0].name}")
    scene = make_scene(read_wav(str(wavs[0])))
    print(f"场景: 前置 1s 噪声底(RMS {RAW_NOISE_RMS}) + 语音缩到 RMS "
          f"{RAW_SPEECH_RMS} + 尾随静音")
    print("判据（照抄状态机）: hit = 平滑prob>=0.4 AND 平滑db>=db_threshold，"
          "其中 db=20log10(RMS(chunk*32767))\n")

    print(f"{'增益上限':>8} {'db门限':>7} {'段数':>5} {'段长(s)':>12} "
          f"{'峰概率':>7} {'语音db(p90)':>11} {'末增益':>7} {'削波%':>6}")
    print("-" * 76)
    res = []
    for db_thr in (60, 40):
        for mg in (100.0, 30.0):
            r = run(scene, mg, db_thr)
            res.append(r)
            print(f"{r['max_gain']:>8.0f} {r['db_threshold']:>7} {r['segs']:>5} "
                  f"{str(r['seg_sec']):>12} {r['prob_max']:>7.3f} "
                  f"{r['db_speech']:>11.1f} {r['gain_final']:>7.1f} "
                  f"{r['clip_pct']:>6.2f}")

    def get(mg, db_thr):
        return next(r for r in res
                    if r["max_gain"] == mg and r["db_threshold"] == db_thr)

    print()
    old60 = get(100.0, 60)
    new40 = get(30.0, 40)
    ok = True
    print(f"已知现场（×100 / 门限60）: 削波 {old60['clip_pct']:.2f}%、"
          f"峰概率 {old60['prob_max']:.3f}、{old60['segs']} 段")
    # 关键判据：门限 60 时即使 prob 很高也可能切不出（电平门限拦掉）
    g60 = get(30.0, 60)
    if g60["prob_max"] >= 0.9 and g60["segs"] == 0:
        print(f"✅ 复现真凶：门限 60 下 prob 高达 {g60['prob_max']:.3f}，"
              f"却切出 {g60['segs']} 段 —— 被【电平门限】拦掉"
              f"（语音 db={g60['db_speech']:.1f} < 60）")
    if new40["segs"] >= 1 and new40["prob_max"] >= 0.4:
        print(f"✅ 修复有效（×30 / 门限40）: 切出 {new40['segs']} 段 "
              f"{new40['seg_sec']}，峰概率 {new40['prob_max']:.3f}，"
              f"语音 db={new40['db_speech']:.1f}，削波 {new40['clip_pct']:.2f}%")
    else:
        print(f"❌ 仍然切不出：{new40['segs']} 段、峰概率 {new40['prob_max']:.3f}")
        ok = False

    ok &= integration_from_config(scene)
    return 0 if ok else 1


def integration_from_config(scene):
    """回归测试（这一条才真正守住线上路径）：
       用 intent_config.json 里的参数、走 VADFactory（与 Agent 完全同路径）造引擎，
       ① 静音场景里的语音 → 必须切出段
       ② 同样电平的【噪声】 → 必须一段都不切（降门限不能把噪声放进来）
    """
    import json
    from vad_silero import VADFactory
    print("\n=== 集成回归（走配置 + VADFactory，与 Agent 同路径）===")
    cfg = json.load(open("intent_config.json", encoding="utf-8"))["behavior"]
    kw = dict(target_rms=float(cfg.get("vad_target_rms", 0.03)),
              max_gain=float(cfg.get("vad_max_gain", 30.0)),
              prob_threshold=float(cfg.get("vad_prob_threshold", 0.4)),
              db_threshold=float(cfg.get("vad_db_threshold", 40)),
              required_hits=int(cfg.get("vad_required_hits", 3)),
              required_misses=int(cfg.get("vad_required_misses", 24)),
              max_speech_ms=int(cfg.get("vad_max_speech_ms", 12000)))
    print(f"  配置: {kw}")

    ok = True

    def play(eng, sig):
        segs = []
        for i in range(0, len(sig) - BLOCK + 1, BLOCK):
            segs += eng.push(sig[i:i + BLOCK])
        t = eng.flush()
        if t is not None:
            segs.append(t)
        return segs

    eng = VADFactory.get_vad_engine("silero_vad", **kw)
    segs = play(eng, scene)
    p = eng.last_seg_prob
    if segs:
        print(f"  ① 语音场景 → {len(segs)} 段 "
              f"{[round(len(s)/SR, 2) for s in segs]}s，段峰概率 {p:.3f}  ✅")
    else:
        print(f"  ① 语音场景 → 0 段  ❌ 又听不到人说话了（峰概率 {p:.3f}）")
        ok = False

    # ② 噪声：与现场实测同电平（噪声底 0.0003 的白噪，以及响一点的 0.0022）
    rng = np.random.default_rng(1)
    for raw in (RAW_NOISE_RMS, 0.0022):
        sig = rng.normal(0, raw, SR * 8).astype(np.float32)
        eng2 = VADFactory.get_vad_engine("silero_vad", **kw)
        ns = play(eng2, sig)
        tag = "✅" if not ns else "❌ 噪声被切成了段"
        if ns:
            ok = False
        print(f"  ② 纯噪声(RMS {raw}) → {len(ns)} 段，峰概率 "
              f"{eng2.last_seg_prob:.3f}  {tag}")
    return ok


if __name__ == "__main__":
    sys.exit(main())
