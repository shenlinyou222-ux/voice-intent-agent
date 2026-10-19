# -*- coding: utf-8 -*-
"""回归测试：handle() 在 Silero 后端下不再崩（复现 12:29:50 的自动退出）。

崩溃现场：真人第一段语音进 handle() →
    AttributeError: 'SileroVADEngine' object has no attribute 'frame_len'
→ 异常穿出 run_mic → 整个常开进程退出。

本测试**直接调真的 Agent.handle()**（模型用假预测器替身，dry-run），
分句后端分别用 Silero / 老 Segmenter，两边都必须正常跑完。
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config                          # noqa: E402
import voice_agent as va                            # noqa: E402
from model_runner import read_wav                   # noqa: E402
from vad_silero import SileroVADEngine              # noqa: E402
from vad import Segmenter                           # noqa: E402

SR = 16000
BLOCK = 480


class FakeModel:
    """替身：handle() 只用到 predict() 的返回值"""

    INTENTS = ["unknown", "open_url", "launch_app", "open_file"]

    def __init__(self):
        self.calls = 0

    def predict(self, wav, text_override=None):
        self.calls += 1
        return {"text": text_override or "打开计算器",
                "intent": "launch_app", "conf": 0.97, "latency_ms": 12.3}


def make_agent(cfg_path, vad_backend):
    cfg = Config(cfg_path)
    cfg.behavior["vad_backend"] = vad_backend
    cfg.behavior["dry_run"] = True
    a = va.Agent(cfg, model_dir=str(Path(cfg_path).parent / "model"), dry=True)
    a.lazy = False                 # 不要真去装 392MB 模型
    a.asr_backend = "ctc"          # 跳过 ASR 分支
    a.model = FakeModel()
    return a


def run_one(tag, agent, wav):
    print(f"\n[{tag}] seg={type(agent.seg).__name__}")
    try:
        agent.handle(np.asarray(wav, dtype=np.float32))
    except Exception as e:
        import traceback
        print(f"  ❌ handle() 抛异常（这正是导致 App 自动退出的路径）: "
              f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    print(f"  ✅ handle() 正常返回；统计 {agent.stats}")
    return True


def main():
    roots = [Path("."), Path("dist/voice_agent")]
    cfg_path = None
    for r in roots:
        p = r / "intent_config.json"
        if p.exists():
            cfg_path = str(p)
            break
    if cfg_path is None:
        print("❌ 找不到 intent_config.json")
        return 1
    print(f"配置: {cfg_path}")

    wav_file = sorted(Path("wavs").glob("*.wav"))
    if not wav_file:
        print("❌ wavs/ 下没有 wav")
        return 1
    wav = read_wav(str(wav_file[0]))
    print(f"测试音频: {wav_file[0].name} ({len(wav)/SR:.2f}s)")

    ok = True

    # ── 1) 用真的分句引擎跑一遍，拿到"真实形状"的段 ──
    print("\n=== A. 真的走一遍分句（Silero 引擎） ===")
    eng = SileroVADEngine(prob_threshold=0.4, db_threshold=60, required_hits=3,
                          required_misses=24, target_rms=0.03)
    segs = []
    for i in range(0, len(wav) - BLOCK, BLOCK):
        segs += eng.push(wav[i:i + BLOCK])
    tail = eng.flush()
    if tail is not None:
        segs.append(tail)
    print(f"  Silero 分出 {len(segs)} 段；last_seg_prob={eng.last_seg_prob:.3f} "
          f"frame_len={eng.frame_len}")
    if not segs:
        print("  （没出段，用整段音频代替——诊断代码路径一样）")
        segs = [wav]

    # ── 2) 崩溃路径：Silero 后端 + handle() ──
    print("\n=== B. 崩溃主路径：Silero 后端调真 handle() ===")
    a_sil = make_agent(cfg_path, "silero")
    assert isinstance(a_sil.seg, SileroVADEngine), type(a_sil.seg)
    for k, s in enumerate(segs):
        print(f"  --- 第 {k+1} 段 ({len(s)/SR:.2f}s) ---")
        ok &= run_one("silero", a_sil, s)

    # ── 3) 反向：老能量后端不能被改坏 ──
    print("\n=== C. 反向回归：老 energy 后端 ===")
    a_en = make_agent(cfg_path, "energy")
    assert isinstance(a_en.seg, Segmenter), type(a_en.seg)
    ok &= run_one("energy", a_en, segs[0])

    # ── 4) 极端输入：空段 / 全零 / 超短（历史踩过 TypeError/NaN） ──
    print("\n=== D. 极端输入（空/全零/1 样本/单点） ===")
    for name, s in (("空数组", np.zeros(0, dtype=np.float32)),
                    ("全零 1s", np.zeros(SR, dtype=np.float32)),
                    ("1 个样本", np.zeros(1, dtype=np.float32)),
                    ("极小值", np.full(SR, 1e-9, dtype=np.float32))):
        print(f"  --- {name} ---")
        ok &= run_one("silero", a_sil, s)

    print("\n" + ("✅ 全部通过：Silero 后端下 handle() 稳定，老后端无回归"
                  if ok else "❌ 有失败项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
