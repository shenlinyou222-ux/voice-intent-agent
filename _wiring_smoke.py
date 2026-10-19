# -*- coding: utf-8 -*-
"""接线冒烟测试（不碰麦克风、不加载模型、不打字）

验证的是"接进去之后 app 还起得来、而且走的是 PTT 门"：
  ① Config/Agent 构造成功，self.vad_backend == "ptt"，self.seg 是 PttGate
  ② 用假按键喂 30ms 块：不按时 0 段；按下→松开 出 1 段
  ③ handle() 会用到的那几个属性都在（frame_len/last_trim/is_ptt/level/gain_now/
     last_prob/flush/set_noise_floor），少一个就是"常开进程自动退出"级别的血债
  ④ trigger_mode="vad" 回滚路径仍然能构造（Silero 引擎）
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import voice_agent as VA            # noqa: E402
from ptt import PttGate             # noqa: E402


class FakeKey:
    name = "FAKE"

    def __init__(self, edges):
        self.edges, self.i = dict(edges), 0

    def poll(self):
        ev = self.edges.get(self.i)
        self.i += 1
        return ev


def check(name, cond, detail=""):
    print(f"    {'✅' if cond else '❌'} {name}" + (f"   {detail}" if detail else ""))
    return bool(cond)


def main():
    ok = True
    cfg = VA.Config(None)
    print(f"\n  ══ 接线冒烟 ══")
    print(f"  trigger_mode = {cfg.behavior.get('trigger_mode')!r}   "
          f"ptt_key = {cfg.behavior.get('ptt_key')!r}   "
          f"ptt_preroll_ms = {cfg.behavior.get('ptt_preroll_ms')}")

    ag = VA.Agent(cfg, device="cpu")
    ok &= check("Agent 构造成功", True)
    ok &= check("走 PTT 门（self.seg 是 PttGate）", isinstance(ag.seg, PttGate),
                f"实得 {type(ag.seg).__name__}")
    ok &= check("vad_backend == 'ptt'", ag.vad_backend == "ptt",
                f"实得 {ag.vad_backend!r}")

    # handle() / 电平表 / run_mic 会用到的属性，一个都不能缺
    need = ["frame_len", "last_trim", "is_ptt", "level", "gain_now", "last_prob",
            "last_seg_prob", "threshold", "rms_floor", "rms_ceiling",
            "voice_margin", "energy_threshold", "state", "preroll_ms", "key_name",
            "min_ms"]
    missing = [n for n in need if not hasattr(ag.seg, n)]
    ok &= check("handle()/电平表要的属性齐全", not missing, f"缺 {missing}")
    for meth in ("push", "flush", "reset", "set_noise_floor"):
        ok &= check(f"方法 {meth}()", callable(getattr(ag.seg, meth, None)))

    # ② 喂块：不按 → 0 段；按下→松开 → 1 段
    gate = PttGate(key_obj=FakeKey({}), preroll_ms=1000)
    rng = np.random.default_rng(3)
    x = (rng.normal(0, 1, 480).astype(np.float32)) * 0.0006
    got = []
    for _ in range(60):
        got += gate.push(x)
    ok &= check("不按键 → 0 段", got == [])

    speech = (rng.normal(0, 1, 480).astype(np.float32)) * 0.005
    gate = PttGate(key_obj=FakeKey({50: "down", 100: "up"}), preroll_ms=1000)
    got = []
    for i in range(140):
        got += gate.push(speech if 50 <= i < 100 else x)
    ok &= check("按下→松开 → 1 段", len(got) == 1, f"实得 {len(got)}")
    if got:
        print(f"       段长 {len(got[0])/16000:.2f}s  "
              f"（按住 50 块=1.5s + 1s 预滚动，两头裁剪后）")
        # 按住后【才】开口 → 段头那 840ms 按键前静音被裁掉（只留 150ms），
        # 这是设计：预滚动是"兜住早开口"的保险，没早开口就不该把静音喂给 ASR。
        ok &= check("段长合理（≈1.5~2.0s：语音+两头余量）",
                    1.5 <= len(got[0]) / 16000 <= 2.0, f"{len(got[0])/16000:.2f}s")
    ok &= check("level/gain_now 可读（电平表不炸）",
                ag.seg.level >= 0 and ag.seg.gain_now > 0,
                f"level={ag.seg.level:.5f} gain={ag.seg.gain_now:.1f}")

    # ④ 回滚路径
    cfg2 = VA.Config(None)
    cfg2.raw["behavior"]["trigger_mode"] = "vad"
    cfg2.raw["behavior"]["vad_backend"] = "silero"
    ag2 = VA.Agent(cfg2, device="cpu")
    ok &= check("回滚 trigger_mode=vad 仍能构造 Silero 引擎",
                type(ag2.seg).__name__ == "SileroVADEngine" and ag2.vad_backend == "silero",
                f"{type(ag2.seg).__name__} / {ag2.vad_backend}")

    print(f"\n  {'🎉 接线冒烟通过' if ok else '⚠️ 有失败项'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
