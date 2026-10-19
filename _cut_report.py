# -*- coding: utf-8 -*-
"""切断定位 · 第二半：对齐"真实起音"与"段起点"

输入：_cutprobe/raw.wav（_cut_probe.py 采的原始音频，无 AGC）
做法：
  1. 按生产同款 PreGain 增益 → 逐 512 点算 Silero 概率和 dB
  2. 用【生产同款状态机逻辑】（含 5 块平滑 + hits/misses）复现切段，逐块留痕
  3. 「真实起音」= 一段安静（prob<0.2 持续 ≥10 块）之后第一次 prob≥0.5 的块
  4. 关键量：段起点块 − 起音块
       ≤0  → 段头在起音之前，还留着 |x|×32ms 的余量 ✓ 没切
       >0  → 段头落在语音【里面】，前面 x×32ms 的语音没进段 → 切断 ✗（≈吃掉 x×32/150 个字）
  5. 反事实：把 pre_buffer 加大 / hits 降为 1 / 关掉 db 硬门限，看各能救回多少

用法：E:\\voice_agent_build\\Scripts\\python.exe -X utf8 _cut_report.py
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from vad_silero import PreGain, SileroVAD, WINDOW, SR  # noqa: E402
import vad_silero as V  # noqa: E402

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "_cutprobe"
CHUNK_MS = WINDOW / SR * 1000          # 32ms
CFG = json.loads((HERE / "intent_config.json").read_text(encoding="utf-8"))["behavior"]

PROD = dict(prob_thr=float(CFG.get("vad_prob_threshold", 0.4)),
            db_thr=float(CFG.get("vad_db_threshold", 40)),
            hits=int(CFG.get("vad_required_hits", 2)),
            misses=int(CFG.get("vad_required_misses", 22)),
            pre=20, smooth=5,
            max_ms=int(CFG.get("vad_max_speech_ms", 12000)))

VARIANTS = [
    ("现状（生产参数）", {}),
    ("pre_buffer 20→40（640→1280ms）", dict(pre=40)),
    ("hits 2→1（去掉起音确认）", dict(hits=1)),
    ("关掉 db 硬门限（db_thr→-inf）", dict(db_thr=-1e9)),
    ("pre=40 + hits=1", dict(pre=40, hits=1)),
    ("pre=40 + hits=1 + 无 db 门限", dict(pre=40, hits=1, db_thr=-1e9)),
]


# ══════════════════════════════════════════════════════════════
#  生产同款状态机（vad_silero.StateMachine.process 的逐行副本 + 留痕）
# ══════════════════════════════════════════════════════════════
class SM:
    def __init__(self, prob_thr, db_thr, hits, misses, pre, smooth=5, max_ms=12000):
        self.prob_thr, self.db_thr = prob_thr, db_thr
        self.hits, self.misses = hits, misses
        self.pre = pre
        self.max_chunks = int(max_ms / CHUNK_MS)
        self.state = "IDLE"
        self.prob_win, self.db_win = [], []
        self.chunks = 0
        self.pre_len = 0                   # pre_buffer 里当前有几块（INACTIVE 后会清零）
        self.start_chunk = 0
        self.hit_count = self.miss_count = 0
        self.log = []                      # 逐块留痕
        self.segments = []                 # (start_chunk, end_chunk)

    def process(self, i, prob, db):
        self.prob_win.append(prob)
        self.db_win.append(db)
        sp = float(np.mean(self.prob_win[-5:]))
        sd = float(np.mean(self.db_win[-5:]))
        hit = (sp >= self.prob_thr and sd >= self.db_thr)
        seg = None
        if self.state == "IDLE":
            self.pre_len += 1                            # pre_buffer.append(chunk)
            if hit:
                self.hit_count += 1
                if self.hit_count >= self.hits:
                    self.state = "ACTIVE"
                    npre = min(self.pre_len, self.pre)   # 段首带回来的预滚动块数
                    self.chunks = npre
                    self.start_chunk = i - npre + 1
                    self.hit_count = 0
                    self.miss_count = 0
            else:
                self.hit_count = 0
        elif self.state == "ACTIVE":
            self.chunks += 1
            if hit:
                self.miss_count = 0
            else:
                self.miss_count += 1
                if self.miss_count >= self.misses:
                    self.state = "INACTIVE"
                    self.miss_count = 0
        elif self.state == "INACTIVE":
            self.segments.append((self.start_chunk, i - 1))
            seg = (self.start_chunk, i - 1)
            self.state = "IDLE"
            self.chunks = 0
            self.pre_len = 1                             # clear() 后再放进当前这块
            self.start_chunk = i
        self.log.append((i, prob, db, sp, sd, hit, self.state, self.hit_count,
                         self.miss_count, self.chunks))
        return seg


def load_raw():
    with wave.open(str(DIR / "raw.wav"), "rb") as w:
        assert w.getframerate() == SR, w.getframerate()
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


class _VClock:
    """虚拟时钟：让 PreGain 的"每 0.5s 才更新一次增益"在离线复放里也成立。

    ⚠️ 血债：PreGain 用 wall-clock 门控。离线全速复放 30s 音频只花 ~0.3s 真实时间
    → 增益只更新一两次、停在 ×1~×3（生产实际常年顶在 ×8）
    → Silero 概率被压低到 max 0.243、离线复放切出 0 段，
      而同一时刻生产（15:44:26）明明切出了一段。离线与现场必须同一个时钟口径。
    """
    def __init__(self, t0=1000.0, step=WINDOW / SR):
        self.t, self.step = t0, step

    def time(self):
        return self.t

    def tick(self):
        self.t += self.step


def repaint(x, seed_gain=8.0):
    """跑一遍 PreGain + Silero，返回 (gain 后音频, 每块 prob, 每块 db, 每块 gain)

    seed_gain=8.0：生产已常开十几分钟，增益早已顶在上限（现场日志 AGC×8.0）；
    离线从 ×1 冷启会白白少掉前几秒的电平。
    """
    clk = _VClock()
    real_time = V.time
    V.time = clk
    try:
        g = PreGain(target_rms=float(CFG.get("vad_target_rms", 0.03)),
                    max_gain=float(CFG.get("vad_max_gain", 30.0)))
        g.gain = float(seed_gain)
        vad = SileroVAD()
        gained, probs, dbs, gains = [], [], [], []
        for i in range(0, len(x) - WINDOW + 1, WINDOW):
            c = g.apply(x[i:i + WINDOW])
            gained.append(c)
            probs.append(vad.prob(c))
            rms = float(np.sqrt(np.mean((c * 32767) ** 2)))
            dbs.append(20 * np.log10(rms + 1e-7) if rms > 0 else -np.inf)
            gains.append(g.last_g)
            clk.tick()
    finally:
        V.time = real_time
    return (np.concatenate(gained) if gained else np.zeros(0, np.float32),
            np.array(probs), np.array(dbs), np.array(gains))


def onsets(probs, rise_thr=0.2, hi_thr=0.5, quiet_run=10):
    """真实起音：安静 ≥quiet_run 块后，第一次越过 rise_thr 的块就是「起音起点」，
    再等到它越过 hi_thr 记下「确凿起音」。返回 [(起音起点块, 确凿起音块), ...]

    ⚠️ 不能写成"越过 0.2 就重置安静计数"——那样在 0.2~0.5 之间爬升的软起音会被判丢
    （冒烟测试实测：0 个起音）。
    """
    out, run, cand = [], 0, None
    for i, p in enumerate(probs):
        if p < rise_thr:
            run += 1
            cand = None
            continue
        if cand is None and run >= quiet_run:
            cand = i                      # 刚越过 0.2 = 听得见的起音
        if cand is not None and p >= hi_thr:
            out.append((cand, i))
            cand = None
            run = 0
    return out


def main():
    if not (DIR / "raw.wav").exists():
        print(f"  先跑 _cut_probe.py 采一段：{DIR/'raw.wav'} 不存在")
        return 2
    x = load_raw()
    meta = json.loads((DIR / "meta.json").read_text(encoding="utf-8")) \
        if (DIR / "meta.json").exists() else {}
    gained, probs, dbs, gains = repaint(x)
    n = len(probs)
    print(f"\n  原始音频 {len(x)/SR:.2f}s   峰值 {np.abs(x).max():.5f}   "
          f"RMS {np.sqrt((x**2).mean()):.5f}   块数 {n}（{CHUNK_MS:.0f}ms/块）")
    print(f"  增益轨迹（虚拟时钟复放）：起 {gains[0]:.1f} → 中位 {np.median(gains):.1f} "
          f"→ 末 {gains[-1]:.1f}（现场日志常年在 ×8.0）")
    print(f"  环境中位电平（原始，块 RMS）{np.median([np.sqrt(np.mean(x[i*WINDOW:(i+1)*WINDOW]**2)) for i in range(n)]):.5f}")
    print(f"  prob: p50={np.percentile(probs,50):.3f} p95={np.percentile(probs,95):.3f} "
          f"max={probs.max():.3f}   ≥0.5 的块数={int((probs>=0.5).sum())}")
    print(f"  db(增益后,int16量纲): p95={np.percentile(dbs,95):.1f}   "
          f"门限={PROD['db_thr']:.0f}  ≥门限的块数={int((dbs>=PROD['db_thr']).sum())}")

    # ── 生产线复现 ──
    sm = SM(PROD["prob_thr"], PROD["db_thr"], PROD["hits"], PROD["misses"], PROD["pre"],
            max_ms=PROD["max_ms"])
    for i in range(n):
        sm.process(i, probs[i], dbs[i])
    on = onsets(probs)
    on_chunks = [c for c, _ in on]
    on_hi = {c: h for c, h in on}
    print(f"\n  ══ 切段复现（生产参数）══  切出 {len(sm.segments)} 段；"
          f"独立起音判据找到 {len(on)} 个起音（安静后 prob 越过 0.2 = 起音起点）")
    if meta.get("seg_lens"):
        print(f"     一致性核对：现场引擎段长 {[round(v/SR,2) for v in meta['seg_lens']]}  "
              f"离线 {[round((b-a+1)*CHUNK_MS/1000,2) for a,b in sm.segments]}")

    # ── 逐段对齐 ──
    print(f"\n  ══ 段头对齐（段起点块 − 起音起点块）══")
    print(f"     {'段':>3} {'起音':>5} {'确凿':>5} {'首hit':>6} {'触发':>5} {'段起点':>6} "
          f"{'段长':>7} {'起点−起音':>10}  判定")
    worst = []
    prev_end = -1
    for k, (a, b) in enumerate(sm.segments):
        t2 = a + PROD["pre"] - 1                       # 触发块（pre_buffer 尾部那块）
        # 起音只能取自【上一段结束之后、触发之前】——否则会抓到上一句的起音，算出假的巨量切断
        o = max([q for q in on_chunks if prev_end < q <= t2], default=None)
        prev_end = b
        d = (a - o) if o is not None else None
        dur = (b - a + 1) * CHUNK_MS / 1000
        h1 = next((j for j in range(o, t2 + 1) if o is not None and sm.log[j][5]), None)
        if d is None:
            verdict = "?（本段前没找到起音）"
        elif d <= 0:
            verdict = f"✓ 段前还留 {-d*CHUNK_MS:.0f}ms"
        elif d <= 40:
            verdict = f"✗ 切掉 {d*CHUNK_MS:.0f}ms = 约{d*CHUNK_MS/250:.1f}~{d*CHUNK_MS/150:.1f}个字"
        else:
            verdict = f"（段内含后续起音 +{d*CHUNK_MS:.0f}ms，非段头切断）"
        print(f"     {k:>3} {o if o is not None else -1:>5} "
              f"{on_hi.get(o, -1) if o is not None else -1:>5} "
              f"{h1 if h1 is not None else -1:>6} {t2:>5} {a:>6} {dur:>6.2f}s "
              f"{(d if d is not None else 0)*CHUNK_MS:>9.0f}ms  {verdict}")
        if d is not None and 0 < d <= 40:
            worst.append((k, d * CHUNK_MS))
        if o is not None and d is not None and d <= 40:
            print(f"          起音路径：起音块{o} → 确凿块{on_hi.get(o,-1)}"
                  f"（+{(on_hi.get(o,o)-o)*CHUNK_MS:.0f}ms）→ 首 hit 块{h1}"
                  f"（+{((h1-o)*CHUNK_MS) if h1 is not None else -1:.0f}ms）"
                  f" → 触发块{t2}（+{(t2-o)*CHUNK_MS:.0f}ms）→ 段起点块{a}")

    # ── 段头里到底是不是"已经响了"（证据，不靠 VAD） ──
    print(f"\n  ══ 段起点前 200ms 的【原始】电平 vs 段内最响 200ms（不依赖 VAD 的直接证据）══")
    for k, (a, b) in enumerate(sm.segments):
        s0 = a * WINDOW
        pre = x[max(0, s0 - 3200):s0]
        seg = x[s0:(b + 1) * WINDOW]
        seg_g = gained[s0:(b + 1) * WINDOW]
        pr = float(np.sqrt((pre ** 2).mean())) if len(pre) else 0.0
        # 段内最响 200ms 窗
        h = 3200
        nw = max(1, len(seg) // h)
        pk = max(float(np.sqrt((seg[i*h:(i+1)*h] ** 2).mean())) for i in range(nw)) if len(seg) else 0.0
        head50 = float(np.sqrt((seg_g[:800] ** 2).mean())) if len(seg_g) >= 800 else float("nan")
        segrms = float(np.sqrt((seg_g ** 2).mean())) if len(seg_g) else float("nan")
        print(f"     段{k}: 段前200ms {pr:.5f} / 段内最响200ms {pk:.5f} = {pr/max(pk,1e-9):5.2f}"
              f"   → {'段前已经在说话 ✗（切进来了）' if pr > 0.5*pk else '段前是安静 ✓（没切）'}"
              f"    段头50ms/段RMS={head50/max(segrms,1e-9):.2f}x")

    # ── 反事实 ──
    print(f"\n  ══ 反事实：改哪个旋钮能救回切断 ══")
    print(f"     {'方案':<34} {'切掉的段数':>8} {'最坏切掉':>9} {'合计损失':>9}")
    for name, ov in VARIANTS:
        p = dict(PROD)
        p.update(ov)
        s = SM(p["prob_thr"], p["db_thr"], p["hits"], p["misses"], p["pre"], max_ms=p["max_ms"])
        for i in range(n):
            s.process(i, probs[i], dbs[i])
        losses = []
        prev_end = -1
        for a, b in s.segments:
            t2 = a + p["pre"] - 1
            o = max([q for q in on_chunks if prev_end < q <= t2], default=None)
            prev_end = b
            if o is not None and 0 < a - o <= 40:
                losses.append((a - o) * CHUNK_MS)
        print(f"     {name:<34} {len(losses):>8} "
              f"{(max(losses) if losses else 0):>8.0f}ms "
              f"{sum(losses):>8.0f}ms")

    (DIR / "report.json").write_text(json.dumps(
        {"n_chunks": n, "segments": sm.segments, "onsets": on,
         "worst_cut_ms": worst,
         "probs": [round(float(v), 4) for v in probs],
         "dbs": [round(float(v), 1) for v in dbs]}, ensure_ascii=False), encoding="utf-8")
    print(f"\n  💾 明细写到 {DIR/'report.json'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
