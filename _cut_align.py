# -*- coding: utf-8 -*-
"""切断定位 · 对齐版：拿 _raw_rec.py 长时间录下来的原始音频，
   按【时间戳】去对 app 自己存档的每一段（utt/index.jsonl），逐段回答：

     段起点（= 存档时刻 − 段长）处，原始音频是不是【已经在说话】？
        · 段前 200ms 电平 ≈ 段内最响电平  → 段头是从说话中间切进来的 ✗
        · 段前 200ms 电平 ≈ 环境底        → 段头前面还留着安静 ✓
     再报一个不依赖 VAD 的"起音时刻"（Silero 概率越过 0.2 的块），
     于是：切断毫秒 = 段起点 − 起音时刻（>0 就是切掉了这么多语音）

   ⚠️ 与 _cut_report.py 有少量重复代码（虚拟时钟 PreGain + 起音判据）。刻意不复用：
      那边只是为了把"离线复放"对齐到现场，改坏了会污染结论。
"""
import json
import sys
import time as _time
import urllib.request
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from vad_silero import PreGain, SileroVAD, WINDOW, SR  # noqa: E402
import vad_silero as V  # noqa: E402

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "_cutprobe"
CFG = json.loads((HERE / "intent_config.json").read_text(encoding="utf-8"))["behavior"]
CHUNK_MS = WINDOW / SR * 1000


class _VClock:
    def __init__(self, t0=1000.0, step=WINDOW / SR):
        self.t, self.step = t0, step

    def time(self):
        return self.t

    def tick(self):
        self.t += self.step


def probs_of(x, seed_gain=8.0):
    clk, real = _VClock(), V.time
    V.time = clk
    try:
        g = PreGain(target_rms=float(CFG.get("vad_target_rms", 0.03)),
                    max_gain=float(CFG.get("vad_max_gain", 30.0)))
        g.gain = float(seed_gain)
        vad = SileroVAD()
        out = []
        for i in range(0, len(x) - WINDOW + 1, WINDOW):
            out.append(vad.prob(g.apply(x[i:i + WINDOW])))
            clk.tick()
    finally:
        V.time = real
    return np.array(out)


def onset_times(probs, rise=0.2, hi=0.5, quiet_run=10):
    """返回 [起音起点秒, ...]（安静 ≥10 块后第一次越过 rise）"""
    out, run, cand = [], 0, None
    for i, p in enumerate(probs):
        if p < rise:
            run += 1
            cand = None
            continue
        if cand is None and run >= quiet_run:
            cand = i
        if cand is not None and p >= hi:
            out.append(cand * CHUNK_MS / 1000.0)
            cand = None
            run = 0
    return out


def asr(seg, sr=SR, gain=8.0, timeout=120):
    import io
    url = CFG.get("asr_http_url", "http://127.0.0.1:30803/v1/audio/transcriptions")
    model = CFG.get("asr_http_model", "qwen3-asr")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(np.asarray(seg) * gain, -1, 1) * 32767).astype("<i2").tobytes())
    bnd = "----align"
    body = (f"--{bnd}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n"
            f"--{bnd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n").encode() + buf.getvalue() + \
        f"\r\n--{bnd}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={bnd}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        t = json.loads(r.read().decode("utf-8")).get("text", "") or ""
    return t.replace("language Chinese", "").replace("<asr_text>", "").strip()


def main():
    meta = json.loads((DIR / "meta.json").read_text(encoding="utf-8"))
    with wave.open(str(DIR / "raw.wav"), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    t0 = float(meta["start_epoch"])
    sec = meta["sec"]
    print(f"\n  原始录音 {sec:.1f}s  起始 {meta.get('start_iso')}  峰值 {meta['peak']:.5f}  "
          f"RMS {meta['rms']:.5f}")
    print("  算 Silero 概率时间线 ...", flush=True)
    p = probs_of(x)
    on = onset_times(p)
    print(f"  prob: p50={np.percentile(p,50):.3f}  p95={np.percentile(p,95):.3f}  "
          f"max={p.max():.3f}   ≥0.5 块数={int((p>=0.5).sum())}   起音 {len(on)} 个")

    rows = [json.loads(l) for l in (HERE / "utt" / "index.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("src") == "mic"]
    inwin = []
    for r in rows:
        ep = _time.mktime(_time.strptime(r["ts"], "%Y-%m-%d %H:%M:%S"))
        off = ep - t0
        if 1.0 <= off <= sec - 1.0:
            inwin.append((off, r))
    print(f"\n  ══ 落在录音窗口内的 app 存档段：{len(inwin)} 条 ══")
    if not inwin:
        print("     （没有。窗口内 app 没切出段，或者用户没说话）")
        return 0

    print(f"     {'时刻':>8} {'段长':>6} {'段起点':>8} {'首50ms/段RMS':>12} "
          f"{'段前200ms/段最响':>16}  判定")
    for off, r in inwin:
        dur = float(r["dur"])
        head = off - dur                                  # 段起点（录音内秒）
        h0, h1 = int(max(0, head * SR)), int(max(0, off * SR))
        seg = x[h0:h1]
        pre = x[max(0, h0 - int(0.2 * SR)):h0]
        if len(seg) < SR // 4:
            continue
        head50 = float(np.sqrt((seg[:800] ** 2).mean())) if len(seg) >= 800 else 0.0
        segrms = float(np.sqrt((seg ** 2).mean()))
        h200 = int(0.2 * SR)
        nw = max(1, len(seg) // h200)
        pk = max(float(np.sqrt((seg[i * h200:(i + 1) * h200] ** 2).mean()))
                 for i in range(nw))
        pr = float(np.sqrt((pre ** 2).mean())) if len(pre) else 0.0
        ratio = pr / max(pk, 1e-9)
        # 最近的起音（在段起点之前 2.5s 内）
        cand = [q for q in on if head - 2.5 <= q <= head + 0.1]
        o = max(cand) if cand else None
        cut = (head - o) * 1000 if o is not None else None
        if ratio > 0.5:
            v = f"✗ 段前已在说话（{ratio:.2f}×段内最响）"
        else:
            v = f"✓ 段前是安静（{ratio:.2f}）"
        if cut is not None:
            v += (f"；起音 {o:.2f}s → 段起点 {head:.2f}s = "
                  + (f"切掉 {cut:.0f}ms ✗" if cut > 100 else
                     (f"起音在段内 {abs(cut):.0f}ms ✓" if cut < -100 else "边界")))
        print(f"     {r['ts'][-8:]:>8} {dur:>5.2f}s {head:>7.2f}s "
              f"{head50/max(segrms,1e-9):>11.2f}x {ratio:>15.2f}  {v}")
        print(f"              app 文本：{r.get('text','')!r}")
        try:
            t = asr(seg)
            print(f"              原始音频转写（同一窗口，×8）：{t!r}")
        except Exception as e:
            print(f"              原始音频转写失败：{type(e).__name__}: {e}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
