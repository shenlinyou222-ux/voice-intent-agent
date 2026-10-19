# -*- coding: utf-8 -*-
"""按住说话 demo v2（只读，不改 app 主流程）

v1 的三个缺口，这版补掉：
  ① 只存了"处理后"（裁过尾的），没法跟原始段对比  → 现在 *_raw.wav + *_proc.wav 都存
  ② ASR 是在主循环里同步做的：那 140~165ms 里【停止读取输入流】。
     本项目血债（probe_overflow.py）：停读 3 秒只读到 241/334 块 = 静默丢 2.8s 音频，
     而 PortAudio 的 overflow 标记报 0 次。→ 现在 ASR 走后台线程，读取永不中断。
  ③ 裁尾阈值取"段头 150ms"当环境底：若你按下就开口，那段头里就是语音 →
     环境底被高估 → 阈值偏高 →【把段尾的话裁掉】。
     → 改成取整段帧能量的 p10（最静的那些帧几乎一定是静音），与是否立刻开口无关；
       并保留 300ms 尾巴（实测挖掉 500ms 会掉字：「…是不是能？」丢「录进去」）。

用法：python -X utf8 _ptt_demo.py --key RCTRL --sec 150 --n 3
"""
import argparse
import collections
import json
import queue
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from ptt import HoldKey                      # noqa: E402
from vad_silero import SR                    # noqa: E402
from _cut_align import asr as asr_post       # noqa: E402

BLOCK = 480                                  # 30ms，和 app 一致


def frame_env(x, h=320):                     # 20ms 帧
    n = len(x) // h
    return np.sqrt((x[:n * h].reshape(n, h) ** 2).mean(axis=1)) if n else np.zeros(0)


def trim_tail(x, keep_ms=300, ratio=6.0, floor=0.0008):
    """裁掉尾部静音。环境底取整段帧能量的 p10（不依赖"段头就是静音"这个假设）。

    ⚠️ 必须裁：Qwen3-ASR 在静音上会编词（实测 30s 近静音被转成
       「《中国好声音》第二季，其实我看是观众演唱。」）。
    返回 (裁剪后音频, 裁掉的毫秒数, 用到的阈值, 环境底)
    """
    h = int(0.02 * SR)
    n = len(x) // h
    if n == 0:
        return x, 0.0, 0.0, 0.0
    e = frame_env(x, h)
    amb = float(np.percentile(e, 10))
    thr = max(amb * ratio, floor)
    idx = np.where(e >= thr)[0]
    if len(idx) == 0:
        return x[:0], len(x) / SR * 1000, thr, amb
    end = min(n, int(idx[-1]) + 1 + int(keep_ms / 20))
    return x[:end * h], (n - end) * h / SR * 1000, thr, amb


def wav_write(p, x):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(np.asarray(x), -1, 1) * 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="RCTRL")
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--preroll-ms", type=float, default=300.0)
    ap.add_argument("--sec", type=float, default=150.0)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "_ptt"))
    ap.add_argument("--gain", type=float, default=8.0,
                    help="送 ASR 的增益（现状送的是 VAD 引擎输出=PreGain 常年 ×8）")
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--keep-ms", type=float, default=300.0, help="裁尾后保留的余量")
    a = ap.parse_args()

    import sounddevice as sd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    key = HoldKey(a.key)
    ring = collections.deque(maxlen=max(1, int(a.preroll_ms / 30)))

    jobs = queue.Queue()                     # 交给后台线程做 ASR，读取不中断
    results = {}

    def _worker():
        while True:
            item = jobs.get()
            if item is None:
                return
            idx, seg, fn = item
            t = ""
            ms = 0.0
            try:
                t0 = time.time()
                t = asr_post(seg, gain=a.gain)
                ms = (time.time() - t0) * 1000
            except Exception as e:
                t = f"❌ {type(e).__name__}: {e}"
            results[idx] = (ms, t)
            print(f"     ✅ #{idx} {ms:5.0f}ms  「{t}」   ({fn})", flush=True)

    th = None
    if not a.no_asr:
        th = threading.Thread(target=_worker, daemon=True, name="asr")
        th.start()

    print(f"  设备 {a.device}：{sd.query_devices(a.device)['name']}")
    print(f"  键位 {key.name}  预滚动 {a.preroll_ms:.0f}ms  送 ASR ×{a.gain:g}  "
          f"裁尾余量 {a.keep_ms:.0f}ms")
    print(f"\n  🎤 按住【{key.name}】说话，松开就出字；最多 {a.n} 句 / {a.sec:.0f}s\n", flush=True)

    cap, t_down, n_done = None, 0.0, 0
    last_print, t0 = 0.0, time.time()
    meta = []
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=BLOCK, device=a.device) as st:
        while time.time() - t0 < a.sec and n_done < a.n:
            blk, ovf = st.read(BLOCK)                    # ⚠️ 永不在这之间做重活
            x = np.asarray(blk[:, 0], dtype=np.float32).copy()
            ev = key.poll()
            if ev == "down":
                cap = list(ring) + [x]
                t_down = time.time()
                print(f"     🔴 按下（预滚动 {len(cap)*30}ms）", flush=True)
            elif ev == "up" and cap is not None:
                cap.append(x)
                hold = time.time() - t_down
                raw = np.concatenate(cap)
                cap = None
                seg, cut_ms, thr, amb = trim_tail(raw, keep_ms=a.keep_ms)
                n_done += 1
                pre_samples = None
                base = f"{time.strftime('%H%M%S')}_ptt{n_done}"
                fp = out / f"{base}_proc.wav"
                fr = out / f"{base}_raw.wav"
                wav_write(fp, seg)
                wav_write(fr, raw)
                e = frame_env(seg)
                idx = np.where(e >= thr)[0]
                speech_ms = (int(idx[0]) * 20, int(idx[-1]) * 20) if len(idx) else (-1, -1)
                h50 = float(np.sqrt((seg[:800] ** 2).mean())) if len(seg) >= 800 else float("nan")
                rms = float(np.sqrt((seg ** 2).mean()))
                print(f"     ⚪ 松开 按住 {hold:.2f}s  原始段 {len(raw)/SR:.2f}s → "
                      f"处理后 {len(seg)/SR:.2f}s（裁尾 {cut_ms:.0f}ms  "
                      f"环境底 {amb:.5f} 阈值 {thr:.5f}）", flush=True)
                print(f"        处理后段内像说话的区间 {speech_ms[0]}~{speech_ms[1]}ms  "
                      f"头50ms/段RMS={h50/max(rms,1e-9):.2f}x", flush=True)
                if len(seg) / SR < 0.25:
                    print("        ⏭ 太短（<0.25s），丢弃", flush=True)
                    n_done -= 1
                    continue
                if th is not None:
                    jobs.put((n_done, seg.copy(), f"{base}_proc.wav"))
                meta.append({"n": n_done, "raw": fr.name, "proc": fp.name,
                             "hold_s": round(hold, 2),
                             "raw_s": round(len(raw) / SR, 2),
                             "proc_s": round(len(seg) / SR, 2),
                             "cut_ms": round(cut_ms), "amb": round(amb, 6),
                             "thr": round(thr, 6),
                             "speech_ms": speech_ms,
                             "head50_over_rms": round(h50 / max(rms, 1e-9), 3)})
            else:
                if cap is not None:
                    cap.append(x)
                else:
                    ring.append(x)
                now = time.time()
                if now - last_print > 1.0 and cap is None:
                    last_print = now
                    print(f"     … 待命 {now-t0:5.0f}s  电平 "
                          f"{float(np.sqrt((x**2).mean())):.5f}  "
                          f"{'⚠️溢出' if ovf else ''}", flush=True)

    if th is not None:
        jobs.put(None)
        th.join(timeout=60)
    for m in meta:
        ms, t = results.get(m["n"], (0.0, "(未做)"))
        m["asr_ms"], m["asr"] = round(ms), t
    (out / "demo.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"\n  ══ 汇总 ══")
    for m in meta:
        print(f"    #{m['n']} 按住 {m['hold_s']}s  原始 {m['raw_s']}s → 处理 {m['proc_s']}s  "
              f"(裁尾 {m['cut_ms']}ms)  头50/段RMS {m['head50_over_rms']}  "
              f"ASR {m.get('asr_ms')}ms  「{m.get('asr')}」")
    print(f"  💾 raw/proc wav + demo.json 在 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
