# -*- coding: utf-8 -*-
"""双路在线台（只打印，不执行）—— 把「A 抢跑 + B 权威 + 对账」整条链在真麦克风上跑通。

A 提前执行路：按住期间每 chunk_s 秒，把【已累积音频】重转一次 ASR（分片离线口径）
              → 文本喂切分器 → 每段只认规则层直通 → 且只留可撤销动作
              → 记录"本会执行什么"（**不真执行**）
B 权威路    ：松手后整段音频转一次 ASR（今天的路径）→ 整句总闸 + 全管线判一条
对账        ：一致 / A 多出(需回滚) / B 多出(需补执行) / 冲突 / 总闸不过(全回滚)
              → 打屏 + 写 _dual/live.jsonl（这就是调 pause 参数的自适应信号源）

为什么先用「分片离线 ASR」而不是 vllm-asr realtime：
  · 零停机（不动 llama-asr）、零显存改动
  · 精度比真流式更高（每次都带完整上下文，没有被 delta 改写的风险）
  · 唯一代价：每 1s 一次短 ASR（~100ms，GPU 有余量）+ 粒度 1s
  架构跑通后把 A 的文本源换成真流式即可，切分器/对账/回滚都不用动。

用法：python -X utf8 _dual_live.py --key RCTRL --n 5
"""
import argparse
import collections
import json
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from ptt import HoldKey                                        # noqa: E402
from intent_split import (Splitter, triggers_from_cfg,          # noqa: E402
                          boundary_from_cfg, names_from_cfg)
from _cut_align import asr as asr_post                          # noqa: E402
from _dual_path_sim import verdict, REVERSIBLE                  # noqa: E402
import voice_agent as va                                        # noqa: E402
from actions import Config                                      # noqa: E402

SR, BLOCK = 16000, 480
CFGJ = json.loads((HERE / "intent_config.json").read_text(encoding="utf-8"))
TR = triggers_from_cfg(CFGJ)
BD = boundary_from_cfg(CFGJ)
NM = names_from_cfg(CFGJ)


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
    ap.add_argument("--preroll-ms", type=float, default=1000)
    ap.add_argument("--chunk-s", type=float, default=1.0)
    ap.add_argument("--min-s", type=float, default=0.8)
    ap.add_argument("--n", type=int, default=5, help="收够几次说话就退出")
    ap.add_argument("--sec", type=float, default=300)
    ap.add_argument("--out", default=str(HERE / "_dual"))
    a = ap.parse_args()

    import sounddevice as sd
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config(None)
    embed = None
    try:
        from intent_embed import EmbedIntent
        embed = EmbedIntent(
            url=str(cfg.behavior.get("intent_embed_url")
                    or "http://172.20.71.101:30802/v1/embeddings"),
            index=str(HERE / str(cfg.behavior.get("intent_index", "intent_index.json"))),
            thr=float(cfg.behavior.get("intent_embed_thr") or 0.30),
            margin=float(cfg.behavior.get("intent_embed_margin") or 0.0))
        print(f"  B 路 kNN 就绪（{len(embed.labels)} 示例 / {len(embed.intents)} 类）")
    except Exception as e:
        print(f"  ⚠️ kNN 不可用（{type(e).__name__}: {e}）→ B 路只走规则层")

    key = HoldKey(a.key)
    preroll_blocks = max(1, int(a.preroll_ms / 30))
    blocks = collections.deque(maxlen=int(60 / 0.03))     # 保留最近 60s
    lock = threading.Lock()
    jobs = collections.deque()                            # (kind, audio, meta)
    jobs_lock = threading.Lock()
    state = {"a_actions": [], "chunk_i": 0, "press_at": None, "n_done": 0}
    stop = {"v": False}

    def snapshot():
        with lock:
            return np.concatenate(blocks)

    def fmt(acts):
        return "、".join(f"{x['intent']}({x['arg']})" for x in acts) or "—"

    def worker():
        while not stop["v"]:
            job = None
            with jobs_lock:
                if jobs:
                    job = jobs.popleft()
            if job is None:
                time.sleep(0.02)
                continue
            kind, audio, meta = job
            t0 = time.time()
            try:
                text = asr_post(audio, gain=8.0)
            except Exception as e:
                print(f"     ❌ ASR 失败 {type(e).__name__}: {e}")
                continue
            ms = (time.time() - t0) * 1000
            segs = [s.text for s in Splitter(TR, boundary=BD, names=NM).segments(text)]
            if kind == "chunk":
                vs = [verdict(cfg, None, s, allow_knn=False) for s in segs]   # A：只认规则直通
                seen = [(x["intent"], x["arg"]) for x in state["a_actions"]]
                new = [v for v in vs
                       if v["ok"] and v["intent"] in REVERSIBLE
                       and (v["intent"], v["arg"]) not in seen]
                state["a_actions"] += [{"intent": v["intent"], "arg": v["arg"],
                                        "why": v["why"]} for v in new]
                print(f"     ⏱+{meta:.1f}s {ms:5.0f}ms  「{text}」")
                print(f"        {'切 ' + str(len(segs)) + ' 段 ' + str(segs) if len(segs) > 1 else '单段'}"
                      f"   A 本轮新增会执行：{fmt(new)}")
            else:
                b = verdict(cfg, embed, text, allow_knn=True)
                aa = state["a_actions"]
                if aa and not b["ok"]:
                    r = "需回滚（B 不执行 → 撤 A）"
                elif aa and b["ok"]:
                    if any((x["intent"], x["arg"]) == (b["intent"], b["arg"]) for x in aa):
                        r = "一致" if len(aa) == 1 else f"一致但 A 多出 {len(aa)-1} 条"
                    else:
                        r = "冲突（A 与 B 不是同一条）"
                elif (not aa) and b["ok"]:
                    r = "需补执行（A 没做、B 补）"
                else:
                    r = "两边都不动"
                print(f"     🅱 权威 {ms:5.0f}ms  「{text}」")
                print(f"        B：{'✅ ' + b['intent'] + '(' + b['arg'] + ')  ' + b['why'] if b['ok'] else '⛔ ' + b['why']}")
                print(f"        A 累计会执行：{fmt(aa)}")
                print(f"     ⚖ 对账：{r}")
                rec = {"t": time.strftime("%H:%M:%S"), "final_text": text,
                       "A": aa, "B": (b if b["ok"] else {"ok": False, "why": b["why"]}),
                       "verdict": r, "n_chunks": state["chunk_i"]}
                with open(out / "live.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                state["a_actions"], state["chunk_i"] = [], 0
                state["n_done"] += 1

    th = threading.Thread(target=worker, daemon=True, name="dual-worker")
    th.start()

    print(f"\n  🎤 按住【{key.name}】说一句（长句多指令最好），松手看对账；{a.n} 次后退出")
    print(f"     预滚动 {a.preroll_ms:.0f}ms   A 每 {a.chunk_s:.1f}s 抢跑一次（只打印不执行）\n")
    pressed, last_chunk, t_press = False, 0.0, 0.0
    cap = None           # ⚠️ 血债：绝不能存"绝对索引"再切滚动 deque！
                         #    blocks 是 maxlen=2000（60s）的滚动缓冲，启动到按下超过 60s
                         #    就满了、开始从左边丢 → 存下的索引会跟着滑到列表尾部，
                         #    切片变成"最近 1 秒的移动窗口"。
                         #    实测症状：按住 10 秒只送出 0.99s；每块文本都是不同碎片
                         #    （「打开。」「开知乎。」「搜索Deep。」）而不是递增前缀。
                         #    正确写法：按下时把预滚动【复制一份】，之后每块往里追加
                         #    （PttGate 就是这么写的，所以生产门没这个问题）。
    _hb = [0.0]
    t0 = time.time()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=BLOCK, device=a.device) as st:
        while time.time() - t0 < a.sec and state["n_done"] < a.n:
            blk, _ = st.read(BLOCK)
            x = np.asarray(blk[:, 0], dtype=np.float32).copy()
            with lock:
                blocks.append(x)
                idx = len(blocks) - 1
            _now = time.time()
            if _now - _hb[0] > 2.0:
                _hb[0] = _now
                with lock:
                    _nb = len(blocks)
                print(f"        · 心跳 {_now-t0:5.1f}s  块数 {_nb}  "
                      f"速率 {_nb*0.03/max(0.001, _now-t0):5.2f}x 实时  "
                      f"cap={len(cap) if cap is not None else 0}块", flush=True)
            ev = key.poll()
            if ev == "down":
                pressed, t_press = True, time.time()
                with lock:
                    cap = list(blocks)[-preroll_blocks:]      # 复制预滚动（不含索引依赖）
                cap.append(x)                                 # 当前块（已在 blocks 里）
                last_chunk, state["chunk_i"] = 0.0, 0
                print(f"     🔴 按下（预滚动 {len(cap)*30}ms）")
            elif ev == "up" and pressed:
                pressed = False
                if cap is not None:
                    cap.append(x)
                    audio = np.concatenate(cap)
                else:
                    audio = np.zeros(0, np.float32)
                cap = None
                hold = time.time() - t_press
                print(f"     ⚪ 松开 按住 {hold:.2f}s  块数 {len(audio)//BLOCK}"
                      f"  送整段 {len(audio)/SR:.2f}s")
                wav_write(out / f"{time.strftime('%H%M%S')}_dual.wav", audio)
                with jobs_lock:
                    jobs.append(("final", audio, hold))
            elif pressed:
                if cap is not None:
                    cap.append(x)
                el = time.time() - t_press
                if el - last_chunk >= a.chunk_s:
                    last_chunk = el
                    state["chunk_i"] += 1
                    audio = np.concatenate(cap) if cap else np.zeros(0, np.float32)
                    print(f"        · 切片 累积 {len(cap)}块 → 送 {len(audio)/SR:.2f}s",
                          flush=True)
                    if len(audio) / SR >= a.min_s:
                        with jobs_lock:
                            jobs.append(("chunk", audio, el))

    stop["v"] = True
    time.sleep(1.5)
    print(f"\n  💾 对账记录 → {out/'live.jsonl'}   音频 → {out}/*.wav")
    return 0


if __name__ == "__main__":
    sys.exit(main())
