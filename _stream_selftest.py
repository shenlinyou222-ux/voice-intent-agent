# -*- coding: utf-8 -*-
"""流式链路离线自测：真 WS + 真引擎，把 15 秒长句按真实节奏喂进去。

验证四件事（都不需要人说话）：
  ① AgentStream 能连上 vLLM /v1/realtime，append/commit/delta 都通
  ② delta → Splitter 切出的段，与离线整段切分是否一致
  ③ A 路（规则直通 + 可撤销）会在流式过程中报出哪些动作
  ④ 松手后整段 ASR 当 B 权威 → 对账（一致 / A 多出需回滚 / B 多出需补）

用法：
  E:\\voice_agent_build\\Scripts\\python.exe -X utf8 _stream_selftest.py [wav] [--fast]
"""
import json
import queue
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from agent_stream import AgentStream, pcm16                            # noqa: E402
from intent_split import (Splitter, triggers_from_cfg, boundary_from_cfg,  # noqa: E402
                          names_from_cfg, strict_shape)
from _dual_path_sim import verdict, REVERSIBLE                          # noqa: E402
from _dual_replay import akey, b_actions                                # noqa: E402
from stream_consume import StreamConsumer                               # noqa: E402
from _cut_align import asr as offline_asr                               # noqa: E402
from actions import Config                                              # noqa: E402

SR, BLOCK = 16000, 480
CFGJ = json.loads((HERE / "intent_config.json").read_text(encoding="utf-8"))
TR = triggers_from_cfg(CFGJ)
BD = boundary_from_cfg(CFGJ)
NM = names_from_cfg(CFGJ)
FAST = "--fast" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
WAV = Path(args[0]) if args else (HERE / "_dual" / "211710_dual.wav")


def main():
    if not WAV.exists():
        print(f"  找不到 {WAV}")
        return 2
    with wave.open(str(WAV), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    cfg = Config(None)
    url = str(cfg.behavior.get("stream_url") or "ws://172.20.71.101:30805/v1/realtime")

    texts = queue.Queue()
    stream = AgentStream(url, model="qwen3-asr", commit_ms=500,
                         on_text=lambda t, fin: texts.put((t, fin)),
                         on_log=lambda m: print(f"   {m}", flush=True)).start()
    for _ in range(100):                       # 等连接
        if stream.state == "ready":
            break
        time.sleep(0.1)
    if stream.state != "ready":
        print(f"  ❌ 连不上（{stream.state}: {stream.last_err}）")
        return 1
    print(f"  ✅ 已连上  音频 {WAV.name}  {len(x)/SR:.2f}s  "
          f"节奏 {'加速' if FAST else '实时(30ms/块)'}\n")

    cons = StreamConsumer(cfg, TR, BD, NM, on_log=lambda m: print(m, flush=True))
    a_actions, closed = [], []
    t0 = time.time()

    def drain():
        while True:
            try:
                t, fin = texts.get_nowait()
            except queue.Empty:
                return
            for a in cons.feed(t):
                a_actions.append((a["intent"], a["arg"], a["resolved"]))
                closed.append(a["text"])
                print(f"     +{time.time()-t0:5.1f}s ⚡A 执行 {a['intent']}({a['arg']})"
                      f"  ← 段「{a['text']}」")

    for i in range(0, len(x) - BLOCK + 1, BLOCK):
        stream.push(pcm16(x[i:i + BLOCK]))
        drain()
        if not FAST:
            time.sleep(BLOCK / SR)
    stream.commit(final=True)
    # 等解码结果：以"连续 3 秒没有新 delta"为收敛判据（别写死等待窗口）
    t_wait = time.time()
    last_n, stable = -1, 0
    while time.time() - t_wait < 40:
        drain()
        time.sleep(0.1)
        if stream.n_delta == last_n:
            stable += 1
            if stable > 30 and stream.n_delta > 0:
                break
        else:
            stable, last_n = 0, stream.n_delta
    for a in cons.flush():
        a_actions.append((a["intent"], a["arg"], a["resolved"]))
        closed.append(a["text"])
        print(f"     flush ⚡A 执行 {a['intent']}({a['arg']})  ← 段「{a['text']}」")
    print(f"     去重跳过 {cons.n_skipped_dup} 次（收尾重解/重复段）")
    stream.stop()

    print(f"\n  ══ ① 流式统计 ══")
    print(f"     delta {stream.n_delta} 条 / commit {stream.n_commit} 次 / 状态 {stream.state}")
    print(f"     切出 {len(closed)} 段：{closed}")

    print(f"\n  ══ ② 与离线整段切分对比 ══")
    whole = offline_asr(x, gain=8.0)
    off_segs = [s.text for s in Splitter(TR, boundary=BD, names=NM).segments(whole)]
    print(f"     离线整段：「{whole[:70]}」")
    print(f"     离线切 {len(off_segs)} 段：{off_segs}")
    same = len(off_segs) == len(closed)
    print(f"     段数一致：{'✅' if same else '❌'}（流式 {len(closed)} vs 离线 {len(off_segs)}）")

    print(f"\n  ══ ③ 对账（A 流式抢跑 vs B 离线整段权威）══")
    b, good, bad, gate = b_actions(cfg, None, whole)
    sa = {akey(i, a, r) for i, a, r in a_actions}
    sb = {akey(i, a, r) for i, a, why, r in b}
    print(f"     A（流式，规则直通+可撤销）：{sorted(sa)}")
    print(f"     B（离线整段权威）        ：{sorted(sb)}")
    if sa == sb:
        print(f"     ⚖ 一致 ✅")
    else:
        print(f"     ⚖ A 多出（需回滚）：{sorted(sa - sb)}")
        print(f"     ⚖ B 多出（需补执行）：{sorted(sb - sa)}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
