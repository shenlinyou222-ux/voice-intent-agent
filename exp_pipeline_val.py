# -*- coding: utf-8 -*-
"""端到端代价测量：砍掉音频塔后，走完整流水线的准确率是多少。

不是"纯模型准确率"，而是【ASR 真实转写 → 纯文本路 → 规则优先 → 门控】之后
最终判定的意图 —— 这才是用户能感知的代价。

做法：monkeypatch voice_agent.dispatch 记录每次真实派发，逐条喂 val wav。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import voice_agent as va                     # noqa: E402
from actions import Config                   # noqa: E402
from model_runner import read_wav            # noqa: E402

HERE = Path(__file__).parent
CAP = []
_orig = va.dispatch


def _rec(intent, arg, cfg, dry=None):
    CAP.append((intent, arg))
    return _orig(intent, arg, cfg, dry=True)


va.dispatch = _rec


def main():
    rows = [json.loads(l) for l in
            open(HERE / "intent_dataset.jsonl", encoding="utf-8")]
    val = [(r, HERE / "wavs" / (r["id"] + ".wav")) for r in rows
           if r.get("split") == "val"]
    val = [(r, p) for r, p in val if p.exists()]
    n = len(val)
    print(f"val {n} 条，端到端（ASR+纯文本路+规则+门控）")

    cfg = Config(str(HERE / "intent_config.json"))
    a = va.Agent(cfg, model_dir=str(HERE / "model"), dry=True)
    a.lazy = False
    a.load_model()
    if a.asr_backend in ("own", "gguf"):
        a.asr = a._make_asr()
        a.asr.load()

    ok = 0
    conf_sum = 0.0
    errs = []
    t0 = time.time()
    for k, (r, p) in enumerate(val):
        CAP.clear()
        try:
            a.handle(read_wav(str(p)))
        except Exception as e:
            errs.append((r["text"], r["intent"], f"<err {type(e).__name__}>"))
            continue
        got = CAP[-1][0] if CAP else "unknown"
        if got == r["intent"]:
            ok += 1
        else:
            errs.append((r["text"], r["intent"], got))
        if (k + 1) % 100 == 0:
            print(f"  ... {k+1}/{n}  准确率 {ok/(k+1)*100:.1f}%  "
                  f"{(time.time()-t0):.0f}s", flush=True)

    print(f"\n{'='*70}\n端到端准确率 {ok}/{n} = {ok/n*100:.2f}%"
          f"（{time.time()-t0:.0f}s）\n{'='*70}")
    print(f"错例 {len(errs)} 条（前 25）:")
    for text, truth, got in errs[:25]:
        print(f"   真={truth:14s} 判={got:14s} 「{text}」")
    per = {}
    for r, p in val:
        per.setdefault(r["intent"], [0, 0])[1] += 1
    for text, truth, got in errs:
        pass
    # 逐意图（用 CAP 无法回溯，简化：只统计错误分布）
    from collections import Counter
    c = Counter(f"{t}→{g}" for _, t, g in errs)
    print("\n错误类型 top10:")
    for k, v in c.most_common(10):
        print(f"   {k:34s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
