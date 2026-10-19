# -*- coding: utf-8 -*-
"""对照实验：Qwen3-ASR 的【热词上下文】和【语言提示】到底影响多大。

现场问题（用户原话）：「启动deepseek转写烂完了」
日志实证：
    「启动DeepSeek Harness」→ 解「启动DVC Harness的。」
    「启动GLM」             → 解「G.M的。」
    「启动记事本」相关        → 解「骑士版。」
根因假设：官方 chat_template.json 里有 system 消息（ns.system_text）作为
上下文通道，我们没给 → 专有名词只能硬猜汉字。

本实验在【真实语料里含英文名的片段】上，逐项对照：
  ① 现状（无上下文，language Chinese）
  ② 加热词上下文
  ③ 加热词 + 不加语言提示（让模型自己判语言）
  ④ 加热词 + 官方 <|audio_start|>/<|audio_end|> 标记
  ⑤ Q8_0 权重（排除 Q4_K_M 量化的锅）
指标：整句完全一致率 + 字符级相似度（去标点/大小写后）
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, hotword_context      # noqa: E402
from asr_gguf import GGUFASR, _load_wav          # noqa: E402

HERE = Path(__file__).parent
GGUF_DIR = Path(r"C:\Users\user\Desktop\ds harness\aigc_demo_framework\models\qwen3-asr-gguf")
Q4 = GGUF_DIR / "Qwen3-ASR-0.6B-Q4_K_M.gguf"
Q8 = GGUF_DIR / "Qwen3-ASR-0.6B-Q8_0.gguf"
MMPROJ = GGUF_DIR / "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"
N_CLIPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def norm(s):
    """比较用归一化：去标点/空白，英文小写"""
    s = (s or "").lower()
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "", s)
    return s


def sim(a, b):
    """字符级相似度（1 - 编辑距离/较长者）"""
    a, b = norm(a), norm(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def build_set(cfg):
    """取含英文名的语料片段（有 wav 的）"""
    rows = [json.loads(l) for l in
            open(HERE / "intent_dataset.jsonl", encoding="utf-8")]
    en_words = [w for w in json.loads(
        (HERE / "model" / "meta.json").read_text(encoding="utf-8"))["en_words"]]
    picks = []
    for r in rows:
        t = r["text"]
        if not re.search(r"[A-Za-z]", t):
            continue
        p = HERE / "wavs" / (r["id"] + ".wav")
        if p.exists():
            picks.append((p, t))
    picks.sort(key=lambda x: (x[1], x[0].name))
    return picks[:N_CLIPS], en_words


def main():
    cfg = Config(str(HERE / "intent_config.json"))
    picks, en_words = build_set(cfg)
    print(f"测试集：{len(picks)} 条含英文名的语料片段")
    for p, t in picks[:6]:
        print(f"   {p.name[:30]:32s} {t}")
    if len(picks) > 6:
        print(f"   ... 共 {len(picks)} 条")

    ctx = hotword_context(cfg, extra=en_words)
    print(f"\n热词上下文（{len(ctx)} 字）: {ctx[:150]}...")

    CONFIGS = [
        ("① 现状(无上下文)", dict(gguf=Q4, mmproj=MMPROJ, language="Chinese")),
        ("② 热词上下文", dict(gguf=Q4, mmproj=MMPROJ, language="Chinese",
                            context=ctx)),
        ("③ 热词+不指定语言", dict(gguf=Q4, mmproj=MMPROJ, language=None,
                              context=ctx)),
        ("④ 热词+audio标记", dict(gguf=Q4, mmproj=MMPROJ, language="Chinese",
                             context=ctx, use_audio_tags=True)),
        ("⑤ Q8_0+热词", dict(gguf=Q8, mmproj=MMPROJ, language="Chinese",
                           context=ctx)),
    ]

    results = {}
    for name, kw in CONFIGS:
        print(f"\n{'='*74}\n{name}\n{'='*74}")
        asr = GGUFASR(n_threads=6, **kw)
        try:
            asr.load()
        except Exception as e:
            print(f"  载入失败: {type(e).__name__}: {e}")
            continue
        t0, ex, sm, n = time.time(), 0, 0.0, 0
        for p, truth in picks:
            w, sr = _load_wav(p)
            try:
                got = asr.transcribe(w)
            except Exception as e:
                got = f"<err {type(e).__name__}>"
            s = sim(truth, got)
            sm += s
            ex += int(norm(truth) == norm(got))
            n += 1
            if n <= 8:
                flag = "✅" if norm(truth) == norm(got) else "❌"
                print(f"  {flag} 期望「{truth}」")
                print(f"     得到「{got}」  相似度 {s:.2f}")
        dt = time.time() - t0
        results[name] = (ex / max(n, 1), sm / max(n, 1), dt / max(n, 1))
        print(f"  → 完全一致 {ex}/{n} = {ex/max(n,1)*100:.1f}%  "
              f"平均相似度 {sm/max(n,1):.3f}  {dt/max(n,1)*1000:.0f}ms/句")
        asr.unload()
        del asr

    print(f"\n{'='*74}\n汇总（{len(picks)} 条含英文名的片段）\n{'='*74}")
    print(f"{'配置':<22}{'完全一致':>10}{'平均相似度':>12}{'ms/句':>9}")
    for name, (ex, sm, ms) in results.items():
        print(f"{name:<22}{ex*100:>9.1f}%{sm:>12.3f}{ms*1000:>9.0f}")
    if results:
        best = max(results.items(), key=lambda kv: kv[1][1])
        print(f"\n最佳: {best[0]}  相似度 {best[1][1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
