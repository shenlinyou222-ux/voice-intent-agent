# -*- coding: utf-8 -*-
"""热词表实验：把【命令词】也喂给 ASR，看「会话/绘画」同音能不能纠回来。

病根证据（你的 14 条真实录音）：两个模型都把「新建会话」转成「新建绘画」，
「切换会话」→「切换绘画」。它们是完美同音（huìhuà），音频里没有任何区别 ——
模型只能靠【上下文】。而现在热词表只有 app/网址名（且只喂英文），
一个中文命令词都没有 → 模型没有任何理由选「会话」。

三组对照（都用 0.6B，同一批真实录音）：
  A 现状：只喂英文名（174 字）
  B +命令词：会话 / 新建会话 / 切换会话 / 启动 / 打开 / 搜索 / 精确查找 …
  C 全表：命令词 + 全部中文闭集名
"""
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, hotword_context          # noqa: E402
from asr_gguf import GGUFASR                         # noqa: E402

HERE = Path(__file__).parent
GD6 = Path(r"C:\Users\user\Desktop\ds harness\aigc_demo_framework\models\qwen3-asr-gguf")
GGUF = GD6 / "Qwen3-ASR-0.6B-Q4_K_M.gguf"
MM = GD6 / "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"

# 命令词：意图动作 + 会话控制
CMD_WORDS = ["会话", "新建会话", "切换会话", "新建DSH会话", "让DSH", "启动", "打开",
             "搜索", "精确查找", "文件", "网址", "播放"]


def read_wav(p):
    with wave.open(str(p), "rb") as w:
        ch, sr = w.getnchannels(), w.getframerate()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a


def main():
    cfg = Config(str(HERE / "intent_config.json"))
    en = json.loads((HERE / "model" / "meta.json").read_text(encoding="utf-8"))["en_words"]
    A = hotword_context(cfg, extra=en, ascii_only=True)
    B = hotword_context(cfg, extra=en + CMD_WORDS, ascii_only=True)
    C = hotword_context(cfg, extra=en + CMD_WORDS, ascii_only=False)

    idx = HERE / "utt" / "index.jsonl"
    clips = []
    for line in idx.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("src") == "mic":
            p = HERE / "utt" / r["wav"]
            if p.exists():
                clips.append((p, r.get("text", "")))
    clips = clips[-14:]
    print(f"真实麦克风录音 {len(clips)} 条；热词长度 A={len(A)} B={len(B)} C={len(C)} 字")
    print(f"\nB 的热词（前 120 字）: {B[:120]}…")

    res = {}
    asr = GGUFASR(gguf=GGUF, mmproj=MM, n_threads=6, language="Chinese", context=A)
    asr.load()
    for tag, ctx in (("A 现状", A), ("B +命令词", B), ("C 全表", C)):
        asr.context = ctx
        rows, t0 = [], time.time()
        for p, old in clips:
            w = read_wav(p)
            rows.append(asr.transcribe(w))
        res[tag] = rows
        print(f"\n{'='*76}\n{tag}（{len(ctx)} 字热词）\n{'='*76}")
        for (p, old), got in zip(clips, rows):
            flag = "  " if got == res["A 现状"][len(rows[:0])] else ""
            mark = "🔁" if got != old else "  "
            print(f"  {mark} {p.name[:38]:40s} 0.6B原:「{old}」")
            print(f"       {tag}  :「{got}」")
        print(f"  （{time.time()-t0:.0f}s）")
    asr.unload()

    print(f"\n{'='*76}")
    print("关键看：「会话/绘画」有没有被纠回来")
    for (p, old), a_, b_, c_ in zip(clips, res["A 现状"], res["B +命令词"], res["C 全表"]):
        if "绘画" in a_ or "绘画" in b_ or "绘画" in c_:
            print(f"  {p.name[:40]:42s}")
            print(f"     A:「{a_}」")
            print(f"     B:「{b_}」")
            print(f"     C:「{c_}」")
    return 0


if __name__ == "__main__":
    sys.exit(main())
