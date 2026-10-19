# -*- coding: utf-8 -*-
"""查"说 ollama 没转对"：找录音 → 同模型 α=0/α=20 对打 → 判断是发音还是机制问题。

判据：
  · α=20 仍完全不变 → 前缀树没生效（我的 bug），与发音无关
  · α=20 变成 Ollama/OLLAMA → 热词机制可用，只是之前 α 太小
  · 三个模型都听成同一个别的词 → 那就是音频/发音层面，不是机制
"""
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from asr_gguf import GGUFASR          # noqa: E402

HERE = Path(__file__).parent
GD17 = Path(r"C:\Users\user\Desktop\ds harness\aigc_demo_framework"
            r"\models\qwen3-asr-gguf-1.7b")
HOT = ["ollama", "Ollama", "notepad", "DeepSeek", "harness", "Trae", "Clash",
       "Steam", "Blender", "Python", "GitHub"]


def read(p):
    with wave.open(str(p), "rb") as w:
        ch, sr = w.getnchannels(), w.getframerate()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    if sr != 16000:
        t = np.linspace(0, len(x) / sr, int(len(x) * 16000 / sr), endpoint=False)
        x = np.interp(t, np.arange(len(x)) / sr, x).astype(np.float32)
    return x


def main():
    idx = HERE / "utt" / "index.jsonl"
    rows = []
    for line in idx.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        rows.append(r)
    # 找可能说了 ollama 的：转写里含 olla/欧拉/olama/阿罗 等，或近 20 条全带上
    cand = [r for r in rows if any(k in r.get("text", "").lower()
                                  for k in ("olla", "欧拉", "olama", "alama"))]
    if not cand:
        cand = [r for r in rows if r.get("src") == "mic"][-12:]
        print("（转写里没有 olla 痕迹，改为最近 12 条逐条打出来看）")
    print(f"候选 {len(cand)} 条\n")

    a = GGUFASR(gguf=GD17 / "Qwen3-ASR-1.7B-Q4_K_M.gguf",
                mmproj=GD17 / "mmproj-Qwen3-ASR-1.7B-Q4_K_M.gguf",
                n_threads=8, n_ctx=4096, language="Chinese")
    a.load()
    a.hotwords = HOT
    a._trie = None
    print(f"{'录音':<34}{'rms':>7}  {'α=0':<24}{'α=20'}")
    print("-" * 92)
    for r in cand:
        p = HERE / "utt" / r["wav"]
        if not p.exists():
            continue
        x = read(p)
        a.hotwords_score = 0.0
        t0 = a.transcribe(x)
        a.hotwords_score = 20.0
        t20 = a.transcribe(x)
        same = "（完全没变 → 前缀树没生效）" if t0 == t20 else ""
        print(f"{p.name[:32]:<34}{r.get('rms', 0):>7}  {t0!r:<24}{t20!r} {same}")
    a.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
