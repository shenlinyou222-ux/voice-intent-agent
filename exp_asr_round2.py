# -*- coding: utf-8 -*-
"""第二轮 ASR 实验：把"英文名到底错在哪"摊开 + 量化档位对比。

第一轮结论（40 条含英文名的干净 TTS）：
    ① 无上下文      67.5% / 0.900 / 273ms
    ② 热词          70.0% / 0.928 / 484ms
    ⑤ Q8_0+热词     72.5% / 0.916 / 802ms
第二轮要回答：
    A. 只喂英文名（174 字）能不能保住收益、把 prefill 拉回来？
    B. 目标句「启动DeepSeek / DeepSeek Harness」到底能不能转对？
    C. bf16（1439MB，无量化）的上限是多少？→ 判断是不是量化的锅
    D. 失败到底长什么样（逐条打印错例，找可修的模式）
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, hotword_context      # noqa: E402
from asr_gguf import GGUFASR, _load_wav          # noqa: E402

HERE = Path(__file__).parent
GD = Path(r"C:\Users\user\Desktop\ds harness\aigc_demo_framework\models\qwen3-asr-gguf")
Q4 = GD / "Qwen3-ASR-0.6B-Q4_K_M.gguf"
Q8 = GD / "Qwen3-ASR-0.6B-Q8_0.gguf"
BF = GD / "Qwen3-ASR-0.6B-bf16.gguf"
MM = GD / "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf"

# 重点词：被听错的那些
KEY = ("deepseek", "harness", "glm", "blender", "ollama", "github", "powershell",
       "huggingface", "modelscope", "steam", "python", "node", "clash", "trae")


def norm(s):
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", (s or "").lower())


def sim(a, b):
    a, b = norm(a), norm(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def main():
    cfg = Config(str(HERE / "intent_config.json"))
    rows = [json.loads(l) for l in
            open(HERE / "intent_dataset.jsonl", encoding="utf-8")]
    en = json.loads((HERE / "model" / "meta.json").read_text(encoding="utf-8"))["en_words"]
    picks = [(HERE / "wavs" / (r["id"] + ".wav"), r["text"]) for r in rows
             if re.search(r"[A-Za-z]", r["text"])]
    picks = [(p, t) for p, t in picks if p.exists()]
    # 重点句优先（含 deepseek/harness 等），再补满 24 条
    key = [(p, t) for p, t in picks if any(k in t.lower() for k in KEY)]
    other = [(p, t) for p, t in picks if (p, t) not in key]
    picks = (key + other)[:24]
    print(f"测试集 {len(picks)} 条（其中含重点词 {len(key)} 条）")
    for p, t in key[:8]:
        print(f"   {t}")

    short = hotword_context(cfg, extra=en, ascii_only=True)
    full = hotword_context(cfg, extra=en, ascii_only=False)
    print(f"\n短热词表 {len(short)} 字 / 全表 {len(full)} 字")

    CONFIGS = [
        ("① 无上下文 Q4", dict(gguf=Q4, mmproj=MM, language="Chinese")),
        ("② 短热词 Q4", dict(gguf=Q4, mmproj=MM, language="Chinese", context=short)),
        ("③ 全表热词 Q4", dict(gguf=Q4, mmproj=MM, language="Chinese", context=full)),
        ("④ 短热词 Q8_0", dict(gguf=Q8, mmproj=MM, language="Chinese", context=short)),
        ("⑤ 短热词 bf16", dict(gguf=BF, mmproj=MM, language="Chinese", context=short)),
    ]
    only = sys.argv[1:] if len(sys.argv) > 1 else None

    out = {}
    for name, kw in CONFIGS:
        if only and not any(o in name for o in only):
            continue
        if not Path(kw["gguf"]).exists():
            print(f"\n{name}: 缺权重 {Path(kw['gguf']).name}")
            continue
        print(f"\n{'='*74}\n{name}\n{'='*74}")
        asr = GGUFASR(n_threads=6, **kw)
        try:
            asr.load()
        except Exception as e:
            print(f"  载入失败: {type(e).__name__}: {e}")
            continue
        ex = sm = 0.0
        n = 0
        t0 = time.time()
        misses = []
        for p, truth in picks:
            w, _ = _load_wav(p)
            try:
                got = asr.transcribe(w)
            except Exception as e:
                got = f"<err {type(e).__name__}>"
            s = sim(truth, got)
            ex += int(norm(truth) == norm(got))
            sm += s
            n += 1
            if norm(truth) != norm(got):
                misses.append((truth, got, s))
        dt = (time.time() - t0) / max(n, 1)
        print(f"  → 完全一致 {ex:.0f}/{n} = {ex/max(n,1)*100:.1f}%  "
              f"相似度 {sm/max(n,1):.3f}  {dt*1000:.0f}ms/句")
        print(f"  错例 {len(misses)} 条（前 10）:")
        for truth, got, s in misses[:10]:
            print(f"    期望「{truth}」")
            print(f"    得到「{got}」  相似度 {s:.2f}")
        out[name] = (ex / max(n, 1), sm / max(n, 1), dt * 1000)
        asr.unload()
        del asr

    print(f"\n{'='*74}\n汇总\n{'='*74}")
    print(f"{'配置':<20}{'完全一致':>10}{'相似度':>10}{'ms/句':>9}")
    for k, (e, s, ms) in out.items():
        print(f"{k:<20}{e*100:>9.1f}%{s:>10.3f}{ms:>9.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
