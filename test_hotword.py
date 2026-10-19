# -*- coding: utf-8 -*-
"""测热词引导 + ASCII 清洗 —— 专治英文专有名词

只测【实际决定路径】：转写 → 规则解析 → 闭集 snap → 能不能命中正确条目。
（闭集命令的意图由规则直接查闭集决定，比模型泛化可靠。）

对照：
  A. 基线（无热词）
  B/C/D. 热词引导 α=1 / 2 / 4
  E. 热词 α=2 但关掉 ASCII 清洗（看它单独贡献多少）
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch

import actions as A
from actions import Config, resolve, resolve_closed, parse

CFG = Config(HERE / "intent_config.json")

CASES = [
    ("打开Blender", "Blender"), ("启动Blender", "Blender"),
    ("运行Blender", "Blender"), ("帮我打开Blender", "Blender"),
    ("启动Ollama", "Ollama"), ("打开Ollama", "Ollama"),
    ("启动PowerShell", "PowerShell"), ("运行PowerShell", "PowerShell"),
    ("启动Python", "Python"), ("打开Steam", "Steam"),
    ("启动Steam", "Steam"), ("打开Trae", "Trae"),
    ("启动Clash", "Clash"), ("打开Node", "Node"),
    ("打开Vulkan测试", "Vulkan测试"), ("启动Edge浏览器", "Edge浏览器"),
    ("打开GitHub", "GitHub"), ("打开HuggingFace", "HuggingFace"),
    ("打开ModelScope", "ModelScope"), ("上CSDN", "CSDN"),
]

outdir = HERE / "_hot_wavs"
outdir.mkdir(exist_ok=True)
tsv = outdir / "jobs.tsv"
tsv.write_text("\n".join(f"h{i}\t{t}" for i, (t, _) in enumerate(CASES)), encoding="utf-8")
tts = Path(r"C:\Users\user\Desktop\ds harness\voice_intent_router\asr_internal\tts_batch.ps1")
if tts.exists():
    subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", str(tts), "-Jobs", str(tsv), "-OutDir", str(outdir)],
                   capture_output=True, text=True)
print(f"  TTS {len(list(outdir.glob('*.wav')))} 个 wav")

from model_runner import IntentModel, read_wav
m = IntentModel(model_dir=HERE / "model", device="cpu")
HOT = m.hotwords_from_config(CFG)
print(f"  热词 {len(HOT)} 个: {HOT}")

# 预抽 logits（encoder 只跑一次）
LOGITS = []
with torch.no_grad():
    for i in range(len(CASES)):
        w = outdir / f"h{i}.wav"
        if not w.exists():
            LOGITS.append(None); continue
        f = m.fe(read_wav(str(w)), sampling_rate=16000, return_tensors="pt",
                 padding=False, return_attention_mask=True)
        inp, msk = f["input_features"], f["attention_mask"].long()
        pad = (-inp.shape[-1]) % 100
        if pad:
            inp = torch.nn.functional.pad(inp, (0, pad))
            msk = torch.nn.functional.pad(msk, (0, pad))
        LOGITS.append(m.ctc(m.A(inp, msk).last_hidden_state))


def run(alpha, ascii_clean=True, first_alpha=0.0):
    orig = A._snap
    if not ascii_clean:
        import functools
        A._snap = functools.partial(orig, max_ratio=0.50)
        _real = orig
        def _noclean(arg, cands, max_ratio=0.50):
            # 复制一份去掉 ASCII 清洗的逻辑
            arg = (arg or "").strip()
            if not arg:
                return None
            al = arg.lower()
            def _best(probes):
                b, bd = None, 10 ** 9
                for c in cands:
                    for p in probes:
                        d = A._lev(p, c.lower())
                        if d < bd:
                            bd, b = d, c
                return b, bd
            b, bd = _best({al})
            if b and bd <= max(1, int(len(b) * max_ratio)):
                return b
            probes = {al}
            for i in range(1, len(al)):
                probes.add(al[i:]); probes.add(al[:i])
            b, bd = _best(probes)
            return b if (b and bd <= max(1, int(len(b) * max_ratio))) else None
        A._snap = _noclean

    ok, det = 0, []
    for i, (text, want) in enumerate(CASES):
        lg = LOGITS[i]
        if lg is None:
            det.append((text, "", "", None)); continue
        hyp = m.ctc_decode_hot(lg, HOT, alpha, first_alpha) if (alpha > 0 or first_alpha > 0) else m.ctc_decode(lg)
        it, _ = parse(hyp, CFG)
        hit = None
        if it in ("launch_app", "open_url", "open_file"):
            hit = resolve_closed(it, resolve(it, hyp, CFG), CFG)
        good = hit is not None and hit["name"] == want
        ok += int(good)
        det.append((text, hyp, it, hit["name"] if hit else None))
    A._snap = orig
    return ok / len(CASES), det


print()
print("=" * 94)
print("  命中率（20 条含英文专有名词的命令）")
print("=" * 94)
res = {}
for name, alpha, clean, fa in [("A. 基线（无热词）", 0, True, 0.0),
                           ("B. 词首偏置 1", 0, True, 1.0),
                           ("C. 词首偏置 2", 0, True, 2.0),
                           ("D. 词首偏置 3", 0, True, 3.0),
                           ("E. 词首 2 + 词中 2", 2, True, 2.0),
                           ("F. 词首 4", 0, True, 4.0)]:
    acc, det = run(alpha, clean, fa)
    res[name] = (acc, det)
    print(f"  {name:30s} {acc*100:6.1f}%")

print()
print("=" * 94)
print("  转写对照（基线 → 热词 α=2）")
print("=" * 94)
d0 = res["A. 基线（无热词）"][1]
d2 = res["D. 词首偏置 3"][1]
for (t, h0, _, a0), (_, h2, _, a2) in zip(d0, d2):
    mark = "  " if h0 == h2 else "★"
    print(f"  {mark} {t:20s} 基线 {h0!r:22s} → 偏置 {h2!r:22s}  命中 {a2}")
