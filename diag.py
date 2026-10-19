# -*- coding: utf-8 -*-
"""快速诊断：看模型对几个样本的转写和意图"""
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model_runner import IntentModel, read_wav

m = IntentModel(model_dir=HERE / "model", device="cpu")
print(f"  intents = {m.INTENTS}")
print(f"  K={m.K}  tau={m.TAU}  R={m.R}")

jobs = [json.loads(l) for l in io.open(HERE / "intent_dataset.jsonl", encoding="utf-8") if l.strip()]
val = [j for j in jobs if j["split"] == "val"]
print(f"  val {len(val)} 条\n")

# 每类抽 3 条看
seen = {}
for j in val:
    seen.setdefault(j["intent"], [])
    if len(seen[j["intent"]]) < 3:
        seen[j["intent"]].append(j)

ok = 0
tot = 0
for it in sorted(seen):
    print(f"--- {it} ---")
    for j in seen[it]:
        wav = read_wav(str(HERE / "wavs" / f"{j['id']}.wav"))
        r = m.predict(wav)
        hit = r["intent"] == j["intent"]
        ok += int(hit); tot += 1
        print(f"  {'✅' if hit else '❌'} 说 {j['text']}")
        print(f"      转 {r['text']}")
        print(f"      判 {r['intent']} ({r['conf']:.3f})  期望 {j['intent']}")
print(f"\n  抽样 {ok}/{tot}")
