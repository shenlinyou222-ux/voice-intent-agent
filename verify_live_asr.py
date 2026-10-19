# -*- coding: utf-8 -*-
"""验证线上接线：ASR 真的带上了热词上下文、且 DeepSeek 能转对。

用真实语料里含 deepseek 的片段，走【和线上完全一样】的路径：
    Config → Agent._make_asr() → transcribe()
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, hotword_context, resolve, resolve_closed  # noqa: E402
from model_runner import read_wav                                     # noqa: E402
import json, re

HERE = Path(__file__).parent
cfg = Config(str(HERE / "intent_config.json"))
rows = [json.loads(l) for l in open(HERE / "intent_dataset.jsonl", encoding="utf-8")]
en = json.loads((HERE / "model" / "meta.json").read_text(encoding="utf-8"))["en_words"]

picks = []
for r in rows:
    t = r["text"]
    if re.search(r"(deepseek|harness|glm|blender|ollama)", t, re.I):
        p = HERE / "wavs" / (r["id"] + ".wav")
        if p.exists():
            picks.append((p, t))
print(f"含 deepseek/harness/glm/blender/ollama 的语料片段: {len(picks)} 条")
for p, t in picks[:10]:
    print(f"   {t}")

# 造 Agent（不装意图模型，只验证 ASR 接线）
from voice_agent import Agent
a = Agent(cfg, model_dir=str(HERE / "model"), dry=True)
a.lazy = False
asr = a._make_asr()
print(f"\nASR 对象: context={len(asr.context or '')} 字  "
      f"audio_tags={asr.use_audio_tags}  language={asr.language}")
print(f"上下文内容: {asr.context}")
asr.load()

print(f"\n{'='*72}\n逐条转写（走线上同一条 transcribe）\n{'='*72}")
ok = 0
for p, truth in picks[:10]:
    w = read_wav(str(p))
    got = asr.transcribe(w)
    def nz(s):
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", (s or "").lower())
    hit = nz(truth) == nz(got)
    ok += hit
    print(f"{'✅' if hit else '❌'} 期望「{truth}」")
    print(f"   得到「{got}」  {asr.last_ms:.0f}ms {asr.last_tokens}tok")
    # 顺带验证纠错层：转写文本 → 意图参数
    for it in ("launch_app", "open_url", "open_file"):
        arg = resolve(it, got, cfg)
        e = resolve_closed(it, arg, cfg)
        if e:
            print(f"   → 纠错层: {it}/{e['name']}  (参数 {arg!r})")
            break
print(f"\n{ok}/{min(10, len(picks))} 完全一致")
