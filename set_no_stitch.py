# -*- coding: utf-8 -*-
"""用户要"边说边打"：关掉 2s 段拼接 -> 每段（704ms 静音判定）一完成就立刻打进输入框。
代价：停顿处会被切成两段分别打字（不再退格拼接重打）。"""
import json
from pathlib import Path

p = Path(__file__).parent / "intent_config.json"
d = json.loads(p.read_text(encoding="utf-8"))
b = d["behavior"]
b["stitch_gap_s"] = 0.0          # 关闭拼接 = 立刻打字
b["dictation_add_space"] = True
b["_comment_stitch_off"] = (
    "用户要『边说边打』：关掉 2s 段拼接 -> 每段(704ms 静音判定)一完成就立刻打进输入框，"
    "感知延迟 ~0.8s。代价：停顿处会切成两段分别打字。要更接近真流式，见 STREAMING_PLAN.md。")
p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("  stitch_gap_s =", b["stitch_gap_s"])
print("  vad_required_misses =", b["vad_required_misses"], "(≈704ms 静音判定)")
print("  asr_backend =", b["asr_backend"], "| intent_backend =", b["intent_backend"])
