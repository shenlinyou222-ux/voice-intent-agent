# -*- coding: utf-8 -*-
"""把 PTT（按住说话）开关写进 intent_config.json。

刻意只加键、不改别的：回滚就是把 trigger_mode 改回 "vad"（一行）。
写之前先备份 intent_config.json.bak-preptt。
"""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
p = HERE / "intent_config.json"
bak = HERE / "intent_config.json.bak-preptt"
if not bak.exists():
    shutil.copy2(p, bak)
    print(f"  备份 → {bak.name}")

d = json.loads(p.read_text(encoding="utf-8"))
b = d["behavior"]

b["trigger_mode"] = "ptt"          # ptt=按住说话 | vad=常开（旧行为，一行回滚）
b["ptt_key"] = "RCTRL"             # 右 Ctrl：不产生字符、不用装全局钩子、不吞键
b["ptt_preroll_ms"] = 1000         # 预滚动环形缓冲：兜住"先开口再按键"（实测早说 ~300ms）
b["ptt_min_ms"] = 250              # 护栏①：按住时长下限（误碰丢弃）
b["ptt_min_speech_ms"] = 120       # 护栏②：检测到的语音跨度下限
                                   #  ⚠️ 必须量这两个，不能量"裁过之后的段长"：
                                   #     两头各补 150/300ms 余量，100ms 误碰会被补成
                                   #     ~550ms 而漏过（自测 ⑤ 实测踩中）。
b["ptt_head_keep_ms"] = 150        # 段头只留"首个语音帧前 150ms"（剪掉长段按前静音）
b["ptt_tail_keep_ms"] = 300        # 段尾留"末个语音帧后 300ms"（实测挖掉 500ms 会掉「录进去」）
b["_comment_ptt"] = (
    "按住说话（2026-10-14 用户指定）。为什么换掉常开 VAD：VAD 的段起点是【概率检测】"
    "的结果，必须靠预滚动去兜；而 StateMachine 收段时把 pre_buffer 清空、之后每块只补 1 块 "
    "—— 上一段刚收完、紧接着又说话时预滚动只剩几块，触发又要等「5 块平滑 + hits」≈200~300ms "
    "→ 段起点落到起音【之后】= 从说话中间切进来。血债实测（同一句话两条路）："
    "utt/162207_mic_dictation_就好像可以了.wav 段头 = 段内 RMS ×2.35（整段最响就在开头），"
    "转写「就好像可以了。」吃掉「感」；而 162212 那条转写完整「感觉好像可以了吧。」"
    "→ 用户输入框里得到「就好像可以了。感觉好像可以了吧。」= 断头残句 + 完整重复。"
    "按键是【已知事件】，段起点 = 按下 − preroll，起音不需要被检测出来，结构上不存在这条路。"
    "顺带收益：①不按就不听（噪声/对话不再切段、不再往输入框打字）；"
    "②松手即送 ASR，省掉 704ms 静音等待（实测 142~155ms 出字，STREAMING_PLAN §5 目标 <0.3s 达成，"
    "不用起 vLLM realtime、不用腾显存、不用写 WS 回退链）；"
    "③段尾裁静音（Qwen3-ASR 会在静音上编词，实测 30s 近静音被转成「《中国好声音》第二季…」）。"
    "代价：拿不到「边说边出字」，要那个才需要流式。回滚：trigger_mode 改回 \"vad\"。"
    "VAD 那条路本身也修了 pre_buffer.clear()（改成一直在滚的环形缓冲），"
    "前后对照：旧代码第 2 段预滚动 0~1 块、段首 0.67× 段内 RMS；新代码满 19~20 块、段首 0.00×。")

p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("  已写入：")
for k in ("trigger_mode", "ptt_key", "ptt_preroll_ms", "ptt_min_ms",
          "ptt_min_speech_ms", "ptt_head_keep_ms", "ptt_tail_keep_ms"):
    print(f"    {k} = {b[k]}")
