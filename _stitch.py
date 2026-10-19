import json
from pathlib import Path
root = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")

# ── ① typein: 退格（拼接时撤回已打进去的半句）──
p = root / "typein.py"; t = p.read_text(encoding="utf-8")
if "def send_backspaces" not in t:
    anchor = "def type_text(s, dry=False, per_char_delay=0.004, verify=True):"
    new = '''def send_backspaces(n, dry=False):
    """按 n 次退格（VK_BACK=0x08）——用于"停顿被误切"后撤回半句再打整句。"""
    if n <= 0:
        return 0
    if dry:
        return n
    sent = 0
    for _ in range(int(n)):
        sent += _send_vk(0x08)
        sent += _send_vk(0x08, up=True)
        time.sleep(0.002)
    return sent


''' + anchor
    t = t.replace(anchor, new, 1); p.write_text(t, encoding="utf-8"); print("  typein: send_backspaces 已加")

# ── ② voice_agent: 拼接逻辑 ──
v = root / "voice_agent.py"; t = v.read_text(encoding="utf-8")
anchor2 = "        # ⚠️ 听写优先级【已回滚】（用户：什么都优先判定进输入框，太烂了）。"
merge = '''        # ⭐⭐ 停顿切段拼接（stitch）：实测用户一句话中间换气 ~0.5-1s 就被切成多段，
        #   后半截变成"新句子"。这里：若上一段在 stitch_gap_s 内且没说完 →
        #   ① 上一段是【听写已打进输入框】的 → 先用退格撤回，再打合并后的整句；
        #   ② 上一段没执行任何动作的 → 直接把两段文本拼起来再判。
        _gap = float(self.cfg.behavior.get("stitch_gap_s", 1.5) or 1.5)
        _st = getattr(self, "_stitch", None)
        if _st and (time.time() - _st.get("t", 0)) <= _gap:
            _prev = _st.get("text", "")
            if _prev and not _prev.rstrip().endswith(("。", "！", "？", ".", "!", "?", "…")):
                if _st.get("dictated"):
                    try:
                        from typein import send_backspaces
                        _n = int(_st.get("n_chars", 0)) + 1
                        send_backspaces(_n)
                        log(f"   ↩️ 撤回上一段 {_n} 个字符（准备打整句）", "ACT", sub=True)
                    except Exception as _e:
                        log(f"   ⚠️ 撤回失败: {type(_e).__name__}: {_e}", "WARN")
                text = _prev + text
                dur = float(dur) + float(_st.get("dur", 0))
                log(f"   ⤵️ 与上一段拼接（停顿被误切）→ 「{text[:50]}」", "ASR")
        self._stitch = None

''' + anchor2
assert anchor2 in t, "听写回滚锚点未匹配"
t = t.replace(anchor2, merge, 1)

# 听写成功后记录"已打进去的半句"，供下一段拼接时撤回
old3 = '''            self.stats["executed"] += 1
            if raw is not None:
                self._save_utt(raw, text, "dictation", 1.0, "", True, dur)'''
new3 = '''            self.stats["executed"] += 1
            # 记下"已打进输入框的半句"，若下一段在 gap 内到达且这句没说完 → 撤回重打
            self._stitch = {"text": (t or "").strip(), "t": time.time(),
                            "dictated": True, "n_chars": len(t or ""),
                            "dur": float(dur or 0)}
            if raw is not None:
                self._save_utt(raw, text, "dictation", 1.0, "", True, dur)'''
assert old3 in t, "听写成功锚点未匹配"
t = t.replace(old3, new3, 1)
v.write_text(t, encoding="utf-8"); print("  voice_agent: 拼接逻辑 + 记录已加")

# ── ③ 配置：静音判定 768ms -> 1280ms（40 帧 × 32ms）+ gap ──
p = root / "intent_config.json"; d = json.loads(p.read_text(encoding="utf-8")); b = d["behavior"]
b["vad_required_misses"] = 40          # 40×32ms=1280ms（原来 24=768ms，把换气当句尾）
b["stitch_gap_s"] = 1.5                # 相邻段间隔 <1.5s 且上段没说完 -> 拼接
b["stitch_enable"] = True
b["_comment_stitch"] = ("停顿切段修复：① vad_required_misses 24->40（768ms->1280ms，"
                        "配置里的 vad_silence_ms 在 Silero 路里没被使用，实际由 required_misses 决定）；"
                        "② stitch：相邻段 <1.5s 且上段不以句末标点结尾 -> 拼接；上段若已听写进输入框，"
                        "先用退格撤回再打整句。")
p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("  config: vad_required_misses=40, stitch_gap_s=1.5")
