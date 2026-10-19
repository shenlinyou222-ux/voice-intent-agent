from pathlib import Path
v = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\voice_agent.py")
t = v.read_text(encoding="utf-8")
if "强制听写模式" not in t:
    import re
    m = re.search(r"\n(\s+)thr = self\.cfg\.behavior\.get\(\"conf_threshold\"", t)
    assert m, "找不到置信度门"
    ind = m.group(1)
    blk = ("\n" + ind + "# ⭐⭐ 强制听写模式（语音「开始输入」）：「开始输入」之后，凡是**不是命令**的话\n"
           + ind + "#   全部打进输入框 —— 而且必须在【置信度门之前】。\n"
           + ind + "#   血债：句向量分类器的 conf 是【余弦相似度】(0.3~1.0)，而 conf_threshold=0.55\n"
           + ind + "#   是给旧模型 softmax 概率定的 ✗ -> 自由说的话常低于 0.55，在到达听写兜底\n"
           + ind + "#   之前就被置信度门拒识 -> 用户感受：「开始输入之后就没法输入到输入框」✗\n"
           + ind + "try:\n"
           + ind + "    import typein as _ti_force\n"
           + ind + "    if _ti_force.get_mode() == \"on\":\n"
           + ind + "        log(f\"   ⌨️ 强制听写模式 -> 直接打进输入框（跳过意图门控）\", \"INT\")\n"
           + ind + "        self._dictate_async(text, dur, raw, \"forced\")\n"
           + ind + "        return\n"
           + ind + "except Exception:\n"
           + ind + "    pass\n")
    t = t[:m.start()] + blk + t[m.start():]
    v.write_text(t, encoding="utf-8")
    print("  ✅ 已在置信度门【之前】插入强制听写分支")
else:
    print("  已存在，跳过")
