import re
from pathlib import Path
v = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\voice_agent.py")
t = v.read_text(encoding="utf-8")
if "_ti_force" in t:
    print("  已有，跳过")
else:
    m = re.search(r"\n(\s+)thr = self\.cfg\.behavior\.get\(\"conf_threshold\"", t)
    assert m, "找不到置信度门"
    ind = m.group(1)
    blk = ("\n" + ind + "# ⭐⭐ 强制听写模式（语音「开始输入」）：不是命令的话全部打进输入框，\n"
           + ind + "#   而且必须在【置信度门之前】—— 血债：句向量分类器的 conf 是【余弦相似度】\n"
           + ind + "#   (0.3~1.0)，而 conf_threshold=0.55 是给旧模型 softmax 概率定的 ✗，\n"
           + ind + "#   自由说的话常低于 0.55 -> 在到达听写兜底之前就被拒识 -> 用户实测感受：\n"
           + ind + "#   「开始输入之后就没法输入到输入框」✗\n"
           + ind + "try:\n"
           + ind + "    import typein as _ti_force\n"
           + ind + "    if _ti_force.get_mode() == \"on\":\n"
           + ind + "        log(\"   ⌨️ 强制听写模式 -> 直接打进输入框（跳过意图门控）\", \"INT\")\n"
           + ind + "        self._dictate_async(text, dur, raw, \"forced\")\n"
           + ind + "        return\n"
           + ind + "except Exception:\n"
           + ind + "    pass\n")
    t = t[:m.start()] + blk + t[m.start():]
    v.write_text(t, encoding="utf-8")
    print("  ✅ 已插入（置信度门之前）")
import subprocess
print("  位置:", [i+1 for i, l in enumerate(t.splitlines()) if "_ti_force" in l])
