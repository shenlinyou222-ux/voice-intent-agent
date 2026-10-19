from pathlib import Path
P = Path("voice_agent.py"); t = P.read_text(encoding="utf-8")
orig = t
# ① 加一个异步听写助手（避免 UIA 阻塞音频主循环）
helper = '''    def _dictate_async(self, text, dur, raw=None, why=""):
        """后台线程里做听写。

        ⚠️ 必须异步：听写里要读 UIA（大文档要几百 ms 甚至更久），
           同步执行会把音频主循环卡住 → 队列爆满 → 实测丢了 362 块(≈11 秒音频)。
        """
        import threading

        def _work():
            try:
                self._dictate(text, dur, raw)
            except Exception as e:
                log(f"   ⚠️ 听写线程异常: {type(e).__name__}: {e}", "WARN")

        threading.Thread(target=_work, daemon=True, name="dictate").start()

    # ── 听写：把转写打进当前焦点输入框 ──
'''
anchor = "    # ── 听写：把转写打进当前焦点输入框 ──\n"
assert anchor in t, "锚点①没找到"
t = t.replace(anchor, helper, 1)
# ② 两个调用点改成异步
old1 = "            if self._dictate(text, dur, raw):\n                return"
new1 = "            self._dictate_async(text, dur, raw, \"不像人话\")\n            return"
if old1 in t:
    t = t.replace(old1, new1)
old2 = "        if final_intent == \"unknown\" and self._dictate(text, dur, raw):\n            return"
new2 = "        if final_intent == \"unknown\":\n            self._dictate_async(text, dur, raw, \"unknown\")\n            return"
if old2 in t:
    t = t.replace(old2, new2)
P.write_text(t, encoding="utf-8")
print("  异步听写已接上；改动:", t != orig)
