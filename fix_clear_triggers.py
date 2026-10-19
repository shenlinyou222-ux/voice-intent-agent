# -*- coding: utf-8 -*-
"""清空输入的触发词收紧：只认完整短语，删掉过短的「清空」等。"""
import json
from pathlib import Path

P = Path(__file__).parent / "intent_config.json"
d = json.loads(P.read_text(encoding="utf-8"))
old = d["triggers"].get("clear_input")
d["triggers"]["clear_input"] = ["清空输入", "清空输入框", "清除输入", "清除输入框"]
d["behavior"]["_comment_clear_input"] = (
    "清空输入：只认完整短语（都含『输入』二字），删掉『清空/全删/清空一下/"
    "清除全部/全部删掉』这些过短或泛化的触发词 —— 用户要求：清空这个太短了。"
    "动作 = Ctrl+A → Delete（跨应用通用）；护栏 = UIA 判定焦点必须在可编辑控件里。"
    "误清可按 Ctrl+Z 撤销。")
P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("旧:", old)
print("新:", d["triggers"]["clear_input"])
