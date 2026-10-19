import json
from pathlib import Path
root = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")

# ══ ① typein.py：听写模式状态 + 自动找/聚焦输入框 ══
p = root / "typein.py"; t = p.read_text(encoding="utf-8")
if "DICTATION_MODE" not in t:
    t += '''

# ══════════════════════════════════════════════════════════════════════
# 听写模式（「开始输入」/「终止输入」语音控制）
#   auto = 老行为：光标在输入框里才听写
#   on   = 强制听写（并可先自动聚焦当前窗口的输入框）
#   off  = 终止听写（非命令的话不再往输入框打）
# ══════════════════════════════════════════════════════════════════════
DICTATION_MODE = "auto"


def set_mode(m):
    global DICTATION_MODE
    DICTATION_MODE = str(m or "auto").lower()
    return DICTATION_MODE


def get_mode():
    return DICTATION_MODE


def find_input_box(max_nodes=4000):
    """在当前前台窗口里找【可能的输入框】（UIA 遍历，返回第一个可编辑控件）。

    为什么要这个：用户不想每次都手动点输入框 —— 说「开始输入」时自动找到并聚焦。
    只认 EditControl（+ 支持 ValuePattern 的 TextControl），排除只读/不可见/禁用。
    """
    try:
        import uiautomation as auto
        w = auto.GetForegroundWindow()
        root = auto.ControlFromHandle(w) if w else None
        if root is None:
            return None, "拿不到前台窗口"
        n = 0
        for c in root.GetChildren():
            pass
        def walk(node, depth=0):
            nonlocal n
            if depth > 12 or n > max_nodes:
                return None
            try:
                kids = node.GetChildren()
            except Exception:
                kids = []
            for c in kids:
                n += 1
                try:
                    if c.ControlTypeName in ("EditControl", "ComboBoxControl"):
                        if c.IsEnabled and not getattr(c, "IsReadOnly", False):
                            return c
                    if c.ControlTypeName == "TextControl" and c.IsEnabled:
                        try:
                            if c.GetValuePattern():
                                return c
                        except Exception:
                            pass
                except Exception:
                    pass
                r = walk(c, depth + 1)
                if r is not None:
                    return r
            return None
        box = walk(root)
        if box is None:
            return None, "没找到输入框"
        return box, "found"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def focus_input():
    """自动聚焦当前窗口的输入框 -> (ok, msg)"""
    box, why = find_input_box()
    if box is None:
        return False, f"没找到可聚焦的输入框（{why}）"
    try:
        box.SetFocus()
        time.sleep(0.15)
        return True, f"已聚焦输入框（{box.ControlTypeName} / {(box.Name or '')[:24]}）"
    except Exception as e:
        return False, f"聚焦失败: {type(e).__name__}: {e}"
'''
    p.write_text(t, encoding="utf-8"); print("  ✅ typein：模式状态 + find_input_box/focus_input")

# ══ ② actions.py：两条新动作（插在 ACTIONS 之前）+ 注册 ══
a = root / "actions.py"; t = a.read_text(encoding="utf-8")
if "act_dictation_on" not in t:
    blk = '''def act_dictation_on(arg, cfg, dry=False):
    """「开始输入」：自动聚焦当前窗口的输入框 + 进入强制听写模式。"""
    import typein
    if dry:
        return True, "[dry] 开始输入（聚焦输入框 + 进入听写模式）", None
    ok, msg = typein.focus_input()
    typein.set_mode("on")
    if ok:
        return True, f"开始输入：{msg}；后续非命令语音都会打进该输入框（说『终止输入』停止）", None
    # 没找到输入框也允许进模式（用户可以手动点，之后照常听写）
    return True, f"已进入听写模式（但{msg}；手动点一下输入框即可）", None


def act_dictation_off(arg, cfg, dry=False):
    """「终止输入」：停止往输入框转写。"""
    import typein
    if dry:
        return True, "[dry] 终止输入", None
    typein.set_mode("off")
    return True, "已终止输入：非命令语音不再打进输入框（说『开始输入』恢复）", None


ACTIONS = {'''
    assert "ACTIONS = {" in t
    t = t.replace("ACTIONS = {", blk, 1)
    t = t.replace('    "clear_input": act_clear_input,',
                  '    "dictation_on": act_dictation_on,\n'
                  '    "dictation_off": act_dictation_off,\n'
                  '    "clear_input": act_clear_input,', 1)
    a.write_text(t, encoding="utf-8"); print("  ✅ actions：act_dictation_on/off + 注册")

# ══ ③ voice_agent.py：_dictate 遵守模式 + 规则直通白名单 ══
v = root / "voice_agent.py"; t = v.read_text(encoding="utf-8")
anchor = "    def _dictate(self, text, dur, raw=None):"
ins = '''    def _dictate(self, text, dur, raw=None):
        # ⭐ 听写模式（语音「开始输入」/「终止输入」控制）
        #   off = 明确终止 -> 不打；on = 强制打（跳过"光标在输入框"判定）；
        #   auto = 老行为（光标在输入框才打）
        try:
            import typein as _ti
            _mode = _ti.get_mode()
        except Exception:
            _mode = "auto"
        if _mode == "off":
            return False
        self._dictation_forced = (_mode == "on")
'''
if anchor in t and "_dictation_forced" not in t:
    t = t.replace(anchor + "\n", ins, 1)
    # 在原来的 in_text_field 判定处：强制模式直接跳过
    old2 = """            ok_field, why = in_text_field()
        except Exception as e:
            log(f"   ⚠️ 光标判定失败: {type(e).__name__}: {e}", "WARN")
            return False
        if not ok_field:
            return False                     # 不在输入框 → 交回原来的拒识逻辑"""
    new2 = """            ok_field, why = in_text_field()
        except Exception as e:
            log(f"   ⚠️ 光标判定失败: {type(e).__name__}: {e}", "WARN")
            return False
        if getattr(self, "_dictation_forced", False):
            why = "强制听写模式(开始输入) -> " + str(why)
        elif not ok_field:
            return False                     # 不在输入框 → 交回原来的拒识逻辑"""
    if old2 in t:
        t = t.replace(old2, new2, 1); print("  ✅ voice_agent：_dictate 遵守模式（含强制）")
    else:
        print("  ⚠️ _dictate 的 in_text_field 判定锚点未匹配（模式仍生效于 off/on 的开头判断）")
    # 规则直通白名单
    oldw = '        if rule_intent == "clear_input":'
    neww = ('        if rule_intent in ("dictation_on", "dictation_off"):\n'
            '            _rule_arg_r = ""            # 无参数命令\n'
            '        elif rule_intent == "clear_input":')
    if oldw in t:
        t = t.replace(oldw, neww, 1); print("  ✅ voice_agent：规则直通白名单已加 dictation_on/off")
    v.write_text(t, encoding="utf-8")

# ══ ④ 配置：触发词 ══
p2 = root / "intent_config.json"; d = json.loads(p2.read_text(encoding="utf-8"))
d["triggers"]["dictation_on"] = ["开始输入", "开始听写", "进入输入模式", "开始转写"]
d["triggers"]["dictation_off"] = ["终止输入", "停止输入", "结束输入", "停止听写", "退出输入"]
d["behavior"]["dictation_mode_default"] = "auto"
d["behavior"]["_comment_dictation_mode"] = ("听写模式：auto(光标在输入框才听写) / on(强制，"
    "由「开始输入」触发并自动聚焦输入框) / off(由「终止输入」触发)。"
    "「开始输入」会用 UIA 在当前前台窗口里找第一个可编辑控件并 SetFocus。")
p2.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("  ✅ 配置：dictation_on/off 触发词已加")

# ══ ⑤ 语料加示例（演示"加意图=加示例"）+ 重建索引 ══
ds = root / "intent_dataset.jsonl"
rows = [json.loads(l) for l in ds.read_text(encoding="utf-8").splitlines() if l.strip()]
have = {r["intent"] for r in rows}
EX = {
 "dictation_on": ["开始输入", "开始听写", "进入输入模式", "帮我开始输入", "我要开始打字",
                  "开始语音输入", "打开听写", "现在开始输入文字"],
 "dictation_off": ["终止输入", "停止输入", "结束输入", "停止听写", "别输入了",
                   "退出输入模式", "不要打字了", "停一下输入"],
}
mx = max((r.get("id", 0) for r in rows), default=0)
add = 0
for it, texts in EX.items():
    for txt in texts:
        mx += 1
        rows.append({"id": mx, "text": txt, "intent": it, "kind": "rule",
                     "split": "train", "arg": ""})
        add += 1
ds.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
print("  ✅ 语料 +%d 条示例（dictation_on/off）" % add)
