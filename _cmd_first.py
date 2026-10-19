from pathlib import Path
v = Path(r"C:\Users\user\Desktop\ds harness\voice_agent\voice_agent.py")
t = v.read_text(encoding="utf-8")
# ① 「不像人话」分支：先看是不是命令；是命令就放行（别推去听写）
old1 = '            self._dictate_async(text, dur, raw, "不像人话")'
new1 = ('            # ⭐⭐ 命令永远优先于听写（血债）：实测「清空输入。」被判"不像人话"\n'
        '            #   （plausible 不认识触发词类命令）-> 直接进了听写 -> 命令本身被\n'
        '            #   打进输入框 ✗。这里先 parse 一次：认得出命令就【放行】继续往下走\n'
        '            #   （规则直通会执行它），认不出才听写。\n'
        '            try:\n'
        '                _ri_c, _ = parse(text, self.cfg)\n'
        '            except Exception:\n'
        '                _ri_c = "unknown"\n'
        '            if _ri_c != "unknown":\n'
        '                log(f"   ⚙️ 不是人话但识别到命令 {_ri_c} -> 放行给规则直通", "INT")\n'
        '            else:\n'
        '                self._dictate_async(text, dur, raw, "不像人话")\n'
        '                return')
if old1 in t and "_ri_c, _ = parse(text, self.cfg)" not in t:
    t = t.replace(old1, new1, 1); print("  ✅ 「不像人话」分支：命令优先")
# ② 强制听写分支：只在【不是命令】时才吃
old2 = '    if _ti_force.get_mode() == "on":'
new2 = ('    if _ti_force.get_mode() == "on":\n'
        '        try:\n'
        '            _ri_f, _ = parse(text, self.cfg)\n'
        '        except Exception:\n'
        '            _ri_f = "unknown"\n'
        '        if _ri_f != "unknown":\n'
        '            log(f"   ⚙️ 强制听写模式下识别到命令 {_ri_f} -> 正常执行", "INT")\n'
        '        else:')
if old2 in t and "强制听写模式下识别到命令" not in t:
    t = t.replace(old2, new2, 1)
    # 把原来紧跟其后的两行缩进一档（变成 else 分支里）
    t = t.replace('        else:\n        log("   ⌨️ 强制听写模式 -> 直接打进输入框（跳过意图门控）", "INT")\n'
                  '        self._dictate_async(text, dur, raw, "forced")\n        return',
                  '        else:\n            log("   ⌨️ 强制听写模式 -> 直接打进输入框（跳过意图门控）", "INT")\n'
                  '            self._dictate_async(text, dur, raw, "forced")\n            return', 1)
    print("  ✅ 强制听写分支：命令优先")
v.write_text(t, encoding="utf-8")
