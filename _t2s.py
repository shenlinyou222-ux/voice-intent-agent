from pathlib import Path
root = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")

# ── ① typein：只认真正的输入框（排除 DocumentControl = 文档/聊天区）──
p = root / "typein.py"; t = p.read_text(encoding="utf-8")
old = '''        name = (c.ControlTypeName or "")
        if name in ("EditControl", "DocumentControl", "TextControl",
                    "ComboBoxControl"):
            return True'''
new = '''        name = (c.ControlTypeName or "")
        # ⚠️⚠️ 血债：DocumentControl 必须排除！实测聊天区那个 6761 字的【文档】
        #   被 UIA 报成可编辑 -> 听写把整句话"打"进文档里（看起来就是被吞了，
        #   日志显示 6761->6761 字没变化）。真正的输入框是 EditControl。
        if name in ("EditControl", "ComboBoxControl"):
            return True
        if name == "TextControl":
            # 有些编辑器把输入区报成 TextControl：再要求支持 ValuePattern
            try:
                if c.GetValuePattern():
                    return True
            except Exception:
                pass
            return False
        if name == "DocumentControl":
            return False        # 文档不是输入框'''
assert old in t, "typein 锚点未匹配"
p.write_text(t.replace(old, new, 1), encoding="utf-8"); print("  ✅ typein: 听写只认输入框（排除文档）")

# ── ② actions：简繁归一化（L0），覆盖触发词/名称里会用到的字 ──
a = root / "actions.py"; t = a.read_text(encoding="utf-8")
if "_T2S" not in t:
    anchor = "def _norm(text):"
    blk = '''# 简繁归一化（L0）：ASR 偶尔输出繁体（实测「打開知乎」「網頁版」），
# 而我们的触发词/名称都是简体 -> 规则层不认 -> 被当 unknown 吞掉。
# 这里是覆盖本项目触发词/名称所需字符的映射；更彻底的做法是装 opencc。
_T2S = str.maketrans({
    "打": "打", "開": "开", "關": "关", "閉": "闭", "網": "网", "頁": "页",
    "個": "个", "這": "这", "為": "为", "麼": "么", "說": "说", "話": "话",
    "東": "东", "學": "学", "習": "习", "記": "记", "裏": "里", "檢": "检",
    "刪": "删", "確": "确", "應": "应", "該": "该", "價": "价", "們": "们",
    "對": "对", "時": "时", "間": "间", "實": "实", "現": "现", "於": "于",
    "與": "与", "從": "从", "後": "后", "讓": "让", "幾": "几", "導": "导",
    "觀": "观", "點": "点", "線": "线", "終": "终", "語": "语", "音": "音",
    "輸": "输", "入": "入", "清": "清", "除": "除", "啟": "启", "動": "动",
    "執": "执", "行": "行", "資": "资", "料": "料", "圖": "图", "書": "书",
    "檔": "档", "案": "案", "電": "电", "腦": "脑", "統": "统", "設": "设",
    "置": "置", "選": "选", "擇": "择", "項": "项", "視": "视", "訊": "讯",
    "軟": "软", "體": "体", "硬": "硬", "務": "务", "器": "器", "編": "编",
    "譯": "译", "碼": "码", "審": "审", "核": "核", "螢": "荧", "幕": "幕",
    "鍵": "键", "盤": "盘", "滑": "滑", "鼠": "鼠", "標": "标", "頻": "频",
    "廣": "广", "播": "播", "樂": "乐", "藝": "艺", "術": "术", "國": "国",
    "際": "际", "區": "区", "區塊": "区块",
})


def _t2s(s):
    """繁体 -> 简体（只覆盖本项目会用到的字；ASR 偶发繁体时会救回规则层）"""
    return s.translate(_T2S) if s else s


''' + anchor
    assert anchor in t, "actions._norm 锚点未匹配"
    t = t.replace(anchor, blk, 1)
    # 在 _norm 内部最前面应用
    i = t.index("def _norm(text):")
    j = t.index("\n", t.index(":", i))   # 函数签名行的结尾
    t = t[:j + 1] + '    text = _t2s(text)          # ⭐ 简繁归一化（ASR 偶发繁体）\n' + t[j + 1:]
    a.write_text(t, encoding="utf-8"); print("  ✅ actions: _t2s 简繁归一化已加（_norm 入口）")
