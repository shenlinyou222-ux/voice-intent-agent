# -*- coding: utf-8 -*-
"""语音听写：把转写文本注入到【当前焦点输入框】。

业界标准两件事：
  ① 判断"光标在不在可编辑控件里"
       · Win32：GetGUIThreadInfo(前台线程) → hwndCaret / hwndFocus，
         再看窗口类名（Edit / RichEdit / Scintilla / 浏览器宿主类）
       · 精确：UI Automation 的 GetFocusedElement + 是否支持 Value/TextPattern
         （浏览器、Electron 这类没有 Win32 caret，只有 UIA 靠得住）
  ② 注入文本：SendInput + KEYEVENTF_UNICODE（逐 UTF-16 码元，中文/emoji 都能送），
     不依赖剪贴板（剪贴板方案会污染用户剪贴板、且部分应用禁用粘贴）

无新依赖：ctypes 手撸；如果装了 uiautomation 就用它做精确判定。
"""
import ctypes
import time
from ctypes import wintypes
from pathlib import Path

user32 = ctypes.WinDLL("user32", use_last_error=True)

# ── 常量 ──
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
ULONG_PTR = ctypes.c_size_t

# 常见"可编辑"窗口类名（小写匹配）
EDIT_CLASSES = ("edit", "richedit", "rich text", "scintilla", "textbox",
                "windows.ui.core.corewindow", "chrome_widgetwin", "mozillawindowclass",
                "sunawtframe", "qt5qwindowicon", "notepad", "ttkcombobox")

# [OPENCODE] 终端窗口：没有 Win32 caret、UIA 也读不到"输入行"（读到的是整个缓冲区），
#   但【键盘输入是有效的】→ 听写要允许往里打，且必须跳过"回读验证"（否则误报没写进去）。
#   · consolewindowclass              = 传统控制台（cmd.exe / conhost / PowerShell 5 窗口）
#   · cascadia_hosting_window_class   = Windows Terminal
#   · windows.ui.input.inputsite.windowclass = Windows Terminal 的输入宿主窗口
#   · mintty / conemu / tabwindowclass = Git Bash / ConEmu / ConEmu 标签
#   · alacritty / wezterm / putty     = 其他常见终端
TERM_CLASSES = ("consolewindowclass", "cascadia_hosting_window_class",
                "windows.ui.input.inputsite.windowclass", "mintty",
                "conemu", "tabwindowclass", "alacritty", "wezterm", "putty")


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("hwndActive", wintypes.HWND),
                ("hwndFocus", wintypes.HWND),
                ("hwndCapture", wintypes.HWND),
                ("hwndMenuOwner", wintypes.HWND),
                ("hwndMoveSize", wintypes.HWND),
                ("hwndCaret", wintypes.HWND),
                ("rcCaret", wintypes.RECT)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _INPUTunion(ctypes.Union):
    # ⚠️⚠️ 血债：联合体必须含 MOUSEINPUT（x64 上 32 字节 = 最大成员）。
    #   我原来只写 c_ubyte*24 → sizeof(INPUT)=28（应为 40）→ SendInput 直接
    #   返回 ERROR_INVALID_PARAMETER(87)，一个字都打不进去（实测日志：
    #   「⚠️ SendInput 失败（已打 0 字，last_error=87）」）。
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT),
                ("padding", ctypes.c_ubyte * 32)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTunion)]


assert ctypes.sizeof(INPUT) == 40, f"INPUT 大小错了: {ctypes.sizeof(INPUT)}（应为 40）"


def _class_of(hwnd) -> str:
    if not hwnd:
        return ""
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def _focus_class_now() -> str:
    """当前【焦点控件】的类名（不依赖 caret_state，避免互相调用）"""
    try:
        hwnd = user32.GetForegroundWindow()
        tid = user32.GetWindowThreadProcessId(hwnd, None)
        gti = GUITHREADINFO()
        gti.cbSize = ctypes.sizeof(GUITHREADINFO)
        if user32.GetGUIThreadInfo(tid, ctypes.byref(gti)) and gti.hwndFocus:
            return _class_of(gti.hwndFocus)
    except Exception:
        pass
    return ""


def is_terminal(hwnd=None):
    """前台窗口是不是【终端】 → (bool, 类名)

    [OPENCODE] 为什么需要：终端里没有 caret、UIA 读不到输入行 —— 原来的判定
    「不在输入框」→ 听写直接不打（用户："听写能不能在终端也写"）。终端的键盘
    输入其实是有效的，所以这里把终端也认成"可输入"。
    """
    cls = _class_of(hwnd) if hwnd else _class_of(user32.GetForegroundWindow())
    fcls = _focus_class_now()
    low = f"{cls} {fcls}".lower()
    hit = any(c in low for c in TERM_CLASSES)
    return hit, (cls or fcls)


def caret_state():
    """→ dict: 前台窗口/焦点控件/类名/是否有 caret/是否像输入框"""
    hwnd = user32.GetForegroundWindow()
    tid = user32.GetWindowThreadProcessId(hwnd, None)
    gti = GUITHREADINFO()
    gti.cbSize = ctypes.sizeof(GUITHREADINFO)
    ok = user32.GetGUIThreadInfo(tid, ctypes.byref(gti))
    focus_cls = _class_of(gti.hwndFocus) or _class_of(hwnd)
    has_caret = bool(gti.hwndCaret)
    looks_edit = any(c in focus_cls.lower() for c in EDIT_CLASSES)
    # [OPENCODE] 终端：也当"可输入"（键盘有效；只是没有 caret / 读不到内容）
    _low = f"{focus_cls} {_class_of(hwnd)}".lower()
    term = any(c in _low for c in TERM_CLASSES)
    return {"fg_hwnd": int(hwnd or 0), "focus_hwnd": int(gti.hwndFocus or 0),
            "caret_hwnd": int(gti.hwndCaret or 0), "focus_class": focus_cls,
            "has_caret": has_caret, "looks_edit": looks_edit,
            "is_terminal": bool(term),
            "call_ok": bool(ok),
            "in_text_field": bool(has_caret or looks_edit or term)}


def _uia_caret():
    """有 uiautomation 就用它精确判断（浏览器/Electron 只有这个靠得住）→ bool|None"""
    try:
        import uiautomation as auto
    except Exception:
        return None
    try:
        c = auto.GetFocusedControl()
        if c is None:
            return False
        name = (c.ControlTypeName or "")
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
            return False        # 文档不是输入框
        # 支持 Value/Text 模式的也算
        for pat in ("ValuePattern", "TextPattern", "LegacyIAccessiblePattern"):
            try:
                if getattr(c, "GetPattern", None) and c.GetPattern(getattr(auto, pat, None)):
                    return True
            except Exception:
                pass
        return False
    except Exception:
        return None


def in_text_field():
    """光标是否在可编辑控件里 → (bool, 说明)"""
    st = caret_state()
    # [OPENCODE] 终端：键盘输入有效，但 UIA 读不到"输入行" → 认成可输入。
    #   ⚠️ 但终端里【auto 模式不自动听写】——否则随便说话都会被打进 shell
    #      （用户实测：「关闭输入后还是路由到听写」）。要往里打就说「开始输入」。
    if st.get("is_terminal"):
        _m = get_mode()
        if _m != "on":
            return False, (f"终端窗口（{st['focus_class'] or '类未知'}）：auto 模式不自动"
                           f"听写（说「开始输入」才开始往里打）")
        return True, (f"终端窗口（{st['focus_class'] or '类未知'}）"
                      f"→ 可键盘输入（读不到输入行，跳过回读验证）")
    u = _uia_caret()
    if u is not None:
        return bool(u), f"UIA={'可编辑' if u else '非输入框'} 类={st['focus_class']}"
    return st["in_text_field"], (f"caret={'有' if st['has_caret'] else '无'} "
                                 f"类={st['focus_class']}")


def _send_vk(vk, up=False):
    ins = (INPUT * 1)()
    ins[0].type = INPUT_KEYBOARD
    ins[0].u.ki = KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP if up else 0, 0, 0)
    return user32.SendInput(1, ctypes.byref(ins), ctypes.sizeof(INPUT))


def clear_field(dry=False):
    """清空当前焦点输入框：Ctrl+A 全选 → Delete。→ (ok, msg)

    为什么用按键而不是 WM_SETTEXT/EM_SETSEL：那两个只对老式 Win32 编辑框有效，
    浏览器/Electron（也就是你实际要用的地方）根本收不到 → 只有按键是通用的。
    ⚠️ 护栏：先确认焦点在可编辑控件里（UIA），否则不清 —— 避免把文档/代码全选删掉。
    ⚠️ 误清了可以 Ctrl+Z 撤销（日志里会提示）。
    """
    ok_field, why = in_text_field()
    # [OPENCODE] 终端：没有"输入框"可清 —— Ctrl+A / Delete 在 shell / REPL / TUI 里
    #   是别的东西（PSReadLine 的移至行首、删字符、甚至程序自己的快捷键），
    #   所以这里明确不做（听写能打进终端，但"清空"不适用）。
    try:
        _term = bool(caret_state().get("is_terminal"))
    except Exception:
        _term = False
    if _term:
        return False, "当前焦点在终端窗口：终端里没有可清空的输入框（不做清空）"
    if not ok_field:
        return False, f"当前焦点不在输入框（{why}）→ 不清空"
    if dry:
        return True, f"[dry] 会清空输入框（Ctrl+A → Delete）（{why}）", None
    VK_CONTROL, VK_A, VK_DELETE = 0x11, 0x41, 0x2E
    n = 0
    n += _send_vk(VK_CONTROL)                 # Ctrl 按下
    n += _send_vk(VK_A)                       # A 按下
    n += _send_vk(VK_A, up=True)              # A 抬起
    n += _send_vk(VK_CONTROL, up=True)        # Ctrl 抬起
    time.sleep(0.05)
    n += _send_vk(VK_DELETE)                  # Delete 按下
    n += _send_vk(VK_DELETE, up=True)         # Delete 抬起
    if n < 6:
        return False, (f"按键注入失败（发了 {n}/6 个事件，"
                       f"last_error={ctypes.get_last_error()}）")
    time.sleep(0.15)
    after = read_focused_value()
    if after is None:
        return True, "已发清空按键（未验证：读不到输入框内容）；误清可按 Ctrl+Z"
    if after.strip() == "":
        return True, "✅ 已确认清空输入框；误清可按 Ctrl+Z 撤销"
    return False, f"❌ 没清掉：输入框还有 {len(after)} 个字符（按键被目标丢弃）"


def read_focused_value():
    """读当前焦点控件的文本 → str；读不到返回 None（不撒谎）。

    为什么要读：SendInput 返回 2 只说明"事件写进了输入队列"，**不代表文本落进
    了输入框**（焦点不在真正可编辑元素上时就会被丢掉）。用户实测指出：
    "如果转写没有写进输入框，日志还是显示成功" —— 那就是假成功。
    """
    try:
        import uiautomation as auto
    except Exception:
        return None
    try:
        c = auto.GetFocusedControl()
        if c is None:
            return None
        # ① ValuePattern（编辑框/浏览器地址栏/搜索框都有）
        try:
            vp = c.GetValuePattern()
            if vp:
                return vp.Value
        except Exception:
            pass
        # ② TextPattern（文档类）
        try:
            tp = c.GetTextPattern()
            if tp:
                return tp.DocumentRange.GetText(-1)
        except Exception:
            pass
        # ③ LegacyIAccessible
        try:
            lp = c.GetLegacyIAccessiblePattern()
            if lp:
                return lp.Value
        except Exception:
            pass
        return None
    except Exception:
        return None


def send_backspaces(n, dry=False):
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


LAST = {}          # 最近一次 type_text 的分段耗时（ms）—— 供调用方做延迟归因
                   # ⚠️ 为什么要它：听写链路的"感知延迟"= ASR + 意图 + 【焦点判定(UIA) +
                   #    读输入框 + 逐字注入 + 验证】。后四项都在这个函数里，而 UIA 读大文档
                   #    可能几百 ms（见 _dictate_async 的注释）。不拆开就只能靠猜。


def type_text(s, dry=False, per_char_delay=0.004, verify=True):
    """把文本注入当前焦点控件 → (ok, msg)。分段耗时写进模块级 LAST。"""
    _tm = {"t0": time.time()}
    # [OPENCODE] 终端窗口：键盘注入有效，但 UIA 读不到"输入行"（TextPattern 给的是
    #   整个滚动缓冲区）→ 回读验证会把"写进去了"误报成"没写进去"，所以跳过验证。
    #   另外：终端里【尾随空格没意义】（标点→空格 / 自动加空格都会产生）→ 去掉。
    _term = False
    try:
        _term = bool(caret_state().get("is_terminal"))
    except Exception:
        _term = False
    if _term:
        _tm["terminal"] = True
        verify = False
        _s2 = (s or "").rstrip()
        if _s2 != (s or ""):
            _tm["rstrip"] = True
            s = _s2

    def _fin(ok, msg):
        _tm["total_ms"] = round((time.time() - _tm["t0"]) * 1000)
        LAST.clear()
        LAST.update(_tm)
        return ok, msg

    if not s:
        return _fin(False, "空文本")
    ok_caret, why = in_text_field()
    _tm["field_ms"] = round((time.time() - _tm["t0"]) * 1000)
    if not ok_caret:
        return _fin(False, f"当前焦点不在输入框（{why}）→ 不打字")
    if dry:
        return _fin(True, f"[dry] 会往输入框打字：{s[:40]!r}（{why}）")
    # ⭐ 注入前先读一次输入框内容（用于事后验证"到底进没进去"）
    before = read_focused_value() if verify else None
    _tm["read_ms"] = round((time.time() - _tm["t0"]) * 1000) - _tm.get("field_ms", 0)
    n = 0
    for ch in s:
        # UTF-16 码元（BMP 外的字符会拆成代理对，SendInput 就是这么吃的）
        units = ch.encode("utf-16-le")
        for i in range(0, len(units), 2):
            code = int.from_bytes(units[i:i + 2], "little")
            ins = (INPUT * 2)()
            for k, flags in ((0, KEYEVENTF_UNICODE),
                             (1, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)):
                ins[k].type = INPUT_KEYBOARD
                ins[k].u.ki = KEYBDINPUT(0, code, flags, 0, 0)
            sent = user32.SendInput(2, ctypes.byref(ins), ctypes.sizeof(INPUT))
            if sent != 2:
                _tm["chars"] = n
                return _fin(False, (f"SendInput 失败（已打 {n} 字，"
                                    f"last_error={ctypes.get_last_error()}）"))
            n += 1
        time.sleep(per_char_delay)
    # ⭐ 到这里为止，字符已经全进输入框了 —— 这一刻才是"用户看见字"的时刻。
    #   下面的验证（轮询读取，确认到"新增"就停）只影响日志"已确认"的时间，
    #   不影响感知延迟；所以两者必须分别报，不能混成一个数。
    _tm["chars"] = n
    _tm["send_ms"] = round((time.time() - _tm["t0"]) * 1000)
    # ⭐⭐ 事后验证（用户实测要求：没打进去就不能报成功）
    #    SendInput 返回 2 只说明"事件进了输入队列"，焦点不在真正可编辑元素上时
    #    会被丢掉 —— 那时旧日志照样写"✅ 已打进输入框"，是假成功。
    if verify:
        # [OPENCODE] 轮询验证（原为固定 sleep 0.15s ≈153ms；字早已进框，等的是"读稳"）
        #   · 每 30ms 读一次，读到"次数增加"立刻停（通常 30–60ms）
        #   · before 读不到（UIA 不支持）→ 只读一次，直接走"未验证"分支，不空等
        #   · 上限 250ms，避免焦点/输入法异常时一直等
        key = s.strip()
        after = None
        if before is None:
            time.sleep(0.03)
            after = read_focused_value()
        else:
            deadline = time.time() + 0.25
            while True:
                time.sleep(0.03)
                after = read_focused_value()
                if after is None:
                    break
                try:
                    if key and after.count(key) > before.count(key):
                        break
                except Exception:
                    break
                if time.time() >= deadline:
                    break
        _tm["verify_ms"] = round((time.time() - _tm["t0"]) * 1000) - _tm["send_ms"]
        if before is None or after is None:
            return _fin(True, (f"已注入 {s[:30]!r}（{n} 码元）⚠️ 未验证："
                               f"读不到输入框内容（UIA 不支持该控件）"))
        # ⚠️⚠️⚠️ 血债（用户第二次抓到假成功）：
        #   不能用 `key in after` —— 焦点可能落在【整个文档】上（实测 12739 字的聊天区），
        #   文档里早就有你之前说过的同一句话 → 这个判断平凡成立 →
        #   日志写「✅ 已确认写进输入框（12739→12739 字）」，实际一个字没进去。
        #   必须要求【出现次数增加】，即"这一次是我们新加进去的"。
        if key and after.count(key) > before.count(key):
            return _fin(True, (f"✅ 已确认写进去（新增 1 处）：{key[:26]!r}"
                               f"（{len(before)}→{len(after)} 字）"))
        return _fin(False, (f"❌ 没写进去：内容里没有【新增】的 {key[:20]!r}"
                            f"（{len(before)}→{len(after)} 字；"
                            f"出现次数 {before.count(key)}→{after.count(key)}；"
                            f"后={after[-24:]!r}）"))
    if _tm.get("terminal"):
        return _fin(True, (f"已注入 {s[:30]!r}（{n} 码元）→ 终端窗口：键盘已发送，"
                           f"不做回读验证（UIA 读不到输入行）"))
    return _fin(True, f"已注入：{s[:40]!r}（{n} 个码元，未验证）")


if __name__ == "__main__":
    st = caret_state()
    print("前台窗口 :", hex(st["fg_hwnd"]))
    print("焦点控件 :", hex(st["focus_hwnd"]), "类名 =", st["focus_class"])
    print("caret    :", hex(st["caret_hwnd"]), "(有" if st["has_caret"] else "(无")
    print("UIA 判定 :", _uia_caret())
    ok, why = in_text_field()
    print("结论     :", "可打字 ✓" if ok else "不在输入框 ✗", "—", why)
    print("（不实际打字；要试就在某个输入框里运行：python typein.py --type 你好）")
    import sys
    if "--type" in sys.argv:
        i = sys.argv.index("--type")
        print(type_text(sys.argv[i + 1] if len(sys.argv) > i + 1 else "测试文本"))


# ══════════════════════════════════════════════════════════════════════
# 听写模式（「开始输入」/「终止输入」语音控制）
#   auto = 老行为：光标在输入框里才听写
#   on   = 强制听写（并可先自动聚焦当前窗口的输入框）
#   off  = 终止听写（非命令的话不再往输入框打）
# ★ [OPENCODE-MP] 2026-10-19：模式必须落盘，重启后保持。
#   血债：应用重启（加载新补丁）后 DICTATION_MODE 回到 auto，
#   用户之前说过「开始输入」→ 在终端里说话却没有任何文字（auto 终端不打），
#   日志里连一句原因都没有，看起来就是"转写坏了"。
# ══════════════════════════════════════════════════════════════════════
_MODE_FILE = Path(__file__).resolve().parent / "dictation_mode.txt"


def _load_mode():
    try:
        m = _MODE_FILE.read_text(encoding="utf-8").strip().lower()
        return m if m in ("auto", "on", "off") else "auto"
    except Exception:
        return "auto"


DICTATION_MODE = _load_mode()


def set_mode(m):
    global DICTATION_MODE
    DICTATION_MODE = str(m or "auto").lower()
    try:
        _MODE_FILE.write_text(DICTATION_MODE, encoding="utf-8")
    except Exception:
        pass
    return DICTATION_MODE


def get_mode():
    return DICTATION_MODE


def find_input_box(max_nodes=12000, max_depth=30, budget_s=3.0):
    """在当前前台窗口里找【最可能是输入框】的控件，并返回它。

    ⚠️ 血债（实测）：
      ① 原来深度上限 12 -> 而 Chromium/Electron 窗口的 UIA 树实测【深 23】,
         半路就放弃 -> 永远"没找到输入框"✗（用户："打开deepseek找不到它的输入框"）
      ② 原来无条件返回第一个 EditControl -> 会先撞上「搜索会话名称」这种搜索框 ✗
    现在的策略：遍历（深度 30 / 节点 12000 / 3 秒预算）后【排序挑】：
      · 只要 EditControl（+ 支持 ValuePattern 的 TextControl）；DocumentControl 排除
        （那是聊天记录/文档，往里面打就是"被吞"）
      · 名字里带 搜索/查找/search 的降权
      · 位置越靠下越优先（消息输入框在窗口底部）
    """
    try:
        import uiautomation as auto
        hwnd = auto.GetForegroundWindow()
        root = auto.ControlFromHandle(hwnd) if hwnd else None
        if root is None:
            return None, "拿不到前台窗口"
        t0 = time.time()
        n = 0
        found = []

        def walk(node, depth=0):
            nonlocal n
            if depth > max_depth or n > max_nodes or (time.time() - t0) > budget_s:
                return
            try:
                kids = node.GetChildren()
            except Exception:
                return
            for c in kids:
                n += 1
                try:
                    ct = c.ControlTypeName
                    if ct in ("EditControl", "ComboBoxControl"):
                        found.append(c)
                    elif ct == "TextControl":
                        try:
                            if c.GetValuePattern():
                                found.append(c)
                        except Exception:
                            pass
                except Exception:
                    pass
                walk(c, depth + 1)

        walk(root)
        if not found:
            return None, f"没找到输入框（遍历 {n} 节点 / 深度上限 {max_depth}）"

        def rank(c):
            nm = (getattr(c, "Name", "") or "").lower()
            bad = 1 if any(x in nm for x in ("搜索", "查找", "search", "find")) else 0
            try:
                y = c.BoundingRectangle.top
            except Exception:
                y = 0
            return (bad, -y)          # 非搜索框优先；越靠下优先

        found.sort(key=rank)
        box = found[0]
        info = "%s「%s」候选%d个" % (box.ControlTypeName,
                                    (getattr(box, "Name", "") or "")[:24], len(found))
        return box, info
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def focus_input():
    """自动聚焦当前窗口的输入框 -> (ok, msg)"""
    # [OPENCODE] 终端没有 UIA 输入框可找（find_input_box 必然找不到）——
    #   但终端本来就"焦点在哪就往哪打"，所以直接算成功（用户手动点一下即可）
    try:
        _term, _cls = is_terminal()
    except Exception:
        _term, _cls = False, ""
    if _term:
        return True, f"终端窗口（{_cls}）→ 直接说就行（在终端里无需聚焦输入框）"
    box, why = find_input_box()
    if box is None:
        return False, f"没找到可聚焦的输入框（{why}）"
    try:
        box.SetFocus()
        time.sleep(0.15)
        return True, f"已聚焦输入框（{box.ControlTypeName} / {(box.Name or '')[:24]}）"
    except Exception as e:
        return False, f"聚焦失败: {type(e).__name__}: {e}"
