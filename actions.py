# -*- coding: utf-8 -*-
"""执行侧 —— 意图命中后真正干活的 Python 代码

设计原则：
  · 所有可执行的东西都从 intent_config.json 读，改配置不用改代码
  · 每个动作都是纯函数，返回 (ok, 人类可读的描述, 附加信息)
  · 支持 dry_run（只打印不执行），方便测试
  · 全部用 stdlib（subprocess / webbrowser / os.startfile），零额外依赖

分句规则（对应配置里的 triggers）：
  打开 <程序名>        → launch_app    （程序名在 apps 闭集里）
  打开 <网址名>        → open_url      （网址名在 urls 闭集里）
  打开 <文件名>        → open_file     （文件名在 files 闭集里）
  打开 <网址>网页      → open_url      （强制按网址闭集匹配）
  打开 <文件>文件      → open_file     （强制按文件闭集匹配）
  打开 <其它>          → open_web      （不在闭集 → 默认浏览器打开/搜索）
  搜索 <词>            → web_search    （默认浏览器搜索）
  搜索文件 <词>        → file_search   （文件管理器搜索）
  精确查找 <词>        → exact_search  （必须含"精确"，用转写文本）
  其它                 → unknown       （不执行）
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import webbrowser
from pathlib import Path

# ⭐ 冻结（PyInstaller）后 __file__ 指向 _internal\，不是 exe 所在目录，
#   所以配置/模型必须按 sys.executable 找。不处理的话会报
#   FileNotFoundError: ...\dist\voice_agent\_internal\intent_config.json
if getattr(sys, "frozen", False):
    HERE = Path(sys.executable).resolve().parent
else:
    HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "intent_config.json"


# ══════════════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════════════
class Config:
    def __init__(self, path=None):
        self.path = Path(path) if path else DEFAULT_CONFIG
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.raw = raw
        self.intents = [i["id"] for i in raw["intents"]]
        self.apps = raw["apps"]
        self.urls = raw["urls"]
        self.files = raw["files"]
        self.search = raw["search"]
        self.triggers = {k: v for k, v in raw["triggers"].items()
                         if not k.startswith("_")}
        self.behavior = raw["behavior"]
        # 别名 → 条目（全部小写，便于匹配）
        self.app_by_alias = {}
        for a in self.apps:
            for al in [a["name"]] + a["aliases"]:
                self.app_by_alias[al.lower()] = a
        self.url_by_alias = {}
        for u in self.urls:
            for al in [u["name"]] + u["aliases"]:
                self.url_by_alias[al.lower()] = u
        self.file_by_alias = {}
        for f in self.files:
            for al in [f["name"]] + f["aliases"]:
                self.file_by_alias[al.lower()] = f

    def candidates(self):
        """所有可作为自由文本纠错候选的名字"""
        out = []
        for a in self.apps:
            out += [a["name"]] + a["aliases"]
        for u in self.urls:
            out += [u["name"]] + u["aliases"]
        for f in self.files:
            out += [f["name"]] + f["aliases"]
        return sorted(set(out), key=len, reverse=True)


# ══════════════════════════════════════════════════════════════
#  分句：把转写文本切成 (意图, 参数)
# ══════════════════════════════════════════════════════════════
# 礼貌前缀：先剥掉再匹配，否则「帮我查一下X」会被「帮我查」抢走参数
POLITE = ("帮我", "给我", "帮忙", "请", "麻烦", "能不能", "可以", "我要", "我想")
# 时间/语气副词：也要剥（「现在打开Ollama」不剥的话参数会变成「现在Ollama」）
LEAD_ADV = ("现在", "马上", "立刻", "赶紧", "快", "先", "再", "然后", "接着", "顺便", "麻烦")
# 句尾语气词 / 动词后缀
TAIL = ("吧", "吗", "呢", "啊", "呀", "哦", "嘛", "一下", "起来", "一下子", "试试")


def _lev(a, b):
    """编辑距离"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _snap_score(arg, cands, max_ratio=0.50):
    """返回 (最佳候选, 距离, 是否全串命中)

    ⭐ 第三个返回值很关键：调用方要靠它跨候选表比优劣。
       踩过的坑：「打开modelcope」时
         apps 的探针 "ode" → "node"       距离 1（探针命中）
         urls 的全串 "modelcope"→"modelscope" 距离 1（全串命中）
       两者距离相同，如果只比距离，先试的 apps 就赢了 → 命中成 Node。
       所以【全串命中必须优先于探针命中】。
    """
    arg = (arg or "").strip()
    if not arg:
        return None, 10 ** 9, False
    al = arg.lower()

    def _best(probes):
        b, bd = None, 10 ** 9
        for c in cands:
            cl = c.lower()
            # ⚠️⚠️ 2 个纯汉字的名字只允许【精确】匹配：汉字差一个字 = 完全不同的词
            #   实测血债：「打开豆瓣」被 _snap 吸成「豆包」（距离 1 落在 2*0.5 阈值内）
            #   -> 去开了豆包。历史上同类的还有「帮我打开微信」-> 微博。
            #   注意比的是【清理后】的参数：'知乎。' 这类带标点的正确参数不能被误伤
            #   （上一版写成 p != cl，把 9/24 用例误伤成拒识）。
            _short_cjk = (len(cl) <= 2 and all("\u4e00" <= ch <= "\u9fff"
                                               for ch in cl))
            for p in probes:
                if _short_cjk and _clean_arg(p).lower() != cl:
                    continue
                d = _lev(p, cl)
                if d < bd:
                    bd, b = d, c
        return b, bd

    ascii_only = "".join(ch for ch in al if ch.isascii())
    variants = [al] + ([ascii_only] if ascii_only and ascii_only != al else [])

    # ① 全串优先
    best, bd = None, 10 ** 9
    for v in variants:
        b, d = _best({v})
        if b is not None and d <= max(1, int(len(b) * max_ratio)) and d < bd:
            best, bd = b, d
    if best is not None:
        return best, bd, True

    # ② 前缀 + 后缀探针兜底
    probes = set()
    for v in variants:
        probes.add(v)
        for i in range(1, len(v)):
            probes.add(v[i:])
            probes.add(v[:i])
    b, d = _best(probes)
    if b is not None and d <= max(1, int(len(b) * max_ratio)):
        return b, d, False
    return None, d, False


def _snap(arg, cands, max_ratio=0.50):
    """在候选里找最接近的（编辑距离），阈值内才返回。

    ⭐ 这一步是必须的：CTC 转写会【吃掉英文单词的首字母】
        Ollama → llama    Steam → team    Python → ython
        PowerShell → owerhell   Edge浏览器 → dge浏览器   Trae → ra
    ⭐⭐ ASCII 清洗：英文名字里混进中文几乎一定是识别错误
        「lede了r」→ 去掉「了」→ "leder" → 距离 blender 只有 2
    """
    return _snap_score(arg, cands, max_ratio)[0]


def _strip_all(text, cfg):
    """把所有触发词 / 后缀 / 礼貌词删掉，只留核心名词。

    用于「模型已判定意图，只需从转写里抠出参数」：
        「把记事本打开」   → 记事本
        「我要用PowerShell」→ powershell
    """
    s = _norm(text)
    junk = (list(cfg.triggers.get("open", []))
            + list(cfg.triggers.get("url_suffix", []))
            + list(cfg.triggers.get("file_suffix", []))
            + ["一下", "起来", "吧", "吗", "呢", "啊", "呀", "哦",
               "这个", "那个", "的", "开", "用", "个"])
    for k in sorted(junk, key=len, reverse=True):
        s = s.replace(k, "")
    return _clean_arg(s)


# 闭集意图 → Config 上的候选表属性名
CLOSED_TABLE = {
    "launch_app": "app_by_alias",
    "open_url": "url_by_alias",
    "open_file": "file_by_alias",
}


def resolve(intent, text, cfg: Config):
    """把（模型判定的意图, 转写文本）解析成可执行的参数。

    ⭐ 这是实际使用的入口（parse 只是纯规则版本，用于自测和交叉校验）。

    闭集意图（launch_app / open_url / open_file）：
        ① 规则 parse 出的 arg 在【该意图的闭集】里精确匹配
        ② 不中 → 在该闭集里 snap（编辑距离 + 后缀匹配）
        ③ 还不中 → 把整句的触发词全删掉再 snap
        ④ 最后整句直接 snap
        ⑤ ⭐ 还不中 → 同音/英文词级纠错（ASR 错例的标准后处理）
    参数化意图（web_search / file_search / exact_search / open_web）：
        用规则切出来的 arg；切不出来就用整句
    """
    ri, arg = parse(text, cfg)

    if intent in CLOSED_TABLE:
        table = getattr(cfg, CLOSED_TABLE[intent])
        if arg and arg.lower() in table:
            return arg
        for cand in (arg, _strip_all(text, cfg), text):
            hit = _snap(cand, table.keys())
            if hit:
                return hit
        # ⭐ 同音纠错 v2（2026-10-17 方案定案接线）：_homophone_hit 的拼音硬规则
        #   由 phon_match 音素级打分替代 —— 修掉两个实测必挂：
        #   「打开记事板」（v1 没剥命令前缀，拼音整串距离 8 永不命中）、
        #   「直呼」（v1 韵母带声调符号，同音被判异音罚 0.4）。
        #   phon_match 不可用时自动回退 v1 老规则，行为不倒退。
        for cand in (arg, _strip_all(text, cfg), text):
            hit = _phon_hit(cand, table.keys())
            if hit:
                return hit
        return arg

    if arg and (ri == intent or ri != "unknown"):
        return arg
    return text.strip()


def _norm(t):
    """剥掉礼貌前缀 + 时间副词 + 句尾语气词。

    ⚠️ 这一步是必须的：不剥的话
         「帮我查一下量化方法」→ 触发词「帮我查」命中 → 参数变成「一下量化方法」
         「帮我精确查找lora」  → 触发词 find 命中 → 参数变成「帮我lora」
         「现在打开Ollama」    → 参数变成「现在Ollama」
    """
    t = t.strip()
    changed = True
    while changed:
        changed = False
        for p in POLITE + LEAD_ADV:
            if t.startswith(p) and len(t) > len(p) + 1:
                t = t[len(p):].strip()
                changed = True
    for s in sorted(TAIL, key=len, reverse=True):
        if t.endswith(s) and len(t) > len(s) + 1:
            t = t[:-len(s)].strip()
            break
    return t


# ASR 转写一定带句末标点（「启动电视。」「打开知乎！」）。参数尾巴挂一个「。」
# 会让闭集精确查找失配、也让快照候选变脏 —— 现场日志就是 '电视。'、'我们的。'、
# '骑士版。' 三个在闭集里找不到。两头都削掉标点/引号（参数两端的标点无意义）。
_ARG_TRIM = "，。！？；：、,.!?;:…~～ \t\r\n\"'“”‘’《》〈〉()（）[]【】"


def _clean_arg(s):
    """参数后处理：去两头标点/空白、去句尾语气词"""
    s = s.strip().strip(_ARG_TRIM).strip()
    for x in ("吧", "吗", "呢", "啊", "呀", "哦", "嘛"):
        if s.endswith(x) and len(s) > len(x) + 1:
            s = s[:-len(x)].strip().strip(_ARG_TRIM).strip()
            break
    return s


def _match_prefix(t, trigs):
    """最长前缀匹配，返回 (触发词, 剩余)。没命中返回 ('', t)"""
    for k in sorted(trigs, key=len, reverse=True):
        if t.startswith(k) and len(t) > len(k):
            return k, t[len(k):]
    return "", t


def _match_any(t, trigs):
    """最长【任意位置】匹配，返回 (触发词, 剩余)。剩余 = 去掉触发词后的全文"""
    for k in sorted(trigs, key=len, reverse=True):
        i = t.find(k)
        if i >= 0:
            return k, (t[:i] + t[i + len(k):]).strip()
    return "", t


def _finish_open(rest, cfg, orig=""):
    """拿到「打开X」里的 X 之后，判定是程序/网址/文件/其它"""
    rest = _clean_arg(rest)
    if not rest:
        # 「打开」「启动」这种裸触发词 —— 不是命令，回退用原文（日志要能看见用户说了啥）
        return "unknown", (orig or rest)
    # 后缀强制按闭集匹配
    # ⭐ 网址后缀：用【包含】而不是 endswith —— 实测踩中（用户真实日志）：
    #   「打开DeepSeek网页版。」以「版」结尾 → endswith 匹配不上 → 掉到应用分支
    #   → 去开了【应用】而不是网页。而且不能只看句尾：
    #   「打开DeepSeek的网页」「打开知乎的网站」同样要能命中。
    #   ⚠️ 护栏：剥掉后缀后【必须能在 urls 闭集里命中】才认（否则
    #   「打开文件管理器」这类会被误吃成 open_url）。
    for suf in cfg.triggers.get("url_suffix", []):
        i = rest.find(suf)
        if i < 0:
            continue
        arg = (rest[:i] + rest[i + len(suf):]).strip().strip("版的了之个这那 ")
        if arg and (arg.lower() in cfg.url_by_alias):
            return "open_url", _clean_arg(arg)
    for suf in cfg.triggers.get("file_suffix", []):
        if rest.endswith(suf) and len(rest) > len(suf):
            return "open_file", _clean_arg(rest[:-len(suf)])
    key = rest.lower()
    if key in cfg.app_by_alias:
        return "launch_app", rest
    if key in cfg.url_by_alias:
        return "open_url", rest
    if key in cfg.file_by_alias:
        return "open_file", rest
    # ⭐ 长得像网址的，直接走 open_web —— 不要让它被同名的收藏夹别名吸走
    #    （「打开github.com」应该开 github.com，不是开收藏夹里的 GitHub）
    #    ⚠️ open_web 默认【关闭】（用户指令：七类命令里屏蔽第七类）：
    #       它是不在闭集时的兜底，会把任意文本送进浏览器/搜索 —— 历史上
    #       噪声被判成 open_web 触发过浏览器搜索，而且麦克风又听回来形成正反馈。
    #       关掉后：不在闭集 → unknown（不执行任何动作）。
    _ow = bool(cfg.behavior.get("enable_open_web", False))
    if _ow and _looks_like_url(rest):
        return "open_web", rest
    # ⭐ 精确匹配失败 → 在【三个闭集里比优劣】
    #    排序键 = (不是全串命中, 距离) —— 全串命中优先，然后比距离
    #    ⚠️ 不能按「程序→网址→文件」固定顺序试：
    #       「打开modelcope」时 apps 的探针 "ode"→"node" 距离 1，
    #       urls 的全串 "modelcope"→"modelscope" 距离也是 1，
    #       只比距离的话先试的 apps 会赢 → 命中成 Node（实测踩过）
    best_kind, best_hit, best_key = None, None, (1, 10 ** 9)
    for kind, table in (("launch_app", cfg.app_by_alias),
                        ("open_url", cfg.url_by_alias),
                        ("open_file", cfg.file_by_alias)):
        hit, d, is_full = _snap_score(rest, table.keys())
        if hit is None:
            continue
        key = (0 if is_full else 1, d)
        if key < best_key:
            best_kind, best_hit, best_key = kind, hit, key
    if best_hit is not None:
        return best_kind, best_hit
    if _ow:
        return "open_web", rest
    return "unknown", orig.strip()


# 简繁归一化（L0）：ASR 偶发输出繁体（实测「打開知乎」「網頁版」），而触发词/
# 名称都是简体 -> 规则层不认 -> 被当 unknown 吞掉。覆盖本项目会用到的字；
# 更彻底的做法是装 opencc。
_T2S = str.maketrans({
    "開": "开", "關": "关", "閉": "闭", "網": "网", "頁": "页", "個": "个",
    "這": "这", "為": "为", "麼": "么", "說": "说", "話": "话", "東": "东",
    "學": "学", "習": "习", "記": "记", "裏": "里", "檢": "检", "刪": "删",
    "確": "确", "應": "应", "該": "该", "價": "价", "們": "们", "對": "对",
    "時": "时", "間": "间", "實": "实", "現": "现", "於": "于", "與": "与",
    "從": "从", "後": "后", "讓": "让", "幾": "几", "導": "导", "觀": "观",
    "點": "点", "線": "线", "終": "终", "語": "语", "輸": "输", "啟": "启",
    "動": "动", "執": "执", "資": "资", "圖": "图", "書": "书", "檔": "档",
    "電": "电", "腦": "脑", "統": "统", "設": "设", "選": "选", "擇": "择",
    "項": "项", "視": "视", "訊": "讯", "軟": "软", "體": "体", "務": "务",
    "編": "编", "譯": "译", "碼": "码", "審": "审", "螢": "荧", "鍵": "键",
    "盤": "盘", "標": "标", "頻": "频", "廣": "广", "樂": "乐", "藝": "艺",
    "術": "术", "國": "国", "際": "际", "區": "区", "轉": "转", "寫": "写",
})


def _t2s(s):
    """繁体 -> 简体（ASR 偶发繁体时救回规则层）"""
    return s.translate(_T2S) if s else s


def parse(text, cfg: Config):
    text = _t2s(text)                 # ⭐ 简繁归一化（ASR 偶发繁体）
    """纯规则解析（不需要模型）—— 用来定"参数"和做交叉校验

    ⚠️ 返回 unknown 时，参数用【原文】而不是归一化后的文本 ——
       日志里要能看到用户到底说了什么（"我想搜索一下" 而不是 "搜索"）。

    ⭐⭐ 两级匹配：先按 _norm 剥过礼貌前缀的文本试，不中就拿【原文】再试一次。
       为什么必须这样：_norm 会把「帮我/给我/我要/我想」剥掉，而触发词表里
       恰恰有「帮我打开/给我开/我要用/帮我搜」这些带前缀的写法 —— 剥完就永远
       匹配不上。实测这就是规则层最大的两个洞：
           web_search → unknown  120 条（「我要搜X」「帮我搜X」「帮我找X」）
           launch_app → unknown   48 条（「给我开X」「我要用X」）
    """
    t = _norm(text)
    if not t:
        return "unknown", text.strip()
    # ⭐ 游戏模式：先让【版本化 profile】的规则直通试一次（坐标/单位/姿态都是确定性的，
    #    比模型可靠）；命中就直接返回，没命中再走桌面规则。
    _game = _game_parse(text, cfg)
    if _game is not None:
        return _game
    got, arg = _parse_core(t, cfg, text)
    if got == "unknown":
        raw = text.strip()
        if raw and raw != t:
            got2, arg2 = _parse_core(raw, cfg, text)
            if got2 != "unknown":
                return got2, arg2
    return got, arg


def _parse_core(t, cfg, orig):
    """实际的规则匹配（t = 归一化后的文本；orig = 原文，用于 unknown 时回显）"""

    # ⭐⭐ 听写模式命令【提前】到最前（无歧义的完整短语）
    #   ⚠️ 血债：原来放在 clear_input 之前（靠后），结果被 open 分支里的
    #   「4d 光杆开动词」 `re.match(r"(开|用|上|去|进)(?=.{2,})", t)` 先接走 ——
    #   「**开**始输入」「**开**始听写」「**开**始打字」全都变成 unknown ✗
    #   （同理「用/上/去/进」开头的听写变体也会中招）。提到最前即可。
    for _k in ("dictation_off", "dictation_on"):
        for _trig in sorted(cfg.triggers.get(_k, []), key=len, reverse=True):
            if _trig in t:
                return _k, ""

    # ① 精确查找（最高优先，必须含"精确"）
    k, rest = _match_prefix(t, cfg.triggers.get("exact_prefix", []))
    if k:
        return "exact_search", _clean_arg(rest)
    if any(x in t for x in cfg.triggers.get("exact_prefix", [])):
        k2, rest2 = _match_any(t, cfg.triggers.get("exact_prefix", []))
        if k2 and rest2:
            return "exact_search", _clean_arg(rest2)

    # ② 搜索文件（必须比 web_search 先判，否则「搜索文件X」会被「搜索」抢走）
    k, rest = _match_prefix(t, cfg.triggers.get("file_search", []))
    if k:
        return "file_search", _clean_arg(rest)
    # ②b 「在文件里找」的两种句式（语料里 60 条 file_search 的真实模式）
    #      · 「在文件里搜X」「在文件里找X」「在文件里查X」
    #      · 「查找X文件 / 查找X文档」（查找 = 找文件，不是网页搜索）
    #    ⚠️⚠️ 绝不能用「句子里出现『文件』」当判据 —— 实测净收益 −20：
    #      会弄坏 20 条标签为 web_search 的句子（「搜索配置文件」「查查日志文件」
    #      = 上网搜配置/日志相关的资料），而一条 file_search 都救不回来。
    #      区分点在【动词】：查找/在文件里搜 → 文件；搜索/搜/查查/找一下 → 网页。
    _m = re.search(r"文件(里|中)(搜|搜索|找|查)", t)
    if _m:
        return "file_search", _clean_arg(t[_m.end():])
    if t.startswith("查找") and (t.endswith("文件") or t.endswith("文档")):
        return "file_search", _clean_arg(t[2:-2])
    # ②c/②d 的前提：**不是打开类命令**、且**确实带搜索意图**
    #   ⚠️ 实测误报 15 条：不加护栏时「打开密码本这个文件」被 ②c 抢走判成
    #      file_search（标签是 open_file —— 打开分支在后面，得让它先走）；
    #      「帮我把这个文件删掉」也被抢（标签 unknown）。
    _open_pref = _match_prefix(t, cfg.triggers.get("open", []))[0]
    _has_search = (any(x in t for x in cfg.triggers.get("web_search", []))
                   or any(x in t for x in cfg.triggers.get("file_search", []))
                   or any(x in t for x in cfg.triggers.get("exact_prefix", [])))
    if not _open_pref and _has_search:
        # ②c 「X这个文件 / X那个文档」= 明确在指某个文件（语料里的 file_search 句式）
        #     ⚠️ 判据必须是【指示代词 + 文件】，不能只看「文件」二字：
        #        「搜索配置文件」标签是 web_search（上网搜配置资料），
        #        「搜索配置文件这个文件」才是 file_search。实测前者 20 条，
        #        用"含文件"当判据会全弄坏（净收益 −20，已实测否定）。
        for _w in ("这个文件", "这个文档", "那个文件", "那个文档"):
            if _w in t:
                _arg = t.replace(_w, "")
                _k, _r = _match_any(_arg, cfg.triggers.get("web_search", []))
                return "file_search", _clean_arg(_r if _r else _arg)
        # ②d 搜索动词紧跟「文件」= 找文件（「找一下文件X」「搜个文件X」）
        #    注意「查查日志文件」「搜索配置文件」不匹配（动词和「文件」之间隔着宾语）
        _m2 = re.search(r"(找|搜|查)(一下|一搜|个|个儿)?文件", t)
        if _m2 and not t.startswith("查找"):
            return "file_search", _clean_arg(t[_m2.end():])

    # ③c DSH 会话：新建 / 续 —— ⭐【触发词 + 后半段原样交给 AI】
    #     用户明确要求："我要让AI自己去执行，我后半段是要作为输入的"
    #     含义：说出 DSH 触发词之后的内容【不是给本地规则执行】的，
    #     而是【原样丢给 DSH 当任务输入】。所以：
    #       · 这一段必须排在 open/web_search 之前（否则「新建会话，搜索X」
    #         会被本地规则抢去开浏览器搜索 —— 正是踩过的坑）
    #       · 参数用 _clean_task（只去句末标点），不剥语气词/引号，尽量原样
    #       · 触发词在句中时，取【后半段】（前缀多是"请/我要/然后"这类填充）
    for _key, _intent in (("dsh_new", "dsh_new"), ("dsh_run", "dsh_run")):
        for _trig in sorted(cfg.triggers.get(_key, []), key=len, reverse=True):
            _i = _loose_find(t, _trig)
            if _i < 0:
                continue
            _tail = _clean_task(t[_i + len(_trig):])
            _head = _clean_task(t[:_i])
            _arg = _tail if len(_tail) >= 2 else _head
            return _intent, _arg

    # ⭐⭐ 站内搜索：「在GitHub里搜X」「GitHub上搜X」「搜GitHub里的X」
    #   实测血债（用户）：说「在GitHub里面搜X」被丢给通用浏览器搜索。
    #   正确：命中 urls 闭集里的站点名 + 有搜索动词 + 有查询词 → site_search
    #   （用该站自己的 search_url，没有就 site: 兜底）。
    _sv = ("搜索", "搜一下", "搜搜", "搜", "查找", "查一下", "查查", "查", "找")
    for _e in cfg.urls:
        _names = [(_e.get("name") or "")] + [str(a) for a in
                                            (_e.get("aliases") or [])]
        for _nm in sorted(_names, key=len, reverse=True):
            if len(_nm) < 2 or _nm.lower() not in t.lower():
                continue
            _i = t.lower().find(_nm.lower())
            _rest = (t[:_i] + t[_i + len(_nm):])
            for _v in _sv:
                _j = _rest.find(_v)
                if _j < 0:
                    continue
                _q = (_rest[:_j] + _rest[_j + len(_v):])
                for _w in ("里面", "里边", "里", "中", "上", "的", "内",
                           "帮我", "给我", "一下", "搜索", "搜"):
                    _q = _q.replace(_w, "")
                _q = _clean_arg(_q)
                if len(_q) >= 1:
                    return "site_search", f"{_e['name']}\t{_q}"

    # ③ 搜索（网页）
    k, rest = _match_prefix(t, cfg.triggers.get("web_search", []))
    if k:
        return "unknown", orig.strip()  # ⛔ web_search 已废弃（用户明令删除）
    # ③b 光杆搜索动词兜底：「搜X」「找X」「查X」
    #    _norm 已经把「我要/帮我/我想」剥掉了，剥完只剩光杆动词，而触发词表里
    #    没有光杆形式 —— 实测这是剩下最大的一类洞（web_search→unknown 60 条：
    #    「我要搜注意力机制」「帮我找知识蒸馏」「我要搜git命令」）。
    #    ⚠️ 必须排在 file_search 规则【之后】：否则「查找文件」「找一下文件X」
    #      会被这里抢成网页搜索。
    _m3 = re.match(r"(搜|搜索|找|查)(?=.{2,})", t)
    if _m3:
        return "unknown", orig.strip()  # ⛔ web_search 已废弃（用户明令删除）
    # ③c 句中的搜索动词（现场实测漏网）：
    #     「然后我们搜索一下今天Dapp排行榜。」→ 前缀匹配失效（句子以"我们"开头），
    #     而且触发词表里只有「搜索」「搜一下」没有「搜索一下」→ 全落空被判 unknown，
    #     最后靠模型判 web_search 但置信度 0.355 < 0.55 → 拒识。
    #     实测用户真实说法基本都是"句中"形式：「我们搜索一下X」「让他搜索X」
    #     「再去规划，搜索一下X」。所以这里按【动词最长优先】在句中找一遍。
    #   ⚠️⚠️ 血债：这里**绝不能放光杆单字**「搜/查/找」—— 实测现场：
    #     「检查一下吧。」/「检查一下。」里的「查」被当成搜索动词，
    #     真的去开了浏览器搜索（ACT web_search('检吧') / ('检一下')）。
    #     单字只允许出现在【句首】（上面的 ③b 用 re.match 锚定，是安全的）。
    _verbs = ("搜索一下", "搜索", "搜一下", "搜搜", "查找", "查一下", "找一下",
              "查询", "检索")
    for _v in _verbs:
        _i = t.find(_v)
        if _i < 0:
            continue
        # ⚠️ 复合词护栏：「检查一下」里的「查一下」不算搜索动词。
        #    实测：「检查一下吧。」被 查一下 命中 → web_search('检吧') 真开浏览器。
        if _i > 0 and t[_i - 1] in "检审核调巡盘抽普排复清自查":
            continue
        _arg = (t[:_i] + t[_i + len(_v):]).strip()
        if len(_clean_arg(_arg)) >= 2:
            return "unknown", orig.strip()  # ⛔ web_search 已废弃（用户明令删除）

    # ④ 打开/启动
    op = cfg.triggers.get("open", [])
    # 4a 前置触发词
    k, rest = _match_prefix(t, op)
    if k:
        return _finish_open(rest, cfg, orig)
    # 4b 后置触发词（「X打开」「X启动一下」）
    for kk in sorted(op, key=len, reverse=True):
        if t.endswith(kk) and len(t) > len(kk) + 1:
            return _finish_open(t[:-len(kk)], cfg, orig)
    # 4c 中间触发词（「把X打开」的「把」已剥；剩下「现在打开X」的副词也剥了）
    k, rest = _match_any(t, op)
    if k and rest:
        return _finish_open(rest, cfg, orig)
    # 4d 光杆开动词兜底：「开X」「用X」「上X」（「开HuggingFace」「上个掘金」）
    #    ⚠️ 只在【参数命中闭集】时才认 —— 否则「开始我们一天的学习」这类会被
    #      判成 open_web 然后去浏览器搜索乱码（安全阀：闭集是确定的）。
    _m4 = re.match(r"(开|用|上|去|进)(?=.{2,})", t)
    if _m4:
        _ki, _ai = _finish_open(_clean_arg(t[_m4.end():]), cfg, orig)
        if _ki != "open_web":
            return _ki, _ai

    # ⭐ 听写模式命令（无参数）：「开始输入」/「终止输入」
    #   ⚠️⚠️ 血债：之前只在配置里加了触发词、也加了 ACTIONS 和规则直通白名单，
    #   但【没接进 parse】-> parse 恒返回 unknown -> 规则直通永不触发 ->
    #   这两条命令 100% 依赖句向量分类器 ✗。后果：分类器一旦判错（或用户说了
    #   没进示例的变体），在强制听写模式下【命令本身会被打进输入框】✗。
    for _k in ("dictation_on", "dictation_off"):
        for _trig in sorted(cfg.triggers.get(_k, []), key=len, reverse=True):
            if _trig in t:
                return _k, ""
    # ⭐ 清空输入框（无参数命令：「清空输入」「清空输入框」）
    for _trig in sorted(cfg.triggers.get("clear_input", []), key=len,
                        reverse=True):
        if _trig in t:
            return "clear_input", ""
    # ⭐ 关闭应用（「关闭Clash」「把Clash关掉」）——只在 apps 闭集里找，绝不乱杀进程
    for _trig in sorted(cfg.triggers.get("close", []), key=len, reverse=True):
        _i = t.find(_trig)
        if _i < 0:
            continue
        _a = _clean_arg((t[:_i] + t[_i + len(_trig):]).strip().strip("版的了之个这那 "))
        if _a and _a.lower() in cfg.app_by_alias:
            return "close_app", _a
    # 后置形式：「Clash关掉」
    for _trig in sorted(cfg.triggers.get("close", []), key=len, reverse=True):
        if t.endswith(_trig) and len(t) > len(_trig):
            _a = _clean_arg(t[:-len(_trig)])
            if _a and _a.lower() in cfg.app_by_alias:
                return "close_app", _a

    # ⭐⭐ 没有动词的「X网页 / X网站 / X官网」：只要 X 命中 urls 闭集就当 open_url
    #   实测血债（日志 16:58:38）：「Deepseek网页。」规则层不认（用户没说"打开"）
    #   → 落到模型 → 模型判 launch_app conf=0.595 → **去开了应用**，而用户要网页。
    for _suf in cfg.triggers.get("url_suffix", []):
        _i = t.find(_suf)
        if _i < 0:
            continue
        _a = (t[:_i] + t[_i + len(_suf):]).strip().strip("版的了之个这那 ")
        if _a and _a.lower() in cfg.url_by_alias:
            return "open_url", _a

    return "unknown", orig.strip()


def hotword_context(cfg, extra=(), max_chars=320, ascii_only=False):
    """拼 ASR 的热词上下文（Qwen3-ASR 的 system 消息）。

    为什么要这个：现场实测「启动DeepSeek」被转成「启动DVC Harness的。」、
    「G.M的。」、「骑士版。」—— 专有名词（尤其中英混说）在没有上下文偏置时
    只能被硬猜成汉字。官方 chat_template.json 里留了 system 消息（ns.system_text）
    这个通道，把闭集名字 + 英文词表喂进去就是标准的 contextual biasing。

    实测（40 条含英文名的语料，exp_asr_hotword.py）：
        无上下文   完全一致 67.5%  相似度 0.900  273ms
        加热词     完全一致 70.0%  相似度 0.928  484ms
    热词有用但把 prompt 拉长 → prefill 变慢。中文名字本来就转得对，
    所以默认 ascii_only=True：只喂【含拉丁字母的名字】——既覆盖被听错的那一类，
    又把 prompt 压到最短。
    """
    # ⭐ 游戏模式（openra_mode=true）：热词完全来自【版本化】的
    #    openra/profiles/<id>/hotwords.txt —— 桌面闭集的名字在红警里毫无意义。
    #    版本戳（hotwords_version）会打进日志，换版本/回滚一目了然。
    _game = _game_hotwords(cfg, ascii_only=ascii_only)
    if _game is not None:
        return _game[:max_chars]

    words = []
    for group in (cfg.apps, cfg.urls, cfg.files):
        for e in group:
            words.append(e["name"])
            words += list(e.get("aliases", []))
    # 顺序很重要（max_chars 会截尾）：
    #   ① 闭集里的英文名（最容易被听错）
    #   ② extra（英文词表 + 【命令词】，如「会话/新建会话/切换会话」——实测它们
    #      决定同音歧义的输赢，绝不能被截掉）
    #   ③ 闭集里的中文名
    latin, cn = [], []
    for w in words:
        w = (w or "").strip()
        if not w:
            continue
        (latin if any(ch.isascii() and ch.isalpha() for ch in w) else cn).append(w)
    out, seen = [], set()
    for w in latin + [str(x).strip() for x in extra if str(x).strip()] + cn:
        hl = w.lower()
        has_latin = any(ch.isascii() and ch.isalpha() for ch in w)
        if hl in seen or (ascii_only and not has_latin):
            continue
        seen.add(hl)
        out.append(w)
    s = "、".join(out)
    return s[:max_chars]


# ══════════════════════════════════════════════════════════════
#  ASR 错例纠错：同音字 / 英文词级（标准后处理）
# ══════════════════════════════════════════════════════════════
# ⚠️ 为什么必须有这一层：ASR 对【专有名词】的错法不是"少一个字母"，
#    而是整段替换成同音/近音的词，编辑距离根本救不回来。现场实证：
#        「打开直呼」            → 知乎      （zhihu == zhihu，同音）
#        「打开记事板」           → 记事本    （jishiban ~ jishiben，差 1）
#        「启动DVC Harness的。」  → DeepSeek  （词元 harness == 别名 harness）
#    这类纠错在中文 ASR 后处理里是标准做法（同音字表 / 拼音匹配），
#    英文名则是词级匹配。
def _pinyin(s):
    """汉字→拼音串（无调、连写、只留字母数字）。没装 pypinyin 就返回 ''"""
    try:
        from pypinyin import lazy_pinyin
    except Exception:
        return ""
    import re as _re
    out = "".join(lazy_pinyin(s or ""))
    return _re.sub(r"[^a-z0-9]", "", out.lower())


def _tokens(s, minlen=4):
    """英文/数字词元（≥minlen 字符才有区分度）"""
    import re as _re
    return {t for t in _re.split(r"[^a-z0-9]+", (s or "").lower())
            if len(t) >= minlen}


def _homophone_hit(s, cands):
    """同音/近音 + 英文词级纠错 → 命中候选（歧义则返回 None）

    保守规则（宁可漏，不可错——开错程序比不开更糟）：
      · 拼音【完全相等】→ 命中（「直呼」=「知乎」）
      · 拼音【差 1】且长度≥3 且首字母相同 → 命中（「记事板」~「记事本」）
      · 英文词元【完全相等】且长度≥4 → 命中（"harness" ∈ 「DVC Harness」）
      · 多个不同候选同时命中 → 不猜，返回 None
    另外：整串匹配不上时，按标点/空白切成片段再试一遍 —— 参数里经常带着
    整句（「直呼，我们看一下吧」），不切就永远匹配不上。
    """
    s = (s or "").strip()
    if not s:
        return None
    import re as _re
    frags = [f for f in _re.split(r"[^\w\u4e00-\u9fff]+", s) if len(f) >= 2]
    variants = [s] + frags
    sps = [(_pinyin(v), _tokens(v)) for v in variants]
    hits = []
    for c in cands:
        cl = (c or "").lower()
        cp = _pinyin(cl)
        ct = _tokens(cl)
        ok = False
        for sp, st in sps:
            if not sp and not st:
                continue
            if sp and cp:
                if sp == cp:
                    ok = True
                elif len(sp) >= 3 and len(cp) >= 3 and sp[0] == cp[0] \
                        and _lev(sp, cp) <= 1:
                    ok = True
            if not ok and st and ct and (st & ct):
                ok = True
            if ok:
                break
        if ok:
            hits.append(c)
    if not hits:
        return None
    uniq = {h.lower() for h in hits}
    if len(uniq) > 1:
        return None                      # 歧义不猜
    return hits[0]


def _phon_hit(s, cands):
    """⑦ 音素级谐音匹配 v2(phon_match.py)—— resolve() 的同音兜底层。

    替代 v1 硬规则(_homophone_hit)做单点判定:打分制 + 阈值 0.60 +
    margin 0.03 拒识。v1 两处实测必挂(前缀不剥 / 声调符号留韵母里)都已修。
    phon_match 不可用(没装 pypinyin)时回退 v1 老规则,行为不倒退。
    ★ 要保证不漏(返回 top-K 列表交给下游),用 resolve_by_rank(),不要用这个。
    """
    if not (s or "").strip():
        return None
    try:
        from phon_match import best_match
        hit, _top, _sec = best_match(s, list(cands))
        return hit
    except Exception:
        return _homophone_hit(s, cands)


# ══════════════════════════════════════════════════════════════
#  「保证不漏」接线路:三表并集 top-K(2026-10-17 方案定案 #3/#4)
# ══════════════════════════════════════════════════════════════
#  与 resolve()::_phon_hit 的分工:
#    · _phon_hit   = 【意图已定、闭集单点兜底】保高精度(阈值+margin 拒识);
#    · resolve_by_rank = 【意图都不确定时的路由层】保不漏 ——
#      规则与模型都没接上时,把 ASR 整句直接对 app∪url∪file 别名打
#      音素级音素-dist,返回 top-K;只有 top1 分够高且唯一才交给调用侧
#      执行,否则返回 None,调用侧维持原行为(拒识/听写/AI),绝不猜。
def closed_alias_union(cfg: Config):
    """方案 #4:调用侧匹配候选 = 三个闭集【别名】表的并集(去重)。

    实测 intent_config.json:app 32 + url 51 + file 0 = 83 条,跨表零重名。
    """
    cached = getattr(cfg, "_closed_alias_union", None)
    if cached is None:
        cached = []
        seen = set()
        for attr in CLOSED_TABLE.values():
            for al in getattr(cfg, attr, {}):
                if al not in seen:
                    seen.add(al)
                    cached.append(al)
        cached.sort()
        try:
            cfg._closed_alias_union = cached
        except Exception:
            pass
    return cached


def alias_owner_map(cfg: Config):
    """别名(小写) → 该别名所属的闭集意图集合。config 跨表重名时含多个意图。"""
    cached = getattr(cfg, "_alias_owner_map", None)
    if cached is None:
        cached = {}
        for intent, attr in CLOSED_TABLE.items():
            for al in getattr(cfg, attr, {}):
                cached.setdefault(al, set()).add(intent)
        try:
            cfg._alias_owner_map = cached
        except Exception:
            pass
    return cached


def closed_alias_stream_variants(alias):
    """别名 → [(统一标注流, 拉丁源掩码)]:纯CJK=1套(掩码全0);
    拉丁 = latin_variants 全套(对应段掩码全1)。

    掩码用于拉丁连读中缝的加重跳过罚分(SylTrie)。
    """
    from phon_match import latin_variants, syllables
    runs = []
    buf, kind = "", None
    for ch in alias:
        if "\u4e00" <= ch <= "\u9fff":
            k = "c"
        elif ch.isascii() and ch.isalpha():
            k = "e"
        else:
            k = None
        if kind is None or k == kind:
            buf += ch
            kind = k
        else:
            if buf.strip() or (kind == "c" and buf):
                runs.append((kind, buf))
            buf, kind = ch, k
    if buf and kind in ("c", "e"):
        runs.append((kind, buf))
    streams, masks = [[]], [[]]
    for k, run in runs:
        if k == "c":
            piece = syllables(run)
            if piece:
                streams = [s + list(piece) for s in streams]
                masks = [m + [0] * len(piece) for m in masks]
        else:
            vs = [v for v in latin_variants(run) if v][:2]
            if not vs:
                continue
            new_s, new_m = [], []
            for s, m in zip(streams, masks):
                for v in vs:
                    new_s.append(s + list(v))
                    new_m.append(m + [1] * len(v))
            streams, masks = new_s, new_m
    return [(s, m) for s, m in zip(streams, masks) if s][:4]


# ══════════════════════════════════════════════════════════════
#  模板前缀生成树（2026-10-18 定案）：动作×别名×词缀 → 统一标注 → trie
#  匹配 = 口令统一标注串 × 模板的子序列对齐(SylTrie.best)
# ══════════════════════════════════════════════════════════════
def closed_family(cfg: Config):
    """别名族自动派生:长别名 = 基础别名+内容词 的字面复合。

    返回 (fam_pairs, fam_sufs):fam_pairs 是 (基础别名, 后缀) 列表,
    fam_sufs 是内容词集合(进词缀内容惩罚名单)。按 Config 实例缓存。
    """
    cached = getattr(cfg, "_closed_family", None)
    if cached is not None:
        return cached
    owners = alias_owner_map(cfg)
    union = closed_alias_union(cfg)
    fam_pairs, fam_sufs = [], []
    for a1 in union:
        o1 = owners.get(a1.lower(), set())
        if len(o1) != 1:
            continue
        for a2 in union:
            if len(a2) <= len(a1) or not a2.startswith(a1):
                continue
            suf = a2[len(a1):]
            if not suf or suf != suf.strip() or " " in suf or len(suf) > 4:
                continue
            if owners.get(a2.lower(), set()) != o1:
                continue
            if (a1, suf) not in fam_pairs:
                fam_pairs.append((a1, suf))
            if suf not in fam_sufs:
                fam_sufs.append(suf)
    try:
        cfg._closed_family = (fam_pairs, fam_sufs)
    except Exception:
        pass
    return fam_pairs, fam_sufs


def closed_templates(cfg: Config):
    """★ 批量生产意图模板:每条模板 = (统一标注流, payload=(intent, arg))。

    · 动作:cfg.triggers['open'](生产触发词,17个 —— 改配置不改码);
      strip 掉含动词槽后缀的多字短语后再拼(「来一下」这种整体动作 OK)。
    · 别名:close_alias_union 三表并集(CJK 一套流;拉丁每套 G2P 变体都挂);
      payload arg = 别名本身(闭集可按别名取条目)。
    · 词缀:url_suffix(网页/网站/页面/官网/网址)+网页版(血债) + file_suffix:
        ① alias+词缀 本身是别名 → payload = 那个条目(DeepSeek+网页→deepseek网页);
        ② url词缀 & intent==open_url → payload=(open_url, alias)(知乎网站→知乎);
        ③ file词缀 & intent==open_file → 同理;
        ④ 其余不出词缀模板(歧义不猜)。
    """
    from phon_match import syllables, latin_variants
    sufs = []
    for s in list((cfg.triggers.get("url_suffix") or [])) + ["网页版"]:
        if s and s not in sufs:
            sufs.append(s)
    fsufs = [s for s in (cfg.triggers.get("file_suffix") or []) if s]
    acts = [a for a in (cfg.triggers.get("open") or []) if a]
    owners = alias_owner_map(cfg)
    union = closed_alias_union(cfg)
    # 别名族后缀对(自动派生,不改码):千问+办公=千问办公、豆包+ai=豆包ai、
    # 知乎+网=知乎网 …… 长别名确实是【基础别名+内容词】字面复合 → 按 rule1
    # (字面别名)出模板,payload=长别名条目本身;基础别名条目只在口令
    # 【真点名了内容词】而没消费时才被罚(词缀内容惩罚)。
    fam_pairs, fam_sufs = closed_family(cfg)
    items = []                                    # (stream, payload, lat_mask)
    for al in union:
        intents = owners.get(al.lower(), set())
        if len(intents) != 1:
            continue
        intent = next(iter(intents))
        asts = closed_alias_stream_variants(al)
        for act in acts:
            asyl = syllables(act)
            for ast, amask in asts:
                base = asyl + ast
                items.append((base, (intent, al, 0),           # S=0:未消费词缀
                              [0] * len(asyl) + list(amask)))
        for suf in sufs + fsufs:
            full = al + suf
            if full in owners and len(owners[full]) == 1:
                pay = (next(iter(owners[full])), full, 1)
            elif suf in sufs and intent == "open_url":
                pay = (intent, al, 1)
            elif suf in fsufs and intent == "open_file":
                pay = (intent, al, 1)
            else:
                continue
            ssyl = syllables(suf)
            for ast, amask in asts:
                for act in acts:
                    asyl = syllables(act)
                    items.append((asyl + ast + ssyl, pay,
                                  [0] * len(asyl) + list(amask)
                                  + [0] * len(ssyl)))
    # 族别名后缀(rule1 形状:字面长别名条目;无 rule2 —— 词条跨条目语义不可猜)
    # 后缀可能是拉丁词(hub/ai)——必须走 G2P 取流,不能用 syllables()(只产 CJK,
    # 会退化成空流 → 族模板塌回基础别名,打开豆包 误中 豆包ai 的教训)。
    for a1, suf in fam_pairs:
        intents = owners.get(a1.lower(), set())
        if len(intents) != 1:
            continue
        intent = next(iter(intents))
        full = a1 + suf
        pay = (intent, full, 1)
        if "\u4e00" <= suf[0] <= "\u9fff":
            spcs = [list(syllables(suf))]            # CJK 后缀:单套读法
        else:
            spcs = [list(v) for v in latin_variants(suf) if v]  # 拉丁后缀:全部变体
        if not spcs or not spcs[0]:
            continue
        a1_asts = closed_alias_stream_variants(a1)
        for act in acts:
            asyl = syllables(act)
            for spc in spcs:
                for ast, amask in a1_asts:
                    items.append((asyl + ast + spc, pay,
                                  [0] * len(asyl) + list(amask)
                                  + [0] * len(spc)))
    return items


def closed_trie(cfg: Config):
    """模板树(按 Config 实例缓存)。"""
    tri = getattr(cfg, "_closed_trie", None)
    if tri is None:
        from phon_match import SylTrie
        tri = SylTrie()
        for stream, payload, lat in closed_templates(cfg):
            tri.add(stream, payload, [bool(b) for b in lat])
        try:
            cfg._closed_trie = tri
        except Exception:
            pass
    return tri


def resolve_by_rank(text, cfg: Config, *, ut=None, affix=None, topk=5,
                    floor=None, adopt=None, margin=None, gap=None):
    """⭐「保证不漏」路由(2026-10-18 统一标注版):口令统一标注串 × 模板树。

    返回 (intent, arg, top_score, second_score);不采纳 = (None, None, top, sec)。
    arg 是【闭集别名】,可直接交给 dispatch()/resolve_closed()。
    ★ ut/affix(2026-10-18 词表置换直喂):ut = 已标注的 [(声母,韵母)] 流
      (ASR 直出拼音词表时,跳过 text_stream 复算);affix = 词缀是否在场的
      布尔判定（流形态无字形可查）。两者缺省时行为与老路径完全一致。

    两路打分按 payload 合并取最优:
      A. 模板子序列匹配 —— 动作+别名+词缀整模板对齐(动作/修饰词天然进裁决:
         「打开DeepSeek网页版」走 deepseek网页,模板更深者胜,不再并列歧义);
      B. 词元精确路(拉丁原文在场且字母数≥4) —— 旧 eng_score 能力保底。
    采纳 = 方案定案三条件【同时】满足:
      ① top1 ≥ adopt         分数够高(0.75,可配 behavior.rank_adopt)
      ② top1 − top2 ≥ margin  唯一(0.06,可配 behavior.rank_margin)
      ③ payload 只属一张闭集表(生成时就已过滤)
    阈值走 cfg.behavior(rank_floor / rank_adopt / rank_margin / rank_gap)。
    """
    top, sec = 0.0, 0.0
    try:
        from phon_match import text_stream, rank_candidates
        f = float(floor if floor is not None
                  else cfg.behavior.get("rank_floor", 0.40))
        a = float(adopt if adopt is not None
                  else cfg.behavior.get("rank_adopt", 0.75))
        m = float(margin if margin is not None
                  else cfg.behavior.get("rank_margin", 0.06))
        g = float(gap if gap is not None
                  else cfg.behavior.get("rank_gap", 0.10))
        sufs_all = [s for s in list(cfg.triggers.get("url_suffix") or [])
                    + ["网页版"] + list(cfg.triggers.get("file_suffix") or [])]
        for s in closed_family(cfg)[1]:
            if s not in sufs_all:
                sufs_all.append(s)
        scores = {}
        if ut is None:
            ut = text_stream(text or "")
        if ut:
            # 只有一条路:口令统一标注串 × 模板树(动作+别名+词缀子序列匹配)。
            # 旧词元别名路已裁撤——它没有结构约束,「网页」俩字部分命中
            # 就能把整条 alias 打满分(油田网页版 撞 deepseek网页 的教训)。
            for sim, payload, depth, cost in closed_trie(cfg).best(
                    ut, gap=g, cap=0.5, min_sim=max(f, 0.30)):
                scores[payload] = max(scores.get(payload, 0.0), sim)
        # 同一条目合并:youtube/油管 是同一别名组的两个写法,不是真歧义。
        #   规范键 = (intent, 条目dict对象id);同分取 S 大的(优先带词缀的记录)。
        by_alias = {"launch_app": cfg.app_by_alias, "open_url": cfg.url_by_alias,
                    "open_file": cfg.file_by_alias}
        merged = {}
        for (it, ar, S), s in scores.items():
            ent = by_alias.get(it, {}).get(ar)
            if ent is None:
                continue
            k = (it, id(ent))
            cur = merged.get(k)
            if cur is None or s > cur[0] or (s == cur[0] and S > cur[1]):
                merged[k] = (s, S, ar)
        # 词缀内容惩罚:口令点名了 网页版/网站/文件… 而该意图没有词缀模板
        # 命中记录(S=0) → 该意图的匹配是"跳过了点名词"得来的,降 0.10。
        # 直喂流形态:affix 由调用方按音节流判定(CJK 字形惩罚原文塌不了)。
        seen = (bool(affix) if affix is not None
                else any(su in (text or "") for su in sufs_all))
        if seen:
            for k in list(merged):
                s, S, ar = merged[k]
                if S == 0:
                    merged[k] = (s - 0.10, S, ar)
        ranked = sorted(((s, k, ar) for k, (s, S, ar) in merged.items() if s >= f),
                        reverse=True)
        if not ranked:
            return None, None, top, sec
        s1, p1, ar1 = ranked[0]
        s2 = ranked[1][0] if len(ranked) > 1 else 0.0
        top, sec = s1, s2
        if s1 < a or (len(ranked) > 1 and s1 - s2 < m):
            return None, None, top, sec          # 分不够 / 歧义,不猜
        return p1[0], ar1, s1, s2
    except Exception:
        return None, None, top, sec


# ══════════════════════════════════════════════════════════════
#  同音容错：触发词也要能认同音错字
# ══════════════════════════════════════════════════════════════
# 现场实测（15:45 日志，用户想触发「新建会话」）：
#     「新建会话」  → ASR 出「新建绘画」     ← huìhuà == huìhuà，完全同音
#     「请切换会话」→ ASR 出「请切换回画」
# 而触发词是【字面】匹配，全落空 → 用户说怎么都不认。
# 所以：触发词匹配加一层同音兜底（中文→拼音比较，ASCII→小写比较）。
_PY_CACHE = {}


def _py_ch(ch):
    v = _PY_CACHE.get(ch)
    if v is None:
        try:
            from pypinyin import lazy_pinyin
            v = "".join(lazy_pinyin(ch)).lower()
        except Exception:
            v = ch.lower()
        _PY_CACHE[ch] = v
    return v


def _py_key(s):
    """中文→拼音，ASCII 字母数字→小写，其它（空格/标点）忽略。"""
    out = []
    for c in (s or ""):
        if "\u4e00" <= c <= "\u9fff":
            out.append(_py_ch(c))
        elif c.isascii() and c.isalnum():
            out.append(c.lower())
    return "".join(out)


def _loose_find(t, trig):
    """在 t 里找 trig：先字面，再同音。→ 起始【字符】下标，找不到 -1。

    ⚠️ 只在字符长度窗口上滑动比较拼音键，所以 ASCII 混合（「让dsh」）也对得上；
       ASCII 大小写由 _py_key 统一小写（ASR 可能出「让DSH」）。
    """
    i = t.find(trig)
    if i >= 0:
        return i
    n = len(trig)
    kt = _py_key(trig)
    if not kt or n <= 0:
        return -1
    for j in range(0, max(0, len(t) - n + 1)):
        if _py_key(t[j:j + n]) == kt:
            return j
    return -1


def _clean_task(s):
    """DSH 任务文本的清理：**只**去两头空白和句末标点。

    ⚠️ 不能用 _clean_arg：它会剥语气词（吧/吗/了…）、削引号 ——
       而这里的内容是【给 AI 的输入】，要尽量原样传（用户明确要求
       "我后半段是要作为输入的"）。只去掉「。」这种纯装饰的句末标点。
    """
    s = (s or "").strip()
    return s.strip("。．.！!？?，,、；;：: \t\r\n").strip()


def resolve_closed(intent, arg, cfg: Config):
    """闭集意图：把参数解析成具体条目"""
    if intent == "launch_app":
        return cfg.app_by_alias.get((arg or "").lower())
    if intent == "open_url":
        return cfg.url_by_alias.get((arg or "").lower())
    if intent == "open_file":
        return cfg.file_by_alias.get((arg or "").lower())
    return None


# ══════════════════════════════════════════════════════════════
#  多意图：把一整句转写切成【分句】，逐句判定
# ══════════════════════════════════════════════════════════════
# ⚠️ 为什么必须切：意图模型是【整句一个 embedding】（E_ALL[整句 tokens] → B → CLS）。
#    一句话里塞两个命令时，两个意图在同一个 CLS 向量里被平均掉，谁也认不出来。
#    实测现场（12:45:49）：「打开知乎，我们看一下吧。」→ 模型整句判 unknown，
#    全靠规则把它救成 open_url；13:03:40 那句
#    「启动我们《DeepSeek Harmony》的讲解版PDF讲解版，开始我们一天的学习。」
#    更是被"太长（43 字 > 30）"整句拒识 —— 里面其实是两个意图。
#    切成「打开知乎」/「我们看一下吧」之后，各自 embedding，第一个就是开知乎。
# ⚠️ 不切空格：ASR 会在英文名中间带空格（「DeepSeek Harmony」「G.M的」），
#    按空格切会把一个名字劈成两半 —— 实测踩中。
CLAUSE_PUNCT = "，,。.；;！!？?、：:\n\r\t"

# 顺承/并列连接词：口语里经常没有标点，但说完一个命令常跟这些词起下一句
CLAUSE_CONNECTORS = (
    "然后", "接着", "之后", "以后", "完了", "顺便", "同时", "另外", "还有",
    "还要", "以及", "并且", "其次", "再者", "再帮", "帮我再",
)

# 触发词（用于"触发词再次出现"这条边界）：长词优先，扫描时不重叠
def _trigger_words(cfg):
    if cfg is None:
        return ["打开", "启动", "运行", "搜索", "查找", "精确"]
    out = []
    for k in ("open", "url_suffix", "file_suffix",
              "web_search", "file_search", "exact_prefix"):
        out += list(cfg.triggers.get(k, []))
    return sorted(set(w for w in out if w), key=len, reverse=True)


def _split_by_triggers(piece, cfg):
    """「打开知乎 打开B站」这种没标点没连词的，靠触发词再次出现切开。

    只在"后面那个触发词前面已经有内容"且"前面还有过触发词"时才切 ——
    否则「搜索文件X」里的「搜索」「文件」会被误当两个边界。
    """
    words = _trigger_words(cfg)
    hits = []                       # (pos, word)，长词优先、不重叠
    taken = [False] * len(piece)
    for w in words:
        start = 0
        while True:
            i = piece.find(w, start)
            if i < 0:
                break
            if not any(taken[i:i + len(w)]):
                hits.append((i, w))
                for j in range(i, i + len(w)):
                    taken[j] = True
            start = i + 1
    hits.sort()
    if len(hits) < 2:
        return [piece]
    cuts = []
    for prev, cur in zip(hits, hits[1:]):
        # 上一个触发词之后还有内容（不是紧挨着），才认为这里起新的一句
        if cur[0] > prev[0] + len(prev[1]):
            cuts.append(cur[0])
    if not cuts:
        return [piece]
    parts, last = [], 0
    for c in cuts:
        parts.append(piece[last:c])
        last = c
    parts.append(piece[last:])
    return parts


def split_clauses(text, cfg=None, max_clauses=6):
    """把转写切成若干分句（多意图用）。返回 list[str]，已去空白/标点。

    切分依据（按顺序叠加）：
      ① 标点：，。；！？ 等（ASR 会带标点，最可靠）
      ② 顺承词：然后/接着/顺便/还有…（口语常无标点）
      ③ 触发词再次出现：两个命令挨着说时最后的边界依据
    """
    import re as _re
    t = (text or "").strip()
    if not t:
        return []
    # ① 标点
    pieces = [p for p in _re.split("[" + _re.escape(CLAUSE_PUNCT) + "]+", t) if p.strip()]
    # ② 顺承词（切在词之前，连接词归入后一句）
    step = []
    for p in pieces:
        cur, cuts = p, []
        for c in CLAUSE_CONNECTORS:
            i = p.find(c)
            while i > 0:                       # i==0 说明本来就以它开头，不用切
                cuts.append(i)
                i = p.find(c, i + 1)
        if not cuts:
            step.append(cur)
            continue
        cuts = sorted(set(cuts))
        last = 0
        for c in cuts:
            step.append(p[last:c])
            last = c
        step.append(p[last:])
    # ③ 触发词再次出现
    step2 = []
    for p in step:
        step2 += _split_by_triggers(p, cfg)
    # 清洗 + 合并超限的尾巴（宁可合并也不要丢掉可能的命令）
    out = []
    for p in step2:
        s = p.strip().strip(CLAUSE_PUNCT).strip()
        if len(s) >= 2:
            out.append(s)
    if len(out) > max_clauses:
        out = out[:max_clauses - 1] + ["".join(out[max_clauses - 1:])]
    return out


# ══════════════════════════════════════════════════════════════
#  动作
# ══════════════════════════════════════════════════════════════
def _looks_like_url(s):
    s = s.strip()
    if not s or " " in s:
        return False
    if s.startswith(("http://", "https://")):
        return True
    return bool(re.match(r"^[\w\-]+(\.[\w\-]+)+(/\S*)?$", s))


def act_launch_app(arg, cfg, dry=False):
    e = resolve_closed("launch_app", arg, cfg)
    if not e:
        return False, f"程序闭集里没有「{arg}」", None
    p = e["path"]
    if not Path(p).exists():
        return False, f"程序路径不存在: {p}", None
    # ⭐ 可选 args：启动 agent / 带参数的会话需要（如 dsh web、dsh headless "任务"）
    #    {"name":"DSH会话","path":"...\\dsh.cmd","args":["web","--no-open"]}
    # ⭐ AppsFolder 激活：Start 菜单注册的应用（WorkBuddy/千问/千问办公…）直接跑
    #   exe 会【立刻退出且无窗口】（实测 WorkBuddyAI 退出码 0）—— 必须由 shell 激活，
    #   这就是用户点开始菜单时做的事（等价：explorer.exe shell:AppsFolder\<AUMID>）。
    if e.get("aumid"):
        cmd = ["explorer.exe", "shell:AppsFolder\\" + str(e["aumid"])]
        if dry:
            return True, f"[dry] 启动 {e['name']}（AppsFolder {e['aumid']}）", {"aumid": e["aumid"]}
        subprocess.Popen(cmd, close_fds=True)
        return True, f"已启动 {e['name']}（AppsFolder {e['aumid']}）", {"aumid": e["aumid"]}
    args = [str(a) for a in (e.get("args") or [])]
    # ⚠️ .cmd/.bat 不能直接 CreateProcess（Windows 报"不是有效的 Win32 应用"）。
    # ⚠️⚠️ 而且必须用 `cmd /c call "<路径>" args…`：路径含空格时
    #     ["cmd.exe","/c",路径,…] 会因命令行里有多个引号被 cmd 剥掉外层引号、
    #     在空格处劈开路径（实测：'C:\…\Programs\DeepSeek' is not recognized）。
    #     四种传法实测只有 call 内建稳（call 自己处理引号，不走 /c 的剥引号规则）。
    if Path(p).suffix.lower() in (".cmd", ".bat"):
        cmd = ["cmd.exe", "/c", "call", p] + args
    else:
        cmd = [p] + args
    if dry:
        return True, ("[dry] 启动 " + e["name"] + " → " + " ".join(cmd)), {
            "path": p, "args": args}
    subprocess.Popen(cmd, close_fds=True,
                     cwd=str(Path(p).parent) if Path(p).parent.exists() else None)
    return True, (f"已启动 {e['name']} ({p}"
                  + (" " + " ".join(args) if args else "") + ")"), {"path": p,
                                                                    "args": args}


def act_open_url(arg, cfg, dry=False):
    """打开网址。**不做任何代理前置检查** —— 系统代理由用户自己在 Clash 里统一管理
    （用户明确要求：外网条目不需要任何前置检查、也不走乱七八糟的代理）。"""
    e = resolve_closed("open_url", arg, cfg)
    if not e:
        return False, f"网址闭集里没有「{arg}」", None
    if dry:
        return True, f"[dry] 打开 {e['name']} → {e['url']}", {"url": e["url"]}
    webbrowser.open(e["url"])
    return True, f"已打开 {e['name']} ({e['url']})", {"url": e["url"]}


def act_close_app(arg, cfg, dry=False):
    """关闭应用（语音"关闭X"）。只在【apps 闭集】里找，绝不按任意名字杀进程。"""
    e = resolve_closed("launch_app", arg, cfg)
    if not e:
        return False, f"程序闭集里没有「{arg}」，不能关", None
    name = Path(e["path"]).name
    if dry:
        return True, f"[dry] 关闭 {e['name']}（结束进程 {name}）", {"image": name}
    try:
        # ⚠️ 实测血债：taskkill /IM（优雅 与 /F）对 Clash Verge 都失败 ——
        #   "The process ... could not be terminated"（它有 Windows 服务 + 核心进程，
        #    且可能以更高权限运行）。所以：多种标准手段依次试，并且【事后真的复查
        #    进程还在不在】—— 不能像以前那样只看返回码就报成功。
        def _running():
            try:
                rr = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {name}"],
                                    capture_output=True, text=True, timeout=15)
                return name.lower() in (rr.stdout or "").lower()
            except Exception:
                return True

        stem = Path(name).stem
        methods = [
            ("taskkill /IM /T /F", ["taskkill", "/IM", name, "/T", "/F"]),
            ("Stop-Process -Force", ["powershell", "-NoProfile", "-Command",
                                     f"Stop-Process -Name '{stem}' -Force "
                                     f"-ErrorAction SilentlyContinue"]),
            ("CIM Terminate", ["powershell", "-NoProfile", "-Command",
                               "Get-CimInstance Win32_Process -Filter "
                               f"\"Name='{name}'\" | ForEach-Object "
                               "{ $_.Terminate() }"]),
        ]
        tried = []
        for label, cmd in methods:
            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            except Exception as ex:
                tried.append(f"{label}({type(ex).__name__})")
                continue
            time.sleep(0.8)
            if not _running():
                return (True, f"已关闭 {e['name']}（{name}，用 {label}）",
                        {"image": name, "method": label})
            tried.append(label)
        return (False,
                f"关不掉 {e['name']}（{name} 仍在运行；试过 {'/'.join(tried)}）"
                f" —— 通常是它以管理员权限运行、普通权限杀不掉："
                f"以管理员身份重启本程序，或从托盘右键退出", None)
    except Exception as ex:
        return False, f"关闭失败: {type(ex).__name__}: {ex}", None


def act_open_file(arg, cfg, dry=False):
    e = resolve_closed("open_file", arg, cfg)
    if not e:
        return False, f"文件闭集里没有「{arg}」", None
    p = e["path"]
    if not Path(p).exists():
        return False, f"文件不存在: {p}", None
    if dry:
        return True, f"[dry] 打开文件 {e['name']} → {p}", {"path": p}
    os.startfile(p)
    return True, f"已打开文件 {e['name']} ({p})", {"path": p}


def act_open_web(arg, cfg, dry=False):
    """不在闭集 → 默认浏览器打开（像网址就直开，否则搜索）

    ⚠️ 默认屏蔽（enable_open_web=false，用户指令：七类命令里屏蔽第七类）。
       它会把任意文本送进浏览器/搜索 —— 历史上噪声被判成 open_web 触发过浏览器
       搜索，麦克风又听回来形成正反馈。要恢复就把 intent_config.json 的
       enable_open_web 改成 true（同时 _finish_open 那边也会跟着恢复）。
    """
    if not cfg.behavior.get("enable_open_web", False):
        return False, "open_web 已屏蔽（不在闭集 → 不执行）", None
    arg = (arg or "").strip()
    if not arg:
        return False, "参数为空", None
    if _looks_like_url(arg):
        url = arg if arg.startswith("http") else "https://" + arg
        if dry:
            return True, f"[dry] 浏览器打开 {url}", {"url": url}
        webbrowser.open(url)
        return True, f"已用默认浏览器打开 {url}", {"url": url}
    return act_web_search(arg, cfg, dry)


def act_web_search(arg, cfg, dry=False):
    """⛔ 已废弃：web_search（"搜索X"→浏览器搜索）按用户明令删除。
    保留函数只为防御：任何残留路径产出 web_search 时都不许开浏览器。"""
    return False, "web_search 意图已删除（这类意图过时）；要站内搜索请说『在X里搜Y』", None


def _act_web_search_disabled(arg, cfg, dry=False):
    q = (arg or "").strip()
    if not q:
        return False, "搜索词为空", None
    url = cfg.search["web_engine"].replace("{q}", urllib.parse.quote(q))
    if dry:
        return True, f"[dry] 浏览器搜索「{q}」→ {url}", {"url": url}
    webbrowser.open(url)
    return True, f"已用默认浏览器搜索「{q}」", {"url": url}


def act_dsh_run(arg, cfg, dry=False):
    """把说的话当作任务交给 DSH —— 默认【在同一个会话里】继续。

    用户约定："除非明确说新建会话，不然就一直在这个会话"。
    没有当前会话时顺手建一个（并记住 id）。
    """
    from dsh_session import run_task, load_current
    cur = load_current(cfg)
    ok, msg, info = run_task(cfg, arg, force_new=False, dry=dry)
    if ok and cur:
        return True, "（续会话 " + cur[:16] + "…）" + msg, info
    return ok, msg, info


def act_dsh_new(arg, cfg, dry=False):
    """明确说「新建会话」时走这条：强制建新会话，并记住新 id。"""
    from dsh_session import run_task
    return run_task(cfg, arg, force_new=True, dry=dry)


def act_file_search(arg, cfg, dry=False):
    """文件管理器搜索：用 Windows Search 协议让资源管理器打开搜索结果"""
    q = (arg or "").strip()
    if not q:
        return False, "搜索词为空", None
    root = cfg.search.get("file_manager_root", "C:\\Users")
    if sys.platform != "win32":
        return False, "文件管理器搜索仅 Windows 支持", None
    # search-ms:query=<词>&crumb=location:<路径>
    uri = ("search-ms:query=" + urllib.parse.quote(q) +
           "&crumb=location:" + urllib.parse.quote(root) + "&")
    if dry:
        return True, f"[dry] 文件管理器搜索「{q}」于 {root}", {"uri": uri}
    subprocess.Popen(["explorer.exe", uri], close_fds=True)
    return True, f"已在文件管理器里搜索「{q}」（范围 {root}）", {"uri": uri}


def act_exact_search(arg, cfg, dry=False):
    """精确查找：明确使用转写文本，不纠错、不联想"""
    q = (arg or "").strip()
    if not q:
        return False, "精确查找词为空", None
    url = cfg.search["web_engine"].replace("{q}", urllib.parse.quote(q))
    if dry:
        return True, f"[dry] 精确查找「{q}」→ {url}", {"url": url, "exact": True}
    webbrowser.open(url)
    return True, f"已精确查找「{q}」（原文，未纠错）", {"url": url, "exact": True}


def act_unknown(arg, cfg, dry=False):
    return False, f"不是命令，不执行（识别为: {arg!r}）", None


def act_site_search(arg, cfg, dry=False):
    """站内搜索。arg 约定：'站点名\\t查询词'。

    为什么要有：用户说「在GitHub里搜transformer」，我原来只当通用网页搜索
    （丢给 Bing）—— 正确做法是用**该站自己的搜索 URL**（没有就用 site: 兜底，
    这是业界通用做法）。
    """
    if "\t" not in (arg or ""):
        return False, f"站内搜索参数格式不对: {arg!r}", None
    site, _, q = arg.partition("\t")
    q = (q or "").strip()
    if not q:
        return False, f"没说搜什么（站点 {site}）", None
    key = site.strip().lower()
    ent = None
    for e in cfg.urls:
        names = [e.get("name", "")] + list(e.get("aliases") or [])
        if any(key == str(n).strip().lower() for n in names):
            ent = e
            break
    if ent is None:
        return False, f"网址闭集里没有「{site}」", None
    tpl = ent.get("search_url")
    if tpl:
        url = tpl.replace("{q}", urllib.parse.quote(q))
    else:
        # 兜底：site: 限定（所有搜索引擎都支持）
        eng = cfg.search.get("web_engine", "https://www.bing.com/search?q={q}")
        url = eng.replace("{q}", urllib.parse.quote(f"site:{ent['url'].split('//')[-1].strip('/')} {q}"))
    if dry:
        return True, f"[dry] 在 {ent['name']} 里搜「{q}」→ {url}", {"url": url}
    webbrowser.open(url)
    return True, f"已在 {ent['name']} 里搜索「{q}」→ {url}", {"url": url}


def act_clear_input(arg, cfg, dry=False):
    """清空当前焦点输入框（语音"清空输入"）。Ctrl+A → Delete，跨应用通用。"""
    if not cfg.behavior.get("enable_clear_input", True):
        return False, "清空输入已关闭（enable_clear_input=false）", None
    import typein
    # ⚠️ 血债：clear_field 的返回元数不一致（有的分支 2 元、有的 3 元）→
    #   实测 `ValueError: not enough values to unpack (expected 3, got 2)`
    #   （日志 09:21:44 说「清空输入」就报这个）。按最少元数安全取值。
    r = typein.clear_field(dry=dry)
    return r[0], r[1], None


def act_dictation_on(arg, cfg, dry=False):
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


ACTIONS = {
    "site_search": act_site_search,
    "dictation_on": act_dictation_on,
    "dictation_off": act_dictation_off,
    "clear_input": act_clear_input,
    "launch_app": act_launch_app,
    "open_url": act_open_url,
    "open_file": act_open_file,
    "open_web": act_open_web,
    "web_search": act_web_search,
    "file_search": act_file_search,
    "exact_search": act_exact_search,
    # ⭐ DSH 会话：dsh_run = 默认（续同一个会话，没会话就新建）；
    #             dsh_new = 明确说「新建会话」时才建新的
    "close_app": act_close_app,
    "dsh_run": act_dsh_run,
    "dsh_new": act_dsh_new,
    "unknown": act_unknown,
}


def dispatch(intent, arg, cfg: Config, dry=None):
    """执行一个意图。返回 (ok, 描述, info)"""
    if dry is None:
        dry = cfg.behavior.get("dry_run", False)
    fn = ACTIONS.get(intent)
    if fn is None:
        return False, f"未知意图 {intent}", None
    t0 = time.time()
    try:
        ok, msg, info = fn(arg, cfg, dry)
    except Exception as e:
        ok, msg, info = False, f"执行异常: {type(e).__name__}: {e}", None
    if cfg.behavior.get("log_actions"):
        dt = (time.time() - t0) * 1000
        try:
            from dlog import log as _log
            _log(f"{intent}({arg!r}) -> {msg}  {dt:.0f}ms", "ACT", sub=True)
        except Exception:
            print(f"[action] {intent}({arg!r}) -> {msg}  ({dt:.0f}ms)", flush=True)
    return ok, msg, info


# ══════════════════════════════════════════════════════════════
#  命令清单
# ══════════════════════════════════════════════════════════════
def list_commands(cfg: Config):
    lines = []
    lines.append("=" * 78)
    lines.append(f"  可执行命令清单  ——  配置: {cfg.path}")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"【1】启动程序（{len(cfg.apps)} 个闭集）→ 执行 exe")
    lines.append("-" * 78)
    for i, a in enumerate(cfg.apps, 1):
        ok = "✅" if Path(a["path"]).exists() else "❌"
        lines.append(f"  {i:2d}. {ok} 说「启动{a['name']}」/「打开{a['name']}」  →  {a['path']}")
    lines.append("")
    lines.append(f"【2】打开网址（{len(cfg.urls)} 个闭集）→ 默认浏览器")
    lines.append("-" * 78)
    for i, u in enumerate(cfg.urls, 1):
        lines.append(f"  {i:2d}. 说「打开{u['name']}」/「打开{u['name']}网页」  →  {u['url']}")
    lines.append("")
    lines.append(f"【3】打开文件（{len(cfg.files)} 个闭集）→ 默认程序")
    lines.append("-" * 78)
    for i, f in enumerate(cfg.files, 1):
        ok = "✅" if Path(f["path"]).exists() else "❌"
        lines.append(f"  {i:2d}. {ok} 说「打开{f['name']}文件」/「打开{f['name']}」  →  {f['path']}")
    lines.append("")
    lines.append("【4】参数化命令（用转写文本，不查闭集）")
    lines.append("-" * 78)
    lines.append(f"    说「打开 xxx」        （xxx 不在上面任何闭集里）→ 默认浏览器打开/搜索")
    lines.append(f"    说「搜索 xxx」        → 默认浏览器搜索（{cfg.search.get('web_engine_name','')}）")
    lines.append(f"    说「搜索文件 xxx」    → 文件管理器搜索（范围 {cfg.search.get('file_manager_root')}）")
    lines.append(f"    说「精确查找 xxx」    → 【必须含「精确」】用转写原文搜索，不纠错")
    lines.append("")
    lines.append("【5】不执行")
    lines.append("-" * 78)
    lines.append("    其它任何话 → 识别为 unknown，什么都不做")
    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


if __name__ == "__main__":
    cfg = Config(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".json") else None)
    if "--list" in sys.argv or len(sys.argv) == 1:
        print(list_commands(cfg))
    if "--selftest" in sys.argv:
        print("\n" + "=" * 78)
        print("  分句自测（纯规则，不需要模型）")
        print("=" * 78)
        cases = [
            "启动记事本", "打开B站", "打开B站网页", "打开季度报告文件",
            "打开github.com", "打开这个不存在的玩意儿", "搜索python教程",
            "搜索文件季度报告", "精确查找resnet50", "今天天气怎么样",
            "帮我打开任务管理器", "运行Blender",
        ]
        for c in cases:
            it, arg = parse(c, cfg)
            ok, msg, _ = dispatch(it, arg, cfg, dry=True)
            print(f"  {c:24s} → {it:14s} arg={arg!r:18s} {msg}")


# ══════════════════════════════════════════════════════════════════════
# L3 实体链接兜底（包一层 resolve_closed）
#   为什么放在这里：resolve_closed 是"参数 -> 闭集实体"的最后一关，
#   它返回 None 就意味着要拒识/乱开。实测血债：「直呼」「记事板」
#   「第四个」「知乎网」「Deep Seek」这类 ASR 近音/近形/经验误听，
#   靠原来的编辑距离会**乱吸**（豆瓣->豆包）或**吸不到**（知乎网）。
#   现在：原来命中就用原来的 ✓；没命中才走四路融合的实体链接，
#   分数 >= entity_link_thr（默认 0.75）才接受，否则仍然拒识。
_RESOLVE_CLOSED_ORIG = resolve_closed


def resolve_closed(intent, arg, cfg, *a, **kw):
    # ⭐ 游戏意图没有"闭集"这一说：参数由 profile 的规则直通切好（坐标/单位/姿态），
    #    模型判出来的 game_* 也直接放行，交给 game_actions 里的动作去执行。
    if str(intent).startswith("game_"):
        return {"id": intent, "arg": arg, "game": True}
    r = _RESOLVE_CLOSED_ORIG(intent, arg, cfg, *a, **kw)
    if r:
        return r
    try:
        from entity_link import link
        table = {"launch_app": cfg.apps, "open_url": cfg.urls,
                 "open_file": cfg.files}.get(intent)
        if not table:
            return None
        pairs, back = [], {}
        for e in table:
            for nm in [e.get("name", "")] + [str(x) for x in (e.get("aliases") or [])]:
                if nm:
                    pairs.append((nm, e.get("name")))
                    back[nm] = e
        name, score, how = link(
            str(arg or ""), pairs,
            thr=float(cfg.behavior.get("entity_link_thr") or 0.75))
        if name and name in back:
            try:
                from dlog import log
                log(f"   🔗 实体链接：{arg!r} -> {name}（{how} {score:.2f}）", "INT")
            except Exception:
                pass
            return back[name]
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════
#  OpenRA 语音控制（版本化 profile）—— 详见 openra/DESIGN.md
# ══════════════════════════════════════════════════════════════
#  设计要点（对应用户要求）：
#    · 热词 / 意图语料 / embedding 索引 三样各自版本化，目录即版本
#      （openra/profiles/<id>/），换域 = 换目录，旧版本留着可 A/B、可回滚
#    · 游戏意图【独立部署】：intent_config.openra.json 只含游戏意图，
#      桌面那套（打开程序/网址/搜索）在游戏部署里完全不加载
#    · 只变两处：ASR 热词（hotwords.txt）+ embedding 索引（index.json），
#      模型权重不动；控制指令走本地 TCP 进游戏
_OPENRA = None
_OPENRA_ERR = None
try:
    from openra import game_actions as _game_actions
    from openra import game_profile as _game_profile
    ACTIONS.update(_game_actions.GAME_ACTIONS)
    _OPENRA = _game_actions
except Exception as _e:            # 没有 openra 目录时不影响原有功能
    _OPENRA_ERR = f"{type(_e).__name__}: {_e}"

# ══════════════════════════════════════════════════════════════
#  AoE4（帝国时代4）语音控制 —— 文件桥：写 Scratch\vb_cmd.txt，
#  游戏内 SCAR 脚本每秒轮询执行（见 mod 的 voicecontroller.scar）。
#  刻意只做 3 个意图（生成农民/训练农民/探活），规则直通、不依赖 embedding。
_AOE4 = None
_AOE4_ERR = None
try:
    from aoe4 import game_actions as _aoe4_actions
    from aoe4 import game_profile as _aoe4_profile
    ACTIONS.update(_aoe4_actions.GAME_ACTIONS)
    _AOE4 = _aoe4_actions
except Exception as _e:
    _AOE4_ERR = f"{type(_e).__name__}: {_e}"

# ══════════════════════════════════════════════════════════════
#  运维语音域（ops）—— 第三个域，与 aoe4/openra 同构。
#  三档路由：规则直通(0ms) → embedding kNN(~30ms) → AI 升级(DSH 会话)。
#  安全分层：只读直通 / 变更【两步确认】/ 危险动词不进闭集。
#  输出：短输出 rich 渲染；长输出拉 Windows Terminal + pygmentize 高亮。
_OPS = None
_OPS_ERR = None
try:
    from ops import game_actions as _ops_actions
    from ops import game_profile as _ops_profile
    ACTIONS.update(_ops_actions.GAME_ACTIONS)
    _OPS = _ops_actions
except Exception as _e:
    _OPS_ERR = f"{type(_e).__name__}: {_e}"


def aoe4_mode(cfg) -> bool:
    try:
        return bool(cfg.behavior.get("aoe4_mode"))
    except Exception:
        return False


def ops_mode(cfg) -> bool:
    """运维域是否激活（与 aoe4_mode / openra_mode 互斥）。

    激活后：规则直通走 ops/profiles/<id>/profile.json 的闭集+短语表，
    ASR 热词走该 profile 的 hotwords.txt，动作走 ops 的 GAME_ACTIONS。
    """
    try:
        return bool(cfg.behavior.get("ops_mode"))
    except Exception:
        return False


def game_mode(cfg) -> bool:
    """是否处于游戏模式（热词与意图都走版本化 profile）。

    ⚠️ 血债：新增一个域时【必须】在这里和 _game_profile_obj 里都加上，
       否则该域的 GAME_ACTIONS 虽然注册进了 ACTIONS（动作能查到），
       但 parse() 永远走不到它 —— 表现为「单元测试全过、集成全 unknown」。
       ops 域就踩过这个坑（只加了 ACTIONS.update，忘了加路由）。
    """
    try:
        return (bool(cfg.behavior.get("openra_mode"))
                or aoe4_mode(cfg)
                or ops_mode(cfg))
    except Exception:
        return False


def _game_profile_obj(cfg):
    if aoe4_mode(cfg):
        if _AOE4 is None:
            return None
        try:
            return _aoe4_profile.load(cfg.behavior.get("aoe4_profile"))
        except Exception:
            return None
    if ops_mode(cfg):
        if _OPS is None:
            return None
        try:
            return _ops_profile.load(cfg.behavior.get("ops_profile"))
        except Exception:
            return None
    if _OPENRA is None:
        return None
    try:
        return _game_profile.load(cfg.behavior.get("openra_profile"))
    except Exception:
        return None


def _game_hotwords(cfg, ascii_only=False):
    """游戏模式下的 ASR 热词（版本化）；非游戏模式返回 None → 走原来的闭集热词。"""
    if not game_mode(cfg):
        return None
    prof = _game_profile_obj(cfg)
    if prof is None:
        return None
    words = prof.hotwords()
    if ascii_only:
        words = [w for w in words if any(c.isascii() and c.isalpha() for c in w)]
    return "、".join(words)


def _game_parse(text, cfg):
    """游戏模式的规则直通；没命中返回 None（交给模型 / 桌面规则）。

    ⭐ 只放行 profile 里【激活】的意图（active_ids）——「收缩意图、一点点加」：
       没激活的意图即使规则能命中也不执行，避免动作词互相抢语义。
    """
    if not game_mode(cfg):
        return None
    prof = _game_profile_obj(cfg)
    if prof is None:
        return None
    hit = prof.parse(text)
    if not hit:
        return None
    if hit[0] not in prof.active_ids:
        return None
    return hit


def parse_multi(text, cfg, max_clauses=6):
    """⭐ 多指令拆分：一句话里多个命令 → [(intent, arg), ...]（≥2 条才算）。

    为什么需要：红警里「电厂、精炼厂」「造两个矿车，再造一个电厂」这种连说很自然，
    原来只执行第一句（实测：「电厂、金炼厂。」只造了电厂）。

    切分用仓库现成的 split_clauses（标点 → 顺承词 → 触发词再出现），
    逐句解析后只保留真的命中意图的分句；不足 2 条返回 None（交回单条路径）。
    """
    if not text:
        return None
    try:
        parts = split_clauses(text, cfg, max_clauses=max_clauses)
    except Exception:
        return None
    if len(parts) < 2:
        return None

    out = []
    for p in parts:
        try:
            got = parse(p, cfg)
        except Exception:
            got = None
        if got and got[0] != "unknown":
            out.append(got)
    if len(out) >= 2:
        return out
    return None


def _game_known_words(cfg):
    """游戏模式下额外算作「已知词」的词：profile 的热词 + 单位/建筑别名。

    ⚠️ 血债：voice_agent.plausible() 有一条判据「必须含触发词或闭集名字」，
       而游戏配置里 apps/urls/files 是空的 → 「两个矿车」这种【没有触发词但完全合法】
       的口令会被判成"不是命令"直接丢掉（实测：转写对了、模型也判 game_build，
       但 B 路没产出动作）。所以游戏词表必须并进去。
    """
    if not game_mode(cfg):
        return set()
    prof = _game_profile_obj(cfg)
    if prof is None:
        return set()
    words = set(prof.hotwords())
    for item in prof.items:
        words.add(item.get("name", ""))
        words.update(item.get("aliases") or [])
    return {w for w in words if w}


def game_status(cfg=None):
    """诊断：profile 版本 + 桥探活（给 --list / 排障用）。"""
    if _OPENRA is None:
        return {"ok": False, "err": f"openra 模块未加载（{_OPENRA_ERR}）"}
    try:
        return _OPENRA.probe(cfg)
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}: {e}"}
