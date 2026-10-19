# -*- coding: utf-8 -*-
"""音素级谐音匹配 v2（中英双路 + 声韵母近似 + 打分制）

对比 v1（actions.py::_homophone_hit 硬规则）的四处改动：

① 【剥命令前缀 + 滑动子串】修 bug
   旧：_pinyin('打开记事板')="dakaijishiban" vs _pinyin('记事本')="jishiben" → 距离 8 → 永不命中
   新：剥掉「打开」→ "jishiban" vs "jishiben" → 距离 1 → 命中

② 【声母/韵母近似表】—— 只放 ASR 真会混的对（保守，避免假命中）
   声母：zh/z  ch/c  sh/s  n/l  f/h  r/l        ← 不放 j/q、b/p（这些是不同音位）
   韵母：an/ang  en/eng  in/ing  an/en  ang/eng  ian/iang  uan/uang

③ 【英文双路】—— 词元匹配（≥4 字符）保留旧版能力
   旧版靠 _tokens 命中「DVC Harness」→ harness ∈ DeepSeek 别名
   v1 把这个丢了 → 补回来

④ 【打分制 + argmax + 阈值 + margin 拒识】
   替代「要么唯一命中要么放弃」
"""
from __future__ import annotations
import re
import unicodedata

try:
    from pypinyin import pinyin as _py
    from pypinyin.contrib.tone_convert import to_initials as _ti, to_finals_tone as _tf
    _HAS_PY = True
except Exception:
    _HAS_PY = False


def _strip_tone(s: str) -> str:
    """去掉声调符号：í→i  ī→i  ǎn→an  ěn→en

    ★ 必须做这一步：to_finals_tone 会把声调留在韵母里，
      导致「直呼」(zh,í) 和「知乎」(zh,ī) 被判成不同音素 —— 罚 0.4 分。
      而 ASR 的声调错误非常常见，谐音匹配不该因声调不同就判错。
    """
    return "".join(c for c in unicodedata.normalize("NFD", s or "")
                   if not unicodedata.combining(c))

# ── 近似音素组（保守：只放 ASR 真会混的）──
_INIT_GROUPS = ["zh z", "ch c", "sh s", "n l", "f h", "r l",
                # 拉丁转写桥组（2026-10-18 统一标注方案）：迪普西克/k→g 类
                "j q x", "g k", "s x", "sh x"]
_FIN_GROUPS = ["an ang", "en eng", "in ing", "an en", "ang eng",
               "ian iang", "uan uang", "o uo", "ai ei"]

def _tab(groups):
    t = {}
    for g in groups:
        xs = g.split()
        for i, a in enumerate(xs):
            for b in xs[i+1:]:
                t[(a, b)] = t[(b, a)] = True
    return t
_SIM_I, _SIM_F = _tab(_INIT_GROUPS), _tab(_FIN_GROUPS)

_PREFIXES = ["帮我精确查找", "帮我精确搜索", "我要精确查找", "精确查找", "精确搜索",
             "精确搜", "查找", "搜索", "搜一下", "帮我", "麻烦", "给我", "我要",
             "我想", "请", "打开", "启动", "开启", "关闭", "关掉", "退出",
             "查看", "看一下", "运行", "执行", "切换", "新建", "创建",
             "删除", "清空", "终止", "停止", "开始"]

def syllables(text: str):
    """中文 → [(声母, 韵母), ...]"""
    out = []
    for ch in (text or ""):
        if "\u4e00" <= ch <= "\u9fff":
            if not _HAS_PY:
                out.append((ch, ""))
                continue
            p = _py(ch)[0][0]
            out.append((_ti(p, strict=False), _strip_tone(_tf(p, strict=False))))
    return out

# 可调权重
W_INIT, W_FIN = 0.6, 0.4
C_SIM = 0.4        # 近似音素替换代价

def _ph_cost(a, b):
    if a == b:
        return 0.0
    ia, fa = a; ib, fb = b
    ci = 0.0 if ia == ib else (C_SIM if (ia, ib) in _SIM_I else 1.0)
    cf = 0.0 if fa == fb else (C_SIM if (fa, fb) in _SIM_F else 1.0)
    return W_INIT * ci + W_FIN * cf


def _half_miss(a, b):
    """★ 模板消费门槛:任一维度完全脱靶(超出相似组) → 不许消费。

    (t,u) vs (t,ian):声母同、韵母全错 —— 读法相差太远的"半个音节"
    不该被算作命中(油田→优图 假阳性的根子)。
    """
    ia, fa = a[0], a[1]
    ib, fb = b[0], b[1]
    if ia != ib and (ia, ib) not in _SIM_I:
        return True
    if fa != fb and (fa, fb) not in _SIM_F:
        return True
    return False

def ph_dist(A, B):
    n, m = len(A), len(B)
    if n == 0 and m == 0: return 0.0
    if n == 0 or m == 0: return float(max(n, m))
    prev = [float(j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [float(i)] + [0.0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1.0, cur[j-1] + 1.0, prev[j-1] + _ph_cost(A[i-1], B[j-1]))
        prev = cur
    return prev[m]

def ph_sim(A, B):
    return max(0.0, 1.0 - ph_dist(A, B) / max(1, len(A), len(B)))

def variants(text: str, substrings: bool = True):
    """音节序列变体

    substrings=True  → 整串 + 去标点片段 + 剥前缀 + 【滑动子串】
    substrings=False → 只有整串 + 去标点片段 + 剥前缀

    ★ 为什么要分开：滑动子串是给【打分】用的（找"整句里夹着的目标词"），
      但【精确匹配快路径】绝不能用它 —— 否则「办公助手」的滑动子串「办公」
      会命中另一个候选，变成"两个精确命中" → 被拒识（实测 8/37 条踩中）。
    """
    s = (text or "").strip()
    if not s: return []
    outs = []
    base = syllables(s)
    if base: outs.append(base)
    for frag in re.split(r"[^\w\u4e00-\u9fff]+", s):
        if len(frag) >= 2:
            v = syllables(frag)
            if v: outs.append(v)
    for pre in _PREFIXES:
        if s.startswith(pre) and len(s) > len(pre):
            v = syllables(s[len(pre):])
            if v: outs.append(v)
    if substrings and len(base) >= 4:
        for L in (4, 3, 2):
            for st in range(0, len(base) - L + 1):
                outs.append(base[st:st + L])
    seen, uniq = set(), []
    for v in outs:
        k = tuple(v)
        if k and k not in seen:
            seen.add(k); uniq.append(v)
    return uniq

# ── 英文词元路（保留旧版能力）──
def _tokens(s, minlen=4):
    return {t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if len(t) >= minlen}

def eng_score(text, cand):
    """英文词元交集 → 命中给满分"""
    st, ct = _tokens(text), _tokens(cand)
    if st and ct and (st & ct):
        return 1.0
    return 0.0

def score_all(text: str, cands):
    """给闭集里每条候选打分，返回 [(分数, 候选), ...] 降序（不拒识、不丢项）"""
    vs_all = variants(text, substrings=True)
    vs_exact = variants(text, substrings=False)
    if not vs_all or not cands:
        return []
    exact_keys = {tuple(v) for v in vs_exact}
    out = []
    for c in cands:
        cv = syllables(c)
        sc = eng_score(text, c)
        if cv:
            sc = max(sc, max((ph_sim(v, cv) for v in vs_all), default=0.0))
            if tuple(cv) in exact_keys:
                sc = max(sc, 1.0)          # 精确同音 → 满分
        if sc > 0:
            out.append((sc, c))
    out.sort(key=lambda x: (-x[0], x[1]))
    return out


def rank_candidates(text: str, cands, topk: int = 5, floor: float = 0.40):
    """★「保证不漏」接口：返回 top-K 候选，绝不做单点拒识

    下游拿到候选列表后自己决定（阈值/交互确认/走 AI）。
    只要真实意图的分数 ≥ floor，它就一定在列表里 —— 不会因为
    margin 拒识而被丢掉。
    """
    sc = score_all(text, cands)
    return [(c, s) for s, c in sc[:topk] if s >= floor]


def best_match(text: str, cands, thr: float = 0.60, margin: float = 0.03):
    """单点判定（高精度场景）：返回 (最佳候选, 分数, 第二分数)

    低于 thr 或 margin 不足 → (None, ...)。
    ★ 要"保证不漏"就用 rank_candidates()，不要用这个。
    """
    sc = score_all(text, cands)
    if not sc:
        return None, 0.0, 0.0
    top = sc[0]
    second = sc[1][0] if len(sc) > 1 else 0.0
    if top[0] < thr:
        return None, top[0], second
    if len(sc) > 1 and sc[1][1].lower() != top[1].lower() and sc[1][0] >= top[0] - margin:
        return None, top[0], second
    return top[1], top[0], second


# ══════════════════════════════════════════════════════════════
#  v2 统一标注 + 模板前缀树（2026-10-18 定案）
#  · 拉丁别名不再只靠词元交集：G2P 转成【同一拼音音节空间】的
#    (声母,韵母)序列 —— 中文与拉丁汇成一张代价表、一棵模板树。
#  · 模板（动作×别名×词缀）批量生成 → 前缀 trie 共享节点；
#    口令标注串（约 7-8 音节）对模板做【子序列匹配】：
#      模板符号必须按序【全部】消费完才算命中（子序列约束），
#      口令侧可跳过语气成分（gap 罚分）。动作与修饰词天然进裁决。
# ══════════════════════════════════════════════════════════════

# ── 拉丁 G2P 三层 ─────────────────────────────────────────────
# 1) 闭集纯拉丁别名【白名单】（权威中文读法；ASR 常见错型全收为变体）
#    来源：intent_config.json 三表并集的 46 个纯拉丁别名，2026-10-18 全量标定
_ENG_EXCEPT = {
    "deepseek": [
        [("d", "i"), ("p", "u"), ("x", "i"), ("k", "e")],    # 迪普西克
        [("d", "i"), ("p", "u"), ("sh", "i"), ("k", "e")],   # 迪普什克
        [("d", "i"), ("p", "u"), ("s", "i"), ("k", "e")],    # 迪普西克(s)
    ],
    "clash": [
        [("k", "e"), ("l", "a"), ("sh", "i")],              # 克拉什(什shí)
        [("k", "e"), ("l", "a"), ("sh", "en")],             # 克拉什(什shén)
        [("g", "e"), ("l", "a"), ("x", "i")],               # 格拉西
        [("k", "a"), ("l", "a"), ("sh", "i")],              # 卡拉什
    ],
    "trae": [[("t", "e"), ("l", "ei")], [("t", "u"), ("l", "ei")]],   # 特雷
    "ollama": [[("", "ou"), ("l", "a"), ("m", "a")]],       # 欧拉玛
    "harness": [[("h", "a"), ("n", "i"), ("s", "i")],       # 哈尼斯
                [("h", "a"), ("n", "en"), ("s", "i")]],     # 哈内斯
    "github": [[("j", "i"), ("t", "e"), ("h", "a"), ("b", "u")]],    # 吉特哈布
    "llama": [[("l", "a"), ("m", "a")]],                    # 拉玛
    "python": [[("p", "ai"), ("s", "en")], [("p", "ai"), ("x", "en")]],  # 派森
    "shell": [[("x", "ie"), ("", "er")], [("sh", "e"), ("l", "e")]],  # 谢尔/谢了
    "dvc": [[("d", "i"), ("w", "ei"), ("x", "i")]],         # 迪威西
    "api": [[("", "ai"), ("p", "ai"), ("", "ai")],          # 诶派艾(诶/艾多音)
            [("", "ei"), ("p", "ai"), ("", "ai")]],
    "gpt": [[("j", "i"), ("p", "i"), ("t", "i")]],          # 吉皮提
    # ── 补齐闭集其余纯拉丁别名的读音（2026-10-18 全量标定）──
    "blender": [[("b", "u"), ("l", "un"), ("d", "e")],      # 布伦德
                [("b", "u"), ("l", "an"), ("d", "e")]],     # 布蓝德
    "steam": [[("s", "i"), ("t", "i"), ("m", "u")],        # 斯提姆
              [("s", "i"), ("t", "i"), ("m", "e")]],        # 斯提姆(英读)
    "edge": [[("", "ai"), ("j", "i")], [("", "e"), ("j", "i")]],     # 艾吉
    "cmd": [[("x", "i"), ("", "ai"), ("m", "u"), ("d", "i")],        # 西艾姆迪
            [("x", "i"), ("", "e"), ("m", "e"), ("d", "i")]],       # 西艾么迪(英读)
    "doubao": [[("d", "ou"), ("b", "ao")]],                 # 豆包（拼音直读）
    "qwen": [[("q", "ian"), ("w", "en")], [("k", "uan"), ("w", "en")]],  # 千问/宽问
    "jiqizhixin": [[("j", "i"), ("q", "i"), ("zh", "i"), ("x", "in")]],  # 机器之心
    "wiki": [[("w", "ei"), ("j", "i")],                     # 维基(中文读法)
             [("w", "i"), ("k", "i")]],                    # wiki(拉丁读法)
    "wikipedia": [[("w", "ei"), ("j", "i"), ("b", "ai"), ("k", "e")],   # 维基百科
                  [("w", "i"), ("k", "i"), ("p", "i"), ("d", "i")]],    # latin
    "arxiv": [[("", "a"), ("k", "e"), ("s", "ai"), ("w", "ei")],     # 阿克赛唯
              [("", "a"), ("e", "l"), ("s", "ai"), ("f", "u")]],         # 阿尔赛夫
    "anthropic": [[("", "an"), ("s", "i"), ("l", "uo"), ("p", "i"), ("k", "e")],  # 安思罗匹克
                  [("", "an"), ("sh", "i"), ("l", "uo"), ("p", "i"), ("k", "e")]],
    "claude": [[("k", "e"), ("l", "ao"), ("d", "e")],       # 克劳德
               [("k", "e"), ("l", "uo"), ("d", "e")]],      # 克罗德
    "chatgpt": [[("q", "ia"), ("t", "e"), ("j", "i"), ("p", "i"), ("t", "i")],   # 恰特吉皮提
                [("k", "a"), ("t", "e"), ("j", "i"), ("p", "i"), ("t", "i")]],# 卡特吉皮提
    "huggingface": [[("h", "a"), ("j", "ing"), ("f", "ei"), ("s", "i")],        # 哈京费斯
                    [("h", "a"), ("g", "e"), ("y", "in"), ("f", "ei"), ("s", "i")]],
    "docker": [[("d", "ao"), ("k", "e")]],                  # 道克
    "dockerhub": [[("d", "ao"), ("k", "e"), ("h", "a"), ("b", "u")]],  # 道克哈布
    "hub": [[("h", "a"), ("b", "u")], [("h", "u"), ("b", "e")]],        # 哈布
    "cnblogs": [[("b", "o"), ("k", "e"), ("y", "uan")]],     # 博客园(中文读法)
    "kaggle": [[("k", "a"), ("g", "ai"), ("", "er")]],       # 卡盖尔
    "stackoverflow": [[("s", "i"), ("t", "ai"), ("k", "e"), ("", "ao"), ("w", "o"),
                       ("f", "u"), ("l", "uo")]],            # 斯泰克奥沃弗洛
    "workbuddy": [[("w", "o"), ("k", "e"), ("b", "a"), ("d", "i")]],   # 沃克巴迪
    "gitee": [[("j", "i"), ("t", "i")]],                    # 吉提(git-ee;不留吉特,gitee/github 会撞)
    "googlescholar": [[("g", "u"), ("g", "e"), ("x", "ue"), ("sh", "u")],    # 谷歌学术
                      [("g", "u"), ("g", "ou"), ("s", "i"), ("k", "a"), ("l", "e")]],
    "google": [[("g", "u"), ("g", "e")], [("g", "u"), ("g", "ou")]],  # 谷歌
    "scholar": [[("s", "i"), ("k", "a"), ("l", "e")]],      # 斯卡勒
    "hackernews": [[("h", "a"), ("k", "e"), ("x", "in"), ("w", "en")],       # 哈克新闻
                   [("h", "a"), ("k", "e"), ("n", "iu"), ("s", "i")]],      # 哈克牛斯
    "hacker": [[("h", "a"), ("k", "e")]],                   # 哈克
    "news": [[("n", "iu"), ("s", "i")], [("n", "en"), ("s", "i")]],  # 牛斯
    "infoq": [[("y", "in"), ("f", "u"), ("k", "iu")],        # 因弗Q
              [("y", "in"), ("f", "u"), ("q", "iu")]],       # 因弗求
    "modelscope": [[("m", "o"), ("d", "e"), ("l", "ei"), ("s", "i"), ("k", "ou")],   # 莫德勒斯扣
                   [("m", "o"), ("d", "a"), ("s", "i"), ("k", "ou")]],             # 魔搭思扣
    "modelcope": [[("m", "o"), ("d", "e"), ("l", "ei"), ("k", "ou"), ("p", "o")]],   # 莫德勒口坡
    "openai": [[("", "ou"), ("p", "en"), ("", "ai")],       # 欧盆艾
               [("", "ou"), ("p", "en"), ("", "ai"), ("", "ai")]],  # 欧盆诶艾
    "oschina": [[("k", "ai"), ("y", "uan"), ("zh", "ong"), ("g", "uo")]],    # 开源中国
    "papers": [[("p", "ei"), ("p", "o"), ("s", "i")],       # 陪珀斯
               [("p", "ei"), ("p", "e"), ("s", "i")]],      # 陪珀斯(俗读)
    "with": [[("w", "ei"), ("s", "i")]],                    # 危思
    "code": [[("k", "ou"), ("d", "e")], [("k", "ao"), ("d", "e")]],  # 口德
    "paperswithcode": [[("p", "ei"), ("p", "o"), ("s", "i"), ("w", "ei"), ("s", "i"),
                        ("k", "ou"), ("d", "e")]],          # 全串(珀pò)
    "powershell": [[("p", "ao"), ("w", "o"), ("x", "ie"), ("", "er")],       # 泡沃谢尔
                   [("p", "ao"), ("w", "e"), ("sh", "e"), ("l", "e")]],     # 泡卧谢了
    "pypi": [[("p", "ai"), ("p", "ai")]],                   # 派派
    "youtube": [[("y", "ou"), ("g", "uan")], [("y", "ou"), ("t", "u")]],     # 油管/优图
    "work": [[("w", "o"), ("k", "e")]],                     # 沃克
    "buddy": [[("b", "a"), ("d", "i")], [("b", "u"), ("d", "i")]],   # 巴迪
    "stack": [[("s", "i"), ("t", "ai"), ("k", "e")]],       # 斯泰克
}

# 2) 全大写按字母念（DVC / API / GPT 这类缩写）
#    每个字母 = 若干【槽】，每槽 = 候选 (声母,韵母) 列表
_ENG_LETTER = {
    "a": [[("", "ai")]], "b": [[("b", "i")]], "c": [[("x", "i"), ("s", "i")]],
    "d": [[("d", "i")]], "e": [[("", "i")]],
    "f": [[("", "e")], [("f", "i")]], "g": [[("j", "i")]],
    "h": [[("", "ai")], [("q", "i")]], "i": [[("", "ai")]],
    "j": [[("j", "ie")]], "k": [[("k", "ei")]],
    "l": [[("", "e")], [("l", "e")]], "m": [[("", "e")], [("m", "e")]],
    "n": [[("", "e")], [("n", "e")]], "o": [[("", "ou")]],
    "p": [[("p", "i")]], "q": [[("k", "iu")]],
    "r": [[("", "a")], [("r", "er")]], "s": [[("", "e")], [("s", "i")]],
    "t": [[("t", "i")]], "u": [[("", "iu")]], "v": [[("w", "ei")]],
    "w": [[("d", "a")], [("b", "u")]],
    "x": [[("", "e")], [("k", "e")], [("s", "i"), ("x", "i")]],
    "y": [[("w", "ai")]], "z": [[("z", "i")], [("x", "i")]],
}

# 3) 拼写规则兜底（近似正字法；最长簇优先）
_ENG_CC = {"sh": ["sh"], "ch": ["q"], "th": ["s"], "ph": ["f"], "zh": ["zh"],
           "ck": ["k"], "wh": ["w"], "qu": ["k"], "dg": ["j"]}
_ENG_VOWEL = {"ee": ["i"], "ea": ["i"], "ie": ["i"], "oo": ["u"], "ao": ["ao"],
              "ou": ["ao", "ou"], "ow": ["ou", "ao"], "ai": ["ai"],
              "ay": ["ai"], "oa": ["ou", "o"], "oi": ["ai"], "oy": ["ai"],
              "ar": ["a"], "er": ["e"], "or": ["o"], "ir": ["e"], "ur": ["e"],
              "a": ["a"], "e": ["e"], "i": ["i"], "o": ["o", "ou"],
              "u": ["u", "a"], "y": ["i"]}
_ENG_ONSET = {"b": ["b"], "c": ["k", "s"], "d": ["d"], "f": ["f"],
              "g": ["g", "j"], "h": ["h"], "j": ["j", "y"], "k": ["k"],
              "l": ["l"], "m": ["m"], "n": ["n", "l"], "p": ["p"],
              "q": ["k"], "r": ["r", "l"], "s": ["s", "x"], "t": ["t"],
              "v": ["w"], "w": ["w"], "x": ["k", "x"], "y": ["y"],
              "z": ["z", "x"]}
# 尾辅音并入前一韵母：a+n=an、i+ng=ing …（找不到就落成独立 filler 音节）
_CODA = {("a", "n"): "an", ("a", "ng"): "ang", ("e", "n"): "en",
         ("e", "ng"): "eng", ("i", "n"): "in", ("i", "ng"): "ing",
         ("o", "n"): "un", ("o", "ng"): "ong", ("u", "n"): "un",
         ("u", "ng"): "ong"}
_FILL_FIN = {"s": "i", "x": "i", "z": "i", "sh": "i", "q": "i", "j": "i",
             "r": "e", "l": "e", "m": "e", "n": "e"}


def _prod(slots, cap=4):
    """槽列表 → 若干套音节序列（每槽取前 2 个候选；总变体 ≤ cap）。"""
    seqs = [[]]
    for sl in slots:
        cands = sl[:2] if sl else [("", "e")]
        seqs = [s + [c] for s in seqs for c in cands]
        if len(seqs) > cap:
            seqs = seqs[:cap]
    out, seen = [], set()
    for s in seqs:
        k = tuple(s)
        if k and k not in seen:
            seen.add(k)
            out.append(list(s))
        if len(out) >= cap:
            break
    return out or [[("", "e")]]


def _commit_v(cons, fins, slots):
    """辅音簇 + 元音簇 → 槽。簇内末辅音当 onset，其余落 filler 独立槽。

    cons 是辅音簇列表（每项 = 候选 onset 的字符串列表）。
    """
    if cons:
        for c in cons[:-1]:
            slots.append([(r, _FILL_FIN.get(r, "e")) for r in c])
        onsets = cons[-1]
    else:
        onsets = [""]
    slot = [(r, f) for r in onsets for f in fins]
    if slot:
        slots.append(slot)


def _eng_slots(word):
    """拉丁词 → 槽列表（每槽 = 候选 (声母,韵母) 元组列表）。"""
    slots, cons, i = [], [], 0
    while i < len(word):
        two = word[i:i + 2]
        if two in _ENG_CC:
            cons.append(list(_ENG_CC[two]))
            i += 2
            continue
        if two in _ENG_VOWEL:
            _commit_v(cons, _ENG_VOWEL[two], slots)
            cons = []
            i += 2
            continue
        one = word[i]
        if one in _ENG_VOWEL:
            _commit_v(cons, _ENG_VOWEL[one], slots)
            cons = []
            i += 1
            continue
        if one in _ENG_ONSET:
            cons.append(list(_ENG_ONSET[one]))
            i += 1
            continue
        i += 1                       # 未知字形丢弃
    if cons:
        r0 = cons[-1][0]
        prev = slots[-1][-1] if slots and slots[-1] else None
        merged = _CODA.get((prev[1], r0)) if prev else None
        if merged and prev:
            slots[-1][-1] = (prev[0], merged)
        else:
            for c in cons:
                slots.append([(r, _FILL_FIN.get(r, "e")) for r in c])
    return slots


def _eng_syl_variants(word):
    """拉丁词 → 若干套 (声母,韵母) 序列。产品词走白名单，大写走字母念，其余规则。"""
    w = (word or "").strip()
    if not w or not all(ch.isascii() and ch.isalpha() for ch in w):
        return []
    wl = w.lower()
    if wl in _ENG_EXCEPT:
        return [list(s) for s in _ENG_EXCEPT[wl]]
    if len(wl) <= 4 and (w.isupper() or not any(c in "aeiouy" for c in wl)):
        # 缩写按字母念：DVC / API 大写；cmd/hf/hn/wb/dsh 无元音短词
        slots = []
        for ch in wl:
            slots.extend(_ENG_LETTER.get(ch, [[("e", "e")]]))
        return _prod(slots, cap=4)
    return _prod(_eng_slots(wl), cap=4)


def latin_variants(word):
    """拉丁词的全部音节序列变体（模板树对每一套都收录）。"""
    return _eng_syl_variants(word)


def text_stream(text):
    """★ 统一标注流：汉字→拼音(声母,韵母)；拉丁 run→G2P(首选变体)。

    中文同音转写（迪普西克/克拉什）与拉丁原文（DeepSeek/Clash）在此汇成
    同一个 (声母,韵母) 空间 —— 模板树对二者一视同仁。
    """
    out, buf = [], ""
    for ch in (text or ""):
        if "\u4e00" <= ch <= "\u9fff":
            if buf:
                v = _eng_syl_variants(buf)
                if v:
                    out.extend(v[0])
                buf = ""
            out.extend(syllables(ch))
        elif ch.isascii() and ch.isalpha():
            buf += ch
        else:
            if buf:
                v = _eng_syl_variants(buf)
                if v:
                    out.extend(v[0])
                buf = ""
    if buf:
        v = _eng_syl_variants(buf)
        if v:
            out.extend(v[0])
    return out


class _TNode:
    __slots__ = ("edges", "pay", "depth", "rem", "lat")

    def __init__(self, depth=0):
        self.edges = {}
        self.pay = []          # 该节点作为模板叶子时的 payload(可多个)
        self.depth = depth     # 根到该节点的模板符号数
        self.rem = 0           # 经此节点可达的最深叶子还需几个符号
        self.lat = False       # 到此符为止的模板符号是否来自拉丁 G2P


class SylTrie:
    """模板前缀树 + 口令标注串的【子序列匹配】。

    add(seq, payload, lat)：模板音节序列注册，同前缀模板共享节点；
    best(seq)：口令标注串 DP ——
        状态 = (trie节点, 累计代价)；两种转移：
          ① 消费当前口令音节（音素模糊代价 ≤ cap 才许消费，含精确 0；
             任一维度完全脱靶的"半个音节"不许消费 —— _half_miss）；
          ② 跳过当前口令音节（gap 罚分，语气成分/附加语）；
            ★ 拉丁连读中缝加重(基 gap_la)：模板上个符号与后继都是拉丁
              时，中间的口令音节不该当语气词 —— 字母缩写/拉丁名里每个
              字母都实义（G-T 把 gpt 的 p 吞掉是本规则的根子）。
        模板符号必须按序【全部】消费完（到达叶子）才计分；
        分数 sim = 1 - cost / 模板长度。
    剪枝：cost 已 > (1-min_sim)×可达叶深 的状态直接丢 —— 物理上到不
    到 min_sim 了，不必算完。
    """

    def __init__(self, gap_la=0.35):
        self.root = _TNode(0)
        self.gap_la = gap_la   # 拉丁连读中缝的加重跳过罚分

    def add(self, seq, payload, lat=None):
        """lat:与 seq 等长的 0/1 列表,标注每个模板符号是否来自拉丁 G2P。

        用于跳过罚分分级:拉丁连读的中缝不允许当语气成分跳过
        (G-T 把 gpt 的 p 跳掉的教训),CJK 与拉丁的衔接处跳过不罚重。
        """
        if not seq:
            return
        L = len(seq)
        self.root.rem = max(self.root.rem, L)
        n = self.root
        for i, k in enumerate(seq):
            key = (k[0], k[1])
            n2 = n.edges.get(key)
            if n2 is None:
                n2 = _TNode(n.depth + 1)
                n.edges[key] = n2
            if lat is not None and i < len(lat) and lat[i]:
                n2.lat = True
            n2.rem = max(n2.rem, L - n2.depth)
            n = n2
        if payload not in n.pay:
            n.pay.append(payload)

    def best(self, seq, gap=0.10, cap=0.5, min_sim=0.30):
        if not seq:
            return []
        states = {id(self.root): (self.root, 0.0)}
        for sym in seq:
            nxt = {}
            for nid, (node, cost) in states.items():
                reach = node.depth + node.rem
                # ① 跳过口令侧音节（语气成分/模板完成后的尾部成分）
                #    叶子(rem=0)也允许 —— 模板已消费完、后面还有"的。/我们…"。
                #    拉丁连读中缝(本节点拉丁 AND 至少一个拉丁后继) → 重罚 gap_la。
                if (node.rem or node.pay) and \
                        (cost + gap) <= (1.0 - min_sim) * reach:
                    g_eff = self.gap_la \
                        if (node.lat and any(ch.lat for ch in node.edges.values())) \
                        else gap
                    step = cost + g_eff
                    cur = nxt.get(nid)
                    if cur is None or step < cur[1]:
                        nxt[nid] = (node, step)
                # ② 消费口令音节（模板符号按序推进；半音节脱靶拒消费）
                for key, ch in node.edges.items():
                    c = _ph_cost(key, sym)
                    if c <= cap and not _half_miss(key, sym):
                        cc = cost + c
                        if ch.rem or ch.pay:
                            if cc <= (1.0 - min_sim) * (ch.depth + ch.rem):
                                cur = nxt.get(id(ch))
                                if cur is None or cc < cur[1]:
                                    nxt[id(ch)] = (ch, cc)
            states = nxt
            if not states:
                break
        res, dedu = [], {}
        for nid, (node, cost) in states.items():
            if node.pay and node.depth:
                sim = 1.0 - cost / node.depth
                if sim < min_sim:
                    continue
                for p in node.pay:
                    res.append((sim, p, node.depth, cost))
        for sim, p, d, c in res:
            if p not in dedu or sim > dedu[p][0]:
                dedu[p] = (sim, d, c)
        return sorted(((s, p, d, c) for p, (s, d, c) in dedu.items()),
                      key=lambda x: (-x[0], x[1]))
