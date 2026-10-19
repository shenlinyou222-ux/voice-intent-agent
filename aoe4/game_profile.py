# -*- coding: utf-8 -*-
"""AoE4 语音控制的【规则层 + 闭集强制对齐】（收缩版）

★ 闭集（用户定义，只放开这一种形态）：
      <数字词>农       数字词 ∈ {一,二,两,三,四,五,六,七,八,九,十} ∪ 阿拉伯数字
   例：一农 二农 三农 五农 十农 2农 ３农
★ 输出：
      aoe4_train_villager(n)   —— 城镇中心训练 n 个农民
★ 其它一律拒识（返回 None）：动词式（生成三农/造农民）、长句、ping、"造房子" …

方向 1：闭集强制对齐（消灭谐音，如「二龙」→「二农」）
   ① 谐音/异体字归一化（_HOMO 表）
   ② 句尾锚定的正则（_PAT）：只认句尾的 <数字>农，长句里的不认
   ③ 音节级模糊匹配（_SYL）：归一化表没覆盖的谐音兜底；分数不够 → 拒识
   ★ 宁可不执行，也不执行错的命令。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILES = HERE / "profiles"
DEFAULT_PROFILE = "aoe4-v1"

# ── ① 归一化表：只覆盖闭集用字（补 / 农），繁→简也在这一层 ✓ ──────────
#   用户指定：闭集只有「补农」一族（不再有 xx农 ✗），繁简只在对齐层归一 ✓
_HOMO = {
    # 「补」的谐音 + 繁体
    "补": "补", "捕": "补", "卜": "补", "哺": "补", "浦": "补", "埔": "补",
    "補": "补", "佈": "补",
    # 「农」的谐音 + 繁体
    "农": "农", "浓": "农", "龙": "农", "弄": "农", "侬": "农", "脓": "农",
    "隆": "农", "笼": "农", "拢": "农", "垄": "农", "聋": "农", "哝": "农",
    "農": "农", "濃": "农", "龍": "农", "儂": "农", "膿": "农", "籠": "农",
    "隴": "农", "壟": "农", "聾": "农", "噥": "农",
    # 「民」的繁体（补农民）
    "民": "民",
}

# ── ② 闭集 = 只有「补农」一族（用户指定：不再有 xx农 ✗） ──────────────
#   "意义相近的词"：都是"补农民"这个意思的常见说法 ✓
_TAIL = r"[。.!！?？啊吧呀哦呢嘛~～\s]{0,1}$"          # 允许 1 个语气/标点尾巴
_CANON = "补农"                                        # 规范写法
_VARIANTS = ["补充农民", "补一下农", "补个农民", "补点农民", "补个农",
             "补农民", "补农"]
_PAT = re.compile(rf"^({'|'.join(_VARIANTS)}){_TAIL}")
_CANDIDATES = _VARIANTS

# ── ③ 音节表（归一化表没覆盖的谐音兜底） ────────────────────────────
_SYL = {
    "补": "bu", "农": "nong", "民": "min",
}


def normalize(text: str) -> str:
    """① 谐音/异体字归一化（含 繁→简；全角数字也归一）。"""
    out = []
    for ch in text or "":
        if "\uff10" <= ch <= "\uff19":
            ch = chr(ord(ch) - 0xFEE0)
        out.append(_HOMO.get(ch, ch))
    return "".join(out)


def _syl(ch: str) -> str:
    return _SYL.get(ch, ch)


def _fuzzy_score(t: str) -> float:
    """③ 句尾 2 字窗口 vs 闭集候选（音节级）；返回最高分（0~1）。"""
    t = t.rstrip("。.!！?？啊吧呀哦呢嘛~～ \t")
    if len(t) != 2:            # ★ 只认恰好两字（"补农"）；长句一律不猜
        return 0.0
    w = t[-2:]
    best = 0.0
    for cand in _CANDIDATES:
        if len(cand) != 2:
            continue
        hit = sum(1 for a, b in zip(w, cand) if a == b or _syl(a) == _syl(b))
        best = max(best, hit / 2.0)
    return best



# ── ④ 工人调配解析：「七肉二金三石」→ (food, wood, gold, stone) ──────────
_ALLOC_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_ALLOC_RES = {"肉": "food", "木": "wood", "金": "gold", "斤": "gold",
              "石": "stone", "矿": "stone", "石头": "stone"}


def parse_alloc(text: str):
    """把「七肉二金三石」解析成 (food, wood, gold, stone)；没有资源词就返回 None。

    支持：汉字数字（一~十，含 十二/二十 这种两位数）、阿拉伯数字；
    资源词：肉 food / 木 wood / 金 斤 gold / 石 矿 石头 stone
    """
    t = normalize(text.strip())
    if not re.search(r"[肉木金斤石矿]", t):
        return None
    found = {}
    # 先抓「数字+资源词」
    for m in re.finditer(r"([0-9]+|[一二两三四五六七八九十]+)\s*([肉木金斤石矿])", t):
        num_s, res_s = m.group(1), m.group(2)
        if num_s.isdigit():
            n = int(num_s)
        else:
            # 汉字数字：支持 十 / 十二 / 二十 / 二十三
            n = 0
            if "十" in num_s:
                a, _, b = num_s.partition("十")
                n = (_ALLOC_NUM.get(a, 1) if a else 1) * 10 + (_ALLOC_NUM.get(b, 0) if b else 0)
            else:
                n = _ALLOC_NUM.get(num_s, 0)
        if n > 0:
            found[_ALLOC_RES[res_s]] = n
    if not found:
        return None
    # 没提到的资源 = 0（用户给的是完整分配 ✓）
    return (found.get("food", 0), found.get("wood", 0),
            found.get("gold", 0), found.get("stone", 0))


class Aoe4Profile:
    FUZZY_MIN = 0.9          # ★ 两字音节都必须对上（0.5 会把「三农」也放进来 ✗）

    def __init__(self, raw: dict):
        self.raw = raw
        # ★ 闭集从 profile.json 读（改闭集 = 改 JSON ✓ 不用动代码 ✓）
        global _VARIANTS, _PAT, _CANDIDATES
        _cs = [str(x) for x in (raw.get("closed_set") or []) if str(x).strip()]
        if _cs:
            _VARIANTS = sorted(_cs, key=len, reverse=True)   # 长词优先（补充农民 先于 补农）
            _PAT = re.compile(rf"^({'|'.join(_VARIANTS)}){_TAIL}")
            _CANDIDATES = _VARIANTS
        self.profile_id = raw.get("profile_id", DEFAULT_PROFILE)
        self.intents = raw.get("intents", [])
        self.active_ids = {i["id"] for i in self.intents}
        self._hotwords = raw.get("hotwords", [])

    def hotwords(self) -> list[str]:
        return list(self._hotwords)

    def parse(self, text: str) -> tuple[str, str] | None:
        """闭集强制对齐：归一化 → 句尾正则 → 音节模糊 → 门控。命中返回 (意图, n)。"""
        if not text:
            return None
        t = normalize(text.strip())
        # ★ 短语表（比闭集优先）：短语 → 动作:参数（来自 profile.json 的 phrases ✓）
        _ph = self.raw.get("phrases") or {}
        # ★ 工人调配：「七肉二金三石」→ alloc（比短语表灵活，组合无限 ✓）
        _al = parse_alloc(t)
        if _al is not None:
            return "aoe4_alloc", ",".join(str(x) for x in _al)
        if t in _ph:
            _spec = str(_ph[t])
            _a, _, _arg = _spec.partition(":")
            return _a, (_arg or "")
        # ★ 闭集是【全汉字】—— 含阿拉伯数字的一律拒识（用户指定：不需要数字）
        if re.search(r"[0-9]", t):
            return None

        m = _PAT.match(t)                              # ② 归一化后整句 ≈「补农」一族
        hit = bool(m)
        if not hit:
            # ③ 音节兜底：只看句尾恰好两字（补农）
            if _fuzzy_score(t) >= self.FUZZY_MIN and _syl(t[-1]) == "nong":
                hit = True
        if not hit:
            return None                                # ★ 拒识（不猜）
        return "aoe4_train_villager", "3"              # ★ 「补农」= 补农民 → 固定 3 个


def load(profile_id: str | None = None) -> Aoe4Profile:
    pid = profile_id or DEFAULT_PROFILE
    p = PROFILES / pid / "profile.json"
    raw: dict = {"profile_id": pid, "intents": [], "hotwords": []}
    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    hw = PROFILES / pid / "hotwords.txt"
    if hw.exists():
        try:
            raw["hotwords"] = [w.strip() for w in hw.read_text(encoding="utf-8").splitlines() if w.strip()]
        except Exception:
            pass
    return Aoe4Profile(raw)


if __name__ == "__main__":
    prof = load()
    print(f"profile={prof.profile_id}  意图={sorted(prof.active_ids)}")
    cases = [
        # 闭集：数字词 + 农（应全部命中 train）
        ("一农", "1"), ("二农", "2"), ("三农", "3"), ("四农", "4"), ("五农", "5"),
        ("六农", "6"), ("七农", "7"), ("八农", "8"), ("九农", "9"), ("十农", "10"),
        ("2农", "2"), ("３农", "3"), ("10农", "10"),
        ("三农。", "3"), ("三农啊", "3"),
        # 谐音（方向 1 的目标）
        ("二龙", "2"), ("仨农", "3"), ("五浓", "5"), ("衣农", "1"),
        ("散农", "3"), ("酒农", "9"), ("巴农", "8"), ("儿农", "2"),
        # 应拒识（收缩后只认 <数字>农）
        ("生成三农", None), ("刷二农", None), ("凭空来三个工人", None),
        ("多造点农民", None), ("造农民", None), ("农民", None),
        ("今天天气不错", None), ("造房子", None), ("二龙在天", None),
        ("打开浏览器", None), ("在吗", None), ("三农二农", None),
    ]
    ok = 0
    for text, exp in cases:
        got = prof.parse(text)
        good = (got is None and exp is None) or (got is not None and got[1] == exp)
        ok += good
        print(f"  {'✓' if good else '✗'} {text:14s} -> {got}   (期望 n={exp})")
    print(f"  {ok}/{len(cases)} 通过")
