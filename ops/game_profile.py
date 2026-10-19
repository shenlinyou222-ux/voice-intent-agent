# -*- coding: utf-8 -*-
"""运维语音域的【规则层 + 三档路由】。

═══ 三档路由（核心设计）═══
    语音 → ASR 文本
      ├─0. 危险动词（rm/删除/清空/格式化/drop…）  → 直接拒识
      ├─1. strong AI 触发（让ds/让AI/帮我看看…）   → ai      ← 优先级最高
      ├─2. 规则直通（短语表 + 候选表）             → rule    0 ms
      ├─3. embedding kNN（句向量，thr=0.45）       → embed   ~30 ms
      ├─4. weak AI 触发（分析/排查/为什么…）        → ai
      └─5. 拒识

═══ ★★ 为什么不用「单字谐音替换表」（血债）═══
    最初写了一版 _HOMO 逐字替换表（在→载、接→节、段→端…）。**实测是错的设计**：
      · 「在」→「载」（为修「负在」）把「谁**在**占内存」变成「谁**载**占内存」
        → 短语表匹配不上 → 落到 items 别名「内存」→ 命中 ops_mem ✗
      · 同类风险字：在/再/接/段/近/现/差/至 —— 全是高频字，**字越常见误伤越大**
      · 而且**误伤比不识别更隐蔽**：本该命中的命令静默漏掉，看不出错

    ✅ 正确做法：**拼音滑窗匹配，且候选至少 2 个字才参与模糊**（用户指定）。
       不做任何逐字替换，而是在输入上滑一个与候选**等长**的窗口，比拼音键。

       实测效果（pypinyin 无调拼音，同一批错例）：
           查磁盘 / 察磁盘   →  chacipan   ✓ 完全相同（察/查 同音）
           查显卡 / 查现卡   →  chaxianka  ✓ 完全相同（现/显 同音）
           查集群 / 查机群   →  chajiqun   ✓ 完全相同（机/集 近音）

       **同音字全部自动覆盖，一个手写表都不需要。**
       而「谁在占内存」保持原样，不会被误伤。

    ★ 为什么必须 ≥2 字：单字候选（如「查」「盘」）做拼音模糊会命中大量无关词
      —— 一个字的拼音太短，撞车概率高。**短于 2 字的候选只做字面匹配。**

    ★ 繁体字：**本域不处理**（用户决定：先不做，后续再说）。
      拼音路会顺带覆盖一部分（因为繁简同音），但**不作为设计目标、不写测试用例**，
      文件里也不出现繁体字。

═══ 匹配顺序（字面优先，拼音兜底）═══
    每一层都是「先字面，再拼音」：
      字面命中 → 确定性最高，直接用
      拼音命中 → 同音/繁体兜底
    这个顺序保证**不会因为模糊匹配而覆盖掉精确匹配**。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILES = HERE / "profiles"
DEFAULT_PROFILE = "ops-v1"

# ★ 拼音模糊匹配的最小候选长度（字符数）。短于此值只做字面匹配。
_PY_MIN = 2

_TAIL = r"[。．.！!？?，,、；;：: \t]*$"      # 允许 1 个语气尾巴

# ★ 引用类说法：「第3个error的pod」/「最后一个」/「倒数第二个」
#
#   ★★ 设计取舍（用户决定）：**不做指代消解**，直接把原话交给 AI。
#      理由：指代消解要把「序数 + 状态过滤 + 表格解析 + 上一条输出」全都串起来，
#      **意图太复杂、传递的东西太多**，而且解析出来的还是**可能已过期的缓存**。
#      改成让 AI **自己重新跑命令**拿最新状态 —— 数据更新，链路更短。
#
#   ⚠️ 必须在候选表【之前】判：否则「...的 pod」会命中 items 别名「pod」，
#      变成又跑一遍 kubectl get pods（那既不是用户要的详情，也不是 AI 的分析）。
_REF_PAT = re.compile(
    r"第\s*[0-9]+\s*[个条行号台]"
    r"|第\s*[一二两三四五六七八九十]+\s*[个条行号台]"
    r"|最后\s*(一)?\s*[个条行]"
    r"|倒数\s*第?\s*[0-9一二两三四五六七八九十]+\s*[个条行]"
    r"|[0-9]+\s*号"
)


# ══════════════════════════════════════════════════════════════════════
#  拼音键
# ══════════════════════════════════════════════════════════════════════
_PY_CACHE: dict = {}
_HAS_PY = None


def _has_pypinyin() -> bool:
    global _HAS_PY
    if _HAS_PY is None:
        try:
            from pypinyin import lazy_pinyin  # noqa: F401
            _HAS_PY = True
        except Exception:
            _HAS_PY = False
    return _HAS_PY


def _py_ch(ch: str) -> str:
    """单字拼音（无调、小写），带缓存。没装 pypinyin 就返回原字。"""
    v = _PY_CACHE.get(ch)
    if v is None:
        if _has_pypinyin():
            try:
                from pypinyin import lazy_pinyin
                v = "".join(lazy_pinyin(ch)).lower()
            except Exception:
                v = ch.lower()
        else:
            v = ch.lower()
        _PY_CACHE[ch] = v
    return v


def _py_key(s: str) -> str:
    """整串拼音键：汉字→拼音，ASCII 字母数字→小写，其它（空格/标点）忽略。

    这样「查 pod」和「查pod」得到同一个键，ASCII 大小写也统一。
    """
    out = []
    for c in (s or ""):
        if "\u4e00" <= c <= "\u9fff":
            out.append(_py_ch(c))
        elif c.isascii() and c.isalnum():
            out.append(c.lower())
    return "".join(out)


def _find(text: str, cand: str) -> int:
    """在 text 里找 cand：**先字面，再拼音滑窗**。返回起始【字符】下标，找不到 -1。

    ★★ 候选短于 _PY_MIN 个字时【不做】拼音匹配 —— 单字拼音太短，撞车概率高。
       例：「查」不做拼音模糊，否则「茶/差/察/叉…」全会命中。
    """
    if not text or not cand:
        return -1
    # ① 字面
    i = text.find(cand)
    if i >= 0:
        return i
    # ASCII 大小写不敏感的字面
    i = text.lower().find(cand.lower())
    if i >= 0:
        return i
    # ② 拼音滑窗（★ 长度门槛）
    n = len(cand)
    if n < _PY_MIN:
        return -1
    kc = _py_key(cand)
    if not kc:
        return -1
    for j in range(0, max(0, len(text) - n + 1)):
        if _py_key(text[j:j + n]) == kc:
            return j
    return -1


def normalize(text: str) -> str:
    """★ 保留此函数仅为 API 兼容（aoe4/openra 有同名函数）。

    运维域**不做**逐字替换 —— 见模块头部的血债说明。
    这里只去掉首尾空白。
    """
    return (text or "").strip()


# ══════════════════════════════════════════════════════════════════════
#  Profile
# ══════════════════════════════════════════════════════════════════════
def _variants_of(item: dict) -> list[str]:
    """取一个 items 条目的全部中文/英文别名 + 名字。"""
    out = [item.get("name", ""), item.get("en", "")]
    out += list(item.get("aliases") or [])
    return [x for x in out if x]


class OpsProfile:
    """运维 profile：短语表 + 候选表（闭集 + items 别名）+ 三档路由。"""

    def __init__(self, raw: dict):
        self.raw = raw
        self.profile_id = raw.get("profile_id", DEFAULT_PROFILE)
        self.intents = raw.get("intents", [])
        self.active_ids = {i["id"] for i in self.intents}
        self._hotwords = raw.get("hotwords", [])
        self._items = raw.get("items", [])
        self._phrases = raw.get("phrases") or {}
        self._closed = [str(x) for x in (raw.get("closed_set") or []) if str(x).strip()]
        # ★ 最近一次 parse() 命中哪一层 —— 给 handle() 打日志用。
        #   ⚠️ 用户实测问过「到底走的哪一层」（短语表 / 闭集 / items 别名 / AI），
        #      日志原来只说「规则命中 ops_metric」，分不出来。
        self.last_layer = ""
        self._safety = raw.get("safety") or {}
        self._ai = raw.get("ai_escalation") or {}
        self._triggers = raw.get("triggers") or {}

        # 候选表 = 闭集 + items 别名（长词优先 —— 否则「查所有pod」会被「查」先匹配掉）
        cands = list(self._closed)
        for it in self._items:
            cands += _variants_of(it)
        cands = [c for c in dict.fromkeys(cands) if c]
        self._cands = sorted(cands, key=len, reverse=True)

        # items 别名 → 条目（小写索引）
        self._alias2item = {}
        for it in self._items:
            for a in _variants_of(it):
                self._alias2item.setdefault(a.lower(), it)

    # ── 只读接口 ──
    def hotwords(self) -> list[str]:
        return list(self._hotwords)

    def items(self) -> list[dict]:
        return list(self._items)

    def triggers_for(self, intent_id: str) -> list[str]:
        return list(self._triggers.get(intent_id) or [])

    def risk_of(self, intent_id: str) -> str:
        for i in self.intents:
            if i["id"] == intent_id:
                return i.get("risk", "read")
        return "read"

    def is_blacklisted(self, text: str) -> bool:
        """危险动词检查 —— 命中就绝不执行。"""
        t = (text or "").lower()
        return any(b.lower() in t for b in (self._safety.get("blacklist") or []))

    def needs_confirm(self, intent_id: str) -> bool:
        return intent_id in (self._safety.get("confirm_required") or [])

    # ── ③ AI 升级判定（分强弱）──
    def ai_strong(self, text: str) -> bool:
        """★ 显式求助（优先级高于规则层）。"""
        t = (text or "")
        return any(k in t for k in (self._ai.get("strong_triggers") or []))

    def ai_weak(self, text: str) -> bool:
        """弱信号（排在规则层和 embedding 之后）。"""
        t = (text or "")
        return any(k in t for k in (self._ai.get("weak_triggers") or []))

    def ai_triggers(self) -> list[str]:
        return list(self._ai.get("strong_triggers") or []) + list(self._ai.get("weak_triggers") or [])

    # ── ① 规则层 ──
    @staticmethod
    def _spec(spec) -> tuple[str, str]:
        """短语表的值 '动作[:参数[:子参数]]' → (动作, 参数)"""
        a, _, arg = str(spec).partition(":")
        return a, arg

    def _resolve_cand(self, cand: str):
        """候选词 → 条目，再映射到意图 id。找不到返回 None。"""
        it = self._alias2item.get(cand.lower())
        if not it:
            return None
        iid = it["id"]
        for c in (f"ops_{iid}", f"ops_k8s_{iid}", f"ops_restart_{iid}", f"ops_stop_{iid}"):
            if c in self.active_ids:
                return c
        return None

    def parse(self, text: str):
        """规则直通：短语表 > 候选表。每层都「先字面，再拼音」。

        返回 (intent, arg) 或 None（拒识）。
        ★ 宁可返回 None，也不猜。
        """
        if not text:
            return None
        raw = text.strip()
        if not raw:
            return None
        if self.is_blacklisted(raw):
            self.last_layer = "黑名单拒识"
            return None                       # ★ 危险动词直接拒识
        t = re.sub(_TAIL, "", raw)
        self.last_layer = ""                  # 每次 parse 重置

        # ⚠️ 集成说明：本函数被 actions._game_parse() 调用，是【规则层】的唯一入口。
        #    三档路由里的②embedding 由 handle() 的 _predict_intent 负责（parse 返回
        #    unknown 时才轮到它）；所以这里只放①规则 和 ③AI 触发 —— 顺序如下。

        # ①a ★ strong AI 触发（显式求助）—— 优先级最高
        #     「让 ds 看看这个 pod 怎么挂了」必须走 AI，不能被候选表的 describe 抢走
        if self.ai_strong(t) and "ops_ask_ai" in self.active_ids:
            self.last_layer = "AI触发-强（显式求助）"
            return "ops_ask_ai", t

        # ①b ★ 引用类说法 → 也交给 AI（不做指代消解，让它自己跑命令拿最新数据）
        if _REF_PAT.search(t) and "ops_ask_ai" in self.active_ids:
            self.last_layer = "AI触发-引用类"
            return "ops_ask_ai", t

        # ② 短语表 —— 字面精确优先（确定性最高）
        for key in (t, raw):
            if key in self._phrases:
                self.last_layer = f"短语表-精确「{key}」"
                return self._spec(self._phrases[key])
        # ①b 短语表 —— 拼音滑窗（同音/繁体兜底；长词优先）
        for key in sorted(self._phrases, key=len, reverse=True):
            if _find(t, key) >= 0:
                self.last_layer = f"短语表-滑窗「{key}」"
                return self._spec(self._phrases[key])

        # ③ 候选表（闭集 + items 别名）—— 长词优先；_find 内部先字面再拼音
        for cand in self._cands:
            if _find(t, cand) >= 0:
                iid = self._resolve_cand(cand)
                if iid:
                    # ⚠️ 用户实测问过「到底走的哪一层」—— 日志原来只说「规则命中」，
                    #    分不清是【短语表】还是【闭集候选表】。这里记下来给 handle 打。
                    _is_closed = cand in self._closed
                    self.last_layer = (f"{'闭集' if _is_closed else 'items别名'}「{cand}」")
                    return iid, ""
                # 候选命中了但没有对应条目（比如闭集里的整句短语）→ 继续试更短的

        # ④ ★ weak AI 触发 —— 排在规则层之后
        #    「这个 pod 怎么挂了」应该先给规则层 describe（Events 段直接给答案，
        #    比 AI 快且确定），规则答不了才升级
        if self.ai_weak(t) and "ops_ask_ai" in self.active_ids:
            self.last_layer = "AI触发-弱（规则答不了）"
            return "ops_ask_ai", t
        self.last_layer = self.last_layer or "规则层未命中（转 embedding）"
        return None


def load(profile_id: str | None = None) -> OpsProfile:
    """加载 profile：profile.json（闭集/短语/items）+ hotwords.txt（热词）。"""
    pid = profile_id or DEFAULT_PROFILE
    d = PROFILES / pid
    raw: dict = {"profile_id": pid, "intents": [], "hotwords": []}
    p = d / "profile.json"
    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[ops] profile.json 解析失败：{type(e).__name__}: {e}", flush=True)
    hw = d / "hotwords.txt"
    if hw.exists():
        try:
            raw["hotwords"] = [w.strip() for w in hw.read_text(encoding="utf-8").splitlines() if w.strip()]
        except Exception:
            pass
    return OpsProfile(raw)


def route(text: str, profile: OpsProfile, embed_cls=None):
    """三档路由 → (intent, arg, tier)；拒识返回 (None, "", None)。

    tier: 'rule' | 'embed' | 'ai' | None

    ⚠️ 集成说明：真实链路上 actions.parse() 调的是 profile.parse()（不是本函数）——
       parse() 内部已经把①规则 / ①a strong AI / ①b 引用类 / ④weak AI 按优先级排好，
       ②embedding 由 handle() 的 _predict_intent 负责。
       本函数是**同一套逻辑的独立入口**（供单元测试 / 直接调用），
       所以 tier 的判定方式改成「按 parse() 返回的意图反推」，与 parse() 保持一致。

    优先级：
        0. 危险动词        → 直接拒识
        1. 规则层（含 strong AI 触发 / 引用类）  → rule 或 ai
        2. embedding kNN   → embed
        3. 拒识（weak AI 触发已在 parse 内处理）
    """
    if not text:
        return None, "", None

    # 0. ★ 危险动词优先拦截
    if profile.is_blacklisted(text):
        return None, "", None

    # 1. 规则层 —— parse() 内部已按优先级排好（含 strong AI / 引用类 / weak AI）
    r = profile.parse(text)
    if r:
        ai_intent = profile._ai.get("intent", "ops_ask_ai")
        return r[0], r[1], ("ai" if r[0] == ai_intent else "rule")

    # 2. embedding kNN
    if embed_cls is not None:
        try:
            res = embed_cls.predict(text)
            if res and res.get("intent") and res["intent"] != "unknown":
                return res["intent"], text, "embed"
        except Exception:
            pass        # ★ 失败不抛，继续往下走（绝不在这里自动升级 AI）

    # 3. 拒识
    return None, "", None
