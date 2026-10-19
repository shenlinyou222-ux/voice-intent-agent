# -*- coding: utf-8 -*-
"""版本化 profile 的运行时：加载、热词、别名解析、规则直通分句。

设计（对应用户要求「新的热词、权重等要使用不同的版本加以控制与区分」）：
  · profile.json 自带三套版本戳：
        hotwords_version —— 热词表内容哈希（喂给 ASR 的 system 提示 + 转写纠错候选）
        intent_version   —— 意图语料哈希
        index_version    —— embedding 索引版本（build_profile_index.py 回填）
  · 换域/换版本 = 换一个 profiles/<id>/ 目录，旧版本原样留着，可随时 A/B、可回滚
  · 原桌面意图（打开程序/网址/搜索）不在游戏 profile 里 —— 按用户要求完全丢掉
"""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent          # openra/
PROFILES = HERE / "profiles"
DEFAULT_PROFILE = "openra-v2"


def _pinyin(s: str) -> str:
    """汉字→拼音（复用 actions 里那套，装了 pypinyin 就有）。"""
    try:
        from actions import _pinyin as _p
        return _p(s) or ""
    except Exception:
        return ""

# 中文数字（个位；十/百 在 _cn_to_int 里按组合处理）
CN_DIGIT = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}
CN_NUM = {**CN_DIGIT, "十": 10}

# 「整句基本就是单位名」判定用的填充字：剩下的字都在这里 → 当"造它"
ITEM_FILLER = set("个两二三四五六七八九十百辆台座艘架名的了我要来买给帮我造建生产下这把先然后再")
# 编队相关的说法
GROUP_TRIGGERS = ("编队", "编为一队", "编成", "编入", "组队", "编一队", "编个队", "编为", "编到")
EXPLORE_TRIGGERS = ("探索", "侦察", "侦查", "探路", "去探", "自动探索", "开图")


class ProfileError(RuntimeError):
    pass


class GameProfile:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.id = self.raw["profile_id"]
        self.version = self.raw.get("version", 1)
        self.hotwords_version = self.raw.get("hotwords_version")
        self.intent_version = self.raw.get("intent_version")
        self.index_version = self.raw.get("index_version")
        self.bridge = self.raw["bridge"]
        self.game = self.raw.get("game", {})
        self.triggers = self.raw.get("triggers", {})
        self.intents = self.raw.get("intents", [])
        # ⭐ 本 profile【激活】的意图集合：只有这些会被规则直通执行。
        #    「收缩意图、一点点加」靠它实现：不在集合里的意图直接落到模型/拒识，
        #    免得一堆动作词互相抢语义（实测「维修工厂」被「维修」抢成修车）。
        self.active_ids = {i["id"] for i in self.intents}
        self.items = self.raw.get("items", [])
        self.queues = self.raw.get("queues", {})
        self.stances = self.raw.get("stances", {})

        # 别名 → actor（长的优先，避免「坦克」抢走「重型坦克」）
        self._alias: list[tuple[str, dict]] = []
        self._alias_py = None
        for item in self.items:
            for alias in item.get("aliases", []):
                self._alias.append((alias.lower(), item))
        self._alias.sort(key=lambda kv: -len(kv[0]))

        # 队列词 → filter（「所有坦克」→ 整条 Vehicle 生产线）
        self._queue_words: list[tuple[str, str]] = []
        for queue, words in self.queues.items():
            for w in words:
                self._queue_words.append((w.lower(), f"queue:{queue}"))
        self._queue_words.sort(key=lambda kv: -len(kv[0]))

        self._stance_words: list[tuple[str, str]] = []
        for stance, words in self.stances.items():
            for w in words:
                self._stance_words.append((w.lower(), stance))
        self._stance_words.sort(key=lambda kv: -len(kv[0]))

        self._triggers: list[tuple[str, str]] = []
        for intent, words in self.triggers.items():
            for w in words:
                self._triggers.append((w.lower(), intent))
        self._triggers.sort(key=lambda kv: -len(kv[0]))

    # ── 版本 ────────────────────────────────────────────────
    def version_report(self) -> dict:
        return {
            "profile_id": self.id,
            "profile_version": self.version,
            "hotwords_version": self.hotwords_version,
            "intent_version": self.intent_version,
            "index_version": self.index_version,
            "index_samples": self.raw.get("index_samples"),
            "hotwords_count": self.raw.get("hotwords_count"),
        }

    # ── 热词（ASR 用；版本化）───────────────────────────────
    def hotwords(self) -> list[str]:
        f = self.path.parent / self.raw.get("hotwords_file", "hotwords.txt")
        if not f.exists():
            return []
        return [w.strip() for w in f.read_text(encoding="utf-8").splitlines() if w.strip()]

    def asr_hotwords(self, max_chars: int = 320, ascii_only: bool = False) -> str:
        """拼成 ASR 的 system 提示词（Qwen3-ASR 的 ns.sys 热词上下文）。

        与 voice_agent 现有 asr_hotwords 行为一致：超长就截断。
        中文热词对 Qwen3-ASR 也有效（实测「新建会话」那种同音误转就是靠这个修的）。
        """
        words = []
        for w in self.hotwords():
            if ascii_only and not w.isascii():
                continue
            words.append(w)
        out = "、".join(words)
        return out[:max_chars]

    # ── 别名解析 ────────────────────────────────────────────
    def _alias_pinyin(self):
        """别名的拼音表（懒加载，只算一次）。"""
        if self._alias_py is None:
            self._alias_py = []
            for alias, item in self._alias:
                py = _pinyin(alias)
                if py and len(py) >= 3:
                    self._alias_py.append((py, alias, item))
        return self._alias_py

    def item(self, text: str) -> dict | None:
        """「矿车」/「重型坦克」/「medium tank」/「2tnk」→ item 记录。

        三级匹配：精确 → 子串 → **拼音近似**（ASR 同音错字兜底）。
        """
        if not text:
            return None
        t = text.strip().lower()
        for alias, item in self._alias:
            if t == alias:
                return item
        for alias, item in self._alias:
            if alias and alias in t:
                return item
        for alias, item in self._alias:
            if alias and t and t in alias:
                return item

        # ⭐ 拼音兜底：ASR 对专有名词的错法是整词替换成同音字，
        #    编辑距离救不回来，但拼音只差一两个字母：
        #    实测「精炼厂」(jinglianchang) 被转成「金炼厂」(jinlianchang)。
        py = _pinyin(t)
        if py and len(py) >= 4:
            best, best_score = None, 0.0
            for apy, _alias, item in self._alias_pinyin():
                s = difflib.SequenceMatcher(None, py, apy).ratio()
                if s > best_score:
                    best, best_score = item, s
            if best is not None and best_score >= 0.82:
                return best
        return None

    def filter_for(self, text: str, default: str = "all") -> str:
        """「所有坦克」→ queue:Vehicle；「重型坦克」→ type:3TNK；否则 default。"""
        if not text:
            return default
        t = text.strip().lower()
        for word, filt in self._queue_words:
            if word in t:
                return filt
        item = self.item(t)
        if item:
            return f"type:{item['id']}"
        if any(k in t for k in ("全部", "所有", "全选", "一切")):
            return "all"
        if "空闲" in t or "闲置" in t:
            return "idle"
        return default

    def stance(self, text: str) -> str | None:
        if not text:
            return None
        t = text.strip().lower()
        for word, stance in self._stance_words:
            if word in t:
                return stance
        for stance in self.stances:
            if stance.lower() in t:
                return stance
        return None

    # ── 参数解析 ────────────────────────────────────────────
    def _item_only(self, t: str, item: dict) -> bool:
        """整句基本就是"一个单位/建筑名"（剩下的字都是量词/动词填充）→ 当"造它"。

        ⚠️ 血债：说「维修工厂」被动作词「维修」抢走，判成 game_repair（修车），
           而不是造维修厂（FIX）。所以这条判定要排在"无参动作词"之前。
        「修理电厂」不会被误判：剩下的「修理」不在填充字里 → 仍走修理。
        """
        best = None
        for alias in item.get("aliases", []):
            if alias and alias in t and (best is None or len(alias) > len(best)):
                best = alias
        if not best:
            return False
        rest = t.replace(best, "", 1)
        return all(ch in ITEM_FILLER for ch in rest)

    def group_spec(self, text: str) -> dict | None:
        """解析语音编队 → {n, mode, parts:[{filter,count}], filter}

        「三辆坦克二步兵编为一队」→ n=1, parts=[type:2tnk×3, type:e1×2]
        「全部坦克编为一队」      → n=1, parts=[type:2tnk×0]（0 = 全要）
        「所有矿车编为二队」      → n=2, parts=[type:harv×0]
        """
        if not text:
            return None
        t = text.strip()
        if not any(k in t for k in GROUP_TRIGGERS):
            return None

        # 队号：1队 / 一队 / 第一队 / 编为3队
        n = 1
        m = re.search(r"([0-9]+)\s*队", t) or re.search(r"第?\s*([一二两三四五六七八九])\s*队", t)
        if m:
            g = m.group(1)
            n = int(g) if g.isdigit() else CN_DIGIT.get(g, 1)
        n = max(1, min(9, n))

        mode = "add" if any(k in t for k in ("加入", "编入", "补充", "加进")) else "set"

        parts = []
        for item in self.items:
            for alias in sorted(item.get("aliases") or [], key=len, reverse=True):
                idx = t.find(alias)
                if idx < 0:
                    continue
                before = t[max(0, idx - 4):idx]
                cnt = self.count(before, 0) if before else 0
                parts.append({"filter": f"type:{item['id']}", "count": cnt, "name": item["name"]})
                break

        if not parts:
            # 没说具体兵种：「编为一队」= 把当前选中的编进去
            return {"n": n, "mode": mode, "parts": [], "filter": None}

        return {"n": n, "mode": mode, "parts": parts, "filter": None}

    @staticmethod
    def _group_arg(spec: dict) -> str:
        """编队参数用 JSON 串传（管线里参数是字符串）。"""
        return json.dumps(spec, ensure_ascii=False)

    @staticmethod
    def _cn_to_int(s: str) -> int | None:
        """中文数字串 → 整数：四十=40 三十=30 十五=15 二十三=23 一百=100。"""
        nums = GameProfile._cn_numbers(s)
        return nums[0] if len(nums) == 1 else None

    @staticmethod
    def _cn_numbers(s: str) -> list[int]:
        """中文数字串 → 整数列表（支持多个数连写）。

        ⚠️ 实测：说「移动到六十 四十」，ASR 会把停顿吃掉变成「六十四十」。
           所以必须能把一串中文数字切成 [60, 40]，否则坐标永远解析不出来。
        规则：`[数字]?十/百` 各成一个数；紧跟其后的个位数字并入（十五=15），
              但下一个又是 `数字+十/百` 时就是新数（六十四十 → 60,40）。
        """
        tokens: list[tuple[str, int]] = []
        i, n = 0, len(s)
        while i < n:
            ch = s[i]
            if ch in CN_DIGIT and i + 1 < n and s[i + 1] in "十百":
                tokens.append(("unit", CN_DIGIT[ch] * (10 if s[i + 1] == "十" else 100)))
                i += 2
            elif ch == "十":
                tokens.append(("unit", 10))
                i += 1
            elif ch == "百":
                tokens.append(("unit", 100))
                i += 1
            elif ch in CN_DIGIT:
                tokens.append(("digit", CN_DIGIT[ch]))
                i += 1
            else:
                i += 1

        nums: list[int] = []
        last_unit = False
        for kind, value in tokens:
            if kind == "digit" and last_unit and nums:
                nums[-1] += value
                last_unit = False
            else:
                nums.append(value)
                last_unit = kind == "unit"
        return nums

    @classmethod
    def numbers(cls, text: str) -> list[int]:
        """抽出文本里的数字：阿拉伯数字 + 中文数字（含 十/百 组合与连写切分）。

        ⚠️ 早先是逐字映射（四十三 → 4,10,3），说「移动到六十 四十」会解析成 (6,10)。
           语音报坐标几乎必然用中文数字，所以必须按 十/百 组合解析。
        """
        out: list[int] = []
        for m in re.finditer(r"\d+", text):
            out.append(int(m.group()))
        for m in re.finditer(r"[零一二两三四五六七八九十百]+", text):
            out += cls._cn_numbers(m.group())
        return out

    @classmethod
    def coords(cls, text: str, coord_max: int = 128) -> tuple[int, int] | None:
        """「移动到 60 40」→ (60,40)。只认「两个数字」，且必须在合理范围内。"""
        nums = cls.numbers(text)
        if len(nums) < 2:
            return None
        x, y = nums[0], nums[1]
        if not (0 <= x <= coord_max and 0 <= y <= coord_max):
            return None
        return x, y

    @classmethod
    def count(cls, text: str, default: int = 1) -> int:
        """「造两个矿车」→ 2。"""
        nums = cls.numbers(text)
        for n in nums:
            if 1 <= n <= 50:
                return n
        return default

    # ── 规则直通分句（不依赖模型，先命中先执行）────────────
    def parse(self, text: str) -> tuple[str, str] | None:
        """返回 (intent, arg)。命中不了返回 None，交给 embedding kNN。"""
        if not text:
            return None
        t = text.strip()

        # 0) ⭐ 编队 / 探索：必须最先判（「三辆坦克二步兵编为一队」里的 坦克/步兵
        #    会被后面的生产分支抢走）
        if any(k in t for k in GROUP_TRIGGERS):
            spec = self.group_spec(t)
            if spec:
                return "game_group", self._group_arg(spec)
        if any(k in t for k in EXPLORE_TRIGGERS):
            return "game_explore", "auto"

        # 1) 姿态：必须带「姿态/开火」
        if any(k in t for k in ("姿态", "开火模式", "切换成", "改成")):
            stance = self.stance(t)
            if stance:
                return "game_stance", stance

        # 2) 放置建筑（要在「建造」之前判：'放置电厂' 不能落进 build）
        for trig in ("放置", "放在", "放下", "放到", "摆下"):
            if trig in t:
                c = self.coords(t, self.game.get("coord_max", 128))
                return "game_place", f"{c[0]},{c[1]}" if c else "auto"

        # 3) 取消生产（要在「建造」之前判，否则「取消建造坦克」会被建造抢走）
        for trig in ("取消建造", "取消生产", "不造了", "不造", "别造", "不要造", "停止生产", "取消"):
            if trig in t:
                item = self.item(t)
                return "game_cancel", item["id"] if item else ""

        # 4) 查询
        if any(k in t for k in ("战况", "什么情况", "有多少钱", "多少钱", "多少单位", "电力够")):
            return "game_state", ""

        # 5) 选中
        for trig in ("选中", "全选", "选择", "选上", "把"):
            if trig in t and any(k in t for k in ("所有", "全部", "全选", "部队", "坦克", "步兵",
                                                  "建筑", "矿车", "飞机", "船", "防御", "单位")):
                return "game_select", self.filter_for(t)

        # 6) 攻击前进（要在「攻击」「移动」之前）
        for trig in ("攻击前进", "边打边进", "进攻推进", "推进到"):
            if trig in t:
                c = self.coords(t, self.game.get("coord_max", 128))
                return "game_attackmove", f"{c[0]},{c[1]}" if c else "nearest"

        # 7) 攻击
        for trig in ("攻击", "集火", "消灭", "干掉", "打掉", "开火", "打他"):
            if trig in t:
                c = self.coords(t, self.game.get("coord_max", 128))
                return "game_attack", f"{c[0]},{c[1]}" if c else "nearest"

        # 8) 守卫
        for trig in ("守卫", "守住", "防守", "保护"):
            if trig in t:
                c = self.coords(t, self.game.get("coord_max", 128))
                return "game_guard", f"{c[0]},{c[1]}" if c else "here"

        # 9) 移动
        for trig in ("移动到", "移动", "开到", "走到", "前进到", "挪到", "调到", "去", "往"):
            if trig in t:
                c = self.coords(t, self.game.get("coord_max", 128))
                if c:
                    return "game_move", f"{c[0]},{c[1]}"

        # 10) 生产
        for trig in ("建造", "造", "生产", "来一个", "来一辆", "来两辆", "出", "造个"):
            if trig in t:
                item = self.item(t)
                if item:
                    return "game_build", f"{item['id']}x{self.count(t)}"

        # 10b) ⭐「数量 + 单位」也算生产：「两个矿车」「三辆坦克」「五艘潜艇」
        #      ⚠️ 血债：原来只认「建造/造/生产」这类动词，于是「两个矿车」既不进规则直通，
        #         又被 plausible() 的"必须含已知词"判据当成"不是命令"丢掉 —— 实测踩过。
        item = self.item(t)
        if item and (self.numbers(t) or any(v in t for v in ("来", "要", "买", "弄", "搞", "给"))):
            return "game_build", f"{item['id']}x{self.count(t)}"

        # 10c) ⭐ 整句就是一个单位/建筑名 → 造它（「电厂」「特斯拉」「维修工厂」）
        #      必须排在下面的"无参动作词"之前，否则「维修工厂」会被「维修」抢成修理。
        if item and self._item_only(t, item):
            return "game_build", f"{item['id']}x{self.count(t)}"

        # 11) 无参命令
        for trig in ("散开", "分散"):
            if trig in t:
                return "game_scatter", ""
        for trig in ("停止", "停下", "别动", "停下来", "取消动作"):
            if trig in t:
                return "game_stop", ""
        for trig in ("采矿", "挖矿", "拉矿", "采资源"):
            if trig in t:
                return "game_harvest", ""
        for trig in ("展开", "部署基地", "建立基地"):
            if trig in t:
                return "game_deploy", ""
        for trig in ("修理", "修复", "维修"):
            if trig in t:
                return "game_repair", ""
        for trig in ("卖掉", "出售", "卖了"):
            if trig in t:
                return "game_sell", ""
        for trig in ("占领", "夺取"):
            if trig in t:
                return "game_capture", ""
        for trig in ("集结点", "集合点"):
            if trig in t:
                c = self.coords(t, self.game.get("coord_max", 128))
                return "game_rally", f"{c[0]},{c[1]}" if c else "here"
        for trig in ("标记", "标个点"):
            if trig in t:
                c = self.coords(t, self.game.get("coord_max", 128))
                return "game_ping", f"{c[0]},{c[1]}" if c else "here"
        for trig in ("喊话", "发消息", "告诉队友"):
            if trig in t:
                return "game_chat", t

        # 12) ⭐ 光说一个单位/建筑名 = 造它（「电厂」「精炼厂」「两个矿车」）
        #     多指令拆分（「电厂、精炼厂」）切出来的分句就是这种形态，必须认。
        #     放在最后：前面所有分支（选中/攻击/修理/卖掉…）优先，避免抢语义。
        item = self.item(t)
        if item and len(t) <= 10:
            return "game_build", f"{item['id']}x{self.count(t)}"

        return None


_CACHE: dict[str, GameProfile] = {}


def load(profile_id: str | None = None) -> GameProfile:
    pid = profile_id or DEFAULT_PROFILE
    if pid in _CACHE:
        return _CACHE[pid]
    path = PROFILES / pid / "profile.json"
    if not path.exists():
        raise ProfileError(f"找不到 profile：{path}")
    prof = GameProfile(path)
    _CACHE[pid] = prof
    return prof


def available() -> list[str]:
    if not PROFILES.exists():
        return []
    return sorted(p.name for p in PROFILES.iterdir() if (p / "profile.json").exists())


if __name__ == "__main__":
    import sys

    pid = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROFILE
    p = load(pid)
    print(json.dumps(p.version_report(), ensure_ascii=False, indent=2))
    print("profiles:", available())
    print("hotwords[:12]:", p.hotwords()[:12])
    for probe in ["选中所有坦克", "移动到 60 40", "攻击最近的敌人", "造两个矿车",
                  "取消建造坦克", "姿态防御", "矿车去采矿", "基地车展开", "现在有多少钱"]:
        print(f"  {probe:16s} -> {p.parse(probe)}")
