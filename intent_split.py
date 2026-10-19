# -*- coding: utf-8 -*-
"""文本域切分裁决器 —— 整段版与流式版共用同一个（见 STREAMING_INTENT_PLAN.md §〇之二）

认定方案：
  · 音频域只负责"尽量早地给候选边界"（停顿），**不做裁决**
  · 文本域是唯一裁决者：**关键词（触发词）定边界**，停顿只做确认
  · 正确性不依赖停顿阈值 —— 所以输入可以是
        (a) 一整段文本（松手后整段 ASR）        → push(text); flush()
        (b) 音频切出来的碎片流（分片/流式 ASR）  → push(frag, pause=True/False) 多次; flush()
    两者必须产出**同一条指令序列**（回归里用"任意切碎再喂"验证）

三个机制（就靠这三条把"阈值"变成性能参数）：
  ① 触发词 = 强制切点：段内已有触发词时，再遇到新触发词就在**新触发词处**切
  ② 缺触发词的碎片 → 向前并回上一段（_repair）
  ③ 停顿只做确认：`pause=True` 且当前段**看着完整**时，才多给一个切点
     —— "看着完整"由调用方注入（默认用 actions 的解析结果），所以阈值调高调低
        都不会弄丢字：多切了会被 ② 并回去

刻意不做的事：
  · 不判意图（下游照旧走 _predict_intent + 规则 + 门控 + 实体链接，一行不改）
  · 不执行任何动作
  · 不做自适应（只暴露 n_frag / n_frag_no_trigger / n_cut_* 计数给第二阶段调参）
"""
from dataclasses import dataclass


@dataclass
class Seg:
    text: str                 # 片段文本（已 strip 两端空白与首尾标点外的空白）
    start: int                # 在输入流里的字符起点（审计/对账用）
    end: int
    cut_by: str               # 'trigger' | 'pause' | 'flush'
    trigger: str = ""         # 本段命中的触发词（空 = 没有触发词）
    merged: bool = False      # 是否由 _repair 并回来的

    @property
    def has_trigger(self) -> bool:
        return bool(self.trigger)


def triggers_from_cfg(cfg_json: dict) -> list:
    """从 intent_config.json 直接取【全部】触发词（解析用；回归可离线跑）。

    ⚠️ 生产侧权威来源是 actions.Config.triggers；本函数只是**同一份 JSON 的另一种读法**。
       回归会拿两者对比，发现漂移就报错。
    """
    out = set()
    for v in (cfg_json.get("triggers") or {}).values():
        if isinstance(v, (list, tuple)):
            out.update(str(x) for x in v if x)
    b = cfg_json.get("behavior") or {}
    for k in ("search_mid_verbs",):
        out.update(str(x) for x in (b.get(k) or []) if x)
    # 长词优先（"搜索一下" 要盖住 "搜索"）
    return sorted(out, key=len, reverse=True)


# ⚠️⚠️ 回归实测抓出来的：触发词里有一批【不能当边界】的
#    · url_suffix（网页/网站/页面/官网/网址）、file_suffix（文件/文档）= 限定词：
#      「打开豆包网页」里的「网页」不是新命令，切了就把一句话劈成两段（实测 450 条里 7% 被误切）
#    · 「把/看/看看/上/去/进入/来一下/开个」= 单字或过泛的动词：
#      「把这个Clash…」的「把」、「和打开网址之类的事情」的「去」都会误触发
#    ❗关键：这些词**只是不当边界**，仍然参与【段内解析】——
#      下游 parse() 拿到的是整段文本，用的是全部触发词，所以「把X打开」「上知乎」这些说法照样能解。
BOUNDARY_EXCLUDE_DEFAULT = (
    "网页", "网站", "页面", "官网", "网址",          # url_suffix
    "文件", "文档",                                  # file_suffix
    "把", "看", "看看", "上", "去", "进入", "来一下", "开个",   # 单字/过泛
)


def boundary_from_cfg(cfg_json: dict, extra_exclude=()) -> list:
    """能当【边界】的触发词 = 全部触发词 − 限定词/过泛词（可用配置覆盖）"""
    b = cfg_json.get("behavior") or {}
    ex = set(BOUNDARY_EXCLUDE_DEFAULT) | set(b.get("split_boundary_exclude") or []) \
        | set(extra_exclude)
    return [t for t in triggers_from_cfg(cfg_json) if t not in ex]


def names_from_cfg(cfg_json: dict) -> list:
    """闭集名 + 别名（用于"触发词紧跟在一个名字后面 = 它是限定词"这条护栏）"""
    out = set()
    for key in ("apps", "urls", "files"):
        for e in (cfg_json.get(key) or []):
            if e.get("name"):
                out.add(str(e["name"]))
            for a in (e.get("aliases") or []):
                out.add(str(a))
    return sorted((n for n in out if n), key=len, reverse=True)


class Splitter:
    def __init__(self, triggers, boundary=None, names=(), pause_cut=False,
                 completeness=None, max_segments=8, min_arg_chars=1):
        # triggers：全部触发词（只在 completeness/归因里用）
        self.triggers = [t for t in sorted(set(triggers), key=len, reverse=True) if t]
        # boundary：能当边界的子集（默认=全部，向后兼容；生产用 boundary_from_cfg）
        self.boundary = [t for t in (sorted(set(boundary), key=len, reverse=True)
                                     if boundary is not None else self.triggers) if t]
        self.names = [n for n in names if n]
        self.pause_cut = bool(pause_cut)
        self.completeness = completeness if completeness is not None else (lambda s: True)
        self.max_segments = int(max_segments)      # 护栏：切太碎说明规则出问题
        self.min_arg_chars = int(min_arg_chars)    # 被切出去那段，触发词后至少要有这么多字
        self.buf = ""
        self.base = 0
        # 统计（第二阶段自适应用；现在只记账不调参）
        self.n_frag = 0
        self.n_frag_no_trigger = 0
        self.n_cut_trigger = 0
        self.n_cut_pause = 0
        self.n_merged = 0
        self.n_blocked_suffix = 0      # 因"跟在名字后面"被拦下的切点
        self.n_blocked_half = 0        # 因"前半句没有参数"被拦下的切点

    # ── 触发词扫描（不重叠、长优先）──
    def _scan(self, s, use_boundary=True):
        pool = self.boundary if use_boundary else self.triggers
        out, i, n = [], 0, len(s)
        while i < n:
            for t in pool:
                if s.startswith(t, i):
                    out.append((i, t))
                    i += len(t)
                    break
            else:
                i += 1
        return out

    # ── 护栏：触发词紧跟在闭集名后面 → 它是限定词（"…豆包|网页"），不是新命令 ──
    def _after_name(self, pos):
        head = self.buf[:pos]
        for nm in self.names:
            if head.endswith(nm):
                return nm
        return None

    # ── 把"没有触发词的段"并回上一段（机制②）──
    def _repair(self, segs):
        out = []
        for s in segs:
            if not s.trigger and out:
                prev = out[-1]
                prev.text = (prev.text + s.text).strip()
                prev.end = s.end
                prev.merged = True
                self.n_merged += 1
                continue
            out.append(s)
        # 开头就没有触发词的段：往后并（它只能是噪声或半句）
        if out and not out[0].trigger and len(out) > 1:
            out[1].text = (out[0].text + out[1].text).strip()
            out[1].start = out[0].start
            out[1].merged = True
            self.n_merged += 1
            out.pop(0)
        return out

    def push(self, text, pause=False):
        """喂一段文本（整句或碎片）。返回本次能**确定**的段（可能为空）。"""
        if text:
            self.buf += text
            self.n_frag += 1
            if not self._scan(text):
                self.n_frag_no_trigger += 1

        segs, hits = [], self._scan(self.buf)
        seg_start, last_trig_end = 0, -1
        for pos, trig in hits:
            if pos > seg_start and last_trig_end >= 0:
                front = self.buf[seg_start:pos]
                # 护栏 A（半句）：被切出去那段，触发词后面必须还有参数
                #   「和打开|网址之类的事情」→ 前半段是「和打开」，触发词后 0 字 → 不切
                if len(front[max(0, last_trig_end - seg_start):].strip()) < self.min_arg_chars:
                    self.n_blocked_half += 1
                    continue
                # 护栏 B（限定词）：触发词紧跟在一个闭集名后面 → 它是后缀，不是新命令
                #   「打开豆包|网页」→ 网页 前面是名字 豆包 → 不切
                if self._after_name(pos):
                    self.n_blocked_suffix += 1
                    continue
                segs.append(Seg(front.strip(), self.base + seg_start,
                                self.base + pos, "trigger", trig))
                self.n_cut_trigger += 1
                seg_start, last_trig_end = pos, pos + len(trig)
            else:
                last_trig_end = max(last_trig_end, pos + len(trig))

        # 机制③：停顿 + 当前段看着完整 → 允许在末尾多切一刀
        tail = self.buf[seg_start:]
        if self.pause_cut and pause and tail.strip() and last_trig_end >= 0 \
                and self.completeness(tail.strip()):
            trig = next((t for _, t in self._scan(tail)), "")
            segs.append(Seg(tail.strip(), self.base + seg_start,
                            self.base + len(self.buf), "pause", trig))
            self.n_cut_pause += 1
            seg_start = len(self.buf)

        if seg_start:
            self.base += seg_start
            self.buf = self.buf[seg_start:]

        out = self._repair(segs)
        if len(out) >= self.max_segments:      # 护栏：异常文本不要炸出几十段
            out = out[:self.max_segments - 1] + [
                Seg("".join(s.text for s in out[self.max_segments - 1:]),
                    out[self.max_segments - 1].start, out[-1].end, "trigger",
                    out[self.max_segments - 1].trigger)]
        return out

    def flush(self):
        """收尾：把缓冲里剩下的吐出来"""
        if not self.buf.strip():
            self.buf, self.base = "", self.base + len(self.buf)
            return []
        trig = next((t for _, t in self._scan(self.buf)), "")
        s = Seg(self.buf.strip(), self.base, self.base + len(self.buf), "flush", trig)
        self.buf, self.base = "", self.base + len(self.buf)
        return self._repair([s])

    def segments(self, text, **kw):
        """便捷入口：整段文本 → 指令序列"""
        return self.push(text, **kw) + self.flush()


def strict_shape(seg_text, boundary, names, slack=5, lead=1):
    """段是否"形状良好"= 真像一条指令（而不是"长句里恰好含了个动词+名字"）。

    判据（真机实测逼出来的）：
      ① 触发词必须在【段首 ≤lead 字】内 —— 「把这个Clash分类到可以被语音打开的应用里面」
         里的「打开」在中间 → 形状不良（这条正是 33 字非命令句的防线的关键）
      ② 触发词后要跟一个闭集名/别名（参数）；无参指令（清空输入/开始输入…）则要求
         触发词之后几乎没有别的字
      ③ 参数之后的剩余（去掉 吧/。/，/然后 这类尾巴）≤ slack —— 防止"参数后面还拖一大段"

    返回 (ok, trigger, name, leftover)
    """
    t = (seg_text or "").strip()
    if not t:
        return False, "", "", -1
    # ⚠️ 必须大小写不敏感：ASR 出的是「打开Deepseek网页版」（小写 s），
    #    配置里是「DeepSeek」——大小写敏感会让形状检查把合法的 5 段判成"不良"
    tl = t.lower()
    hit = None
    for tr in boundary:                      # boundary 已按长度降序
        p = tl.find(tr.lower())
        if 0 <= p <= lead and (hit is None or len(tr) > len(hit[1])):
            hit = (p, tr)
    if hit is None:
        return False, "", "", -1
    p, tr = hit
    rest = t[p + len(tr):]
    restl = rest.lower()
    nm = ""
    npos = -1
    for cand in names:                       # names 已按长度降序
        q = restl.find(cand.lower())
        if q >= 0:
            nm, npos = cand, q
            break
    tail = rest[npos + len(nm):] if nm else rest
    for junk in ("然后", "，然后", "吧", "。", "，", ",", ".", "！", "？", "!", "?", " ",
                 "、", "；", ";", "：", ":"):
        tail = tail.replace(junk, "")
    return len(tail.strip()) <= slack, tr, nm, len(tail.strip())


def split_commands(text, triggers, boundary=None, names=(), pause_cut=False,
                   completeness=None):
    return Splitter(triggers, boundary=boundary, names=names, pause_cut=pause_cut,
                    completeness=completeness).segments(text)
