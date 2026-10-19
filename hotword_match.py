# -*- coding: utf-8 -*-
"""转写后处理：闭集热词匹配

═══ 为什么要有这个模块 ═══

CTC 转写对英文专有名词很弱，典型错误是【丢掉首字母】：
    Blender    → lender      Ollama   → llama
    Steam      → team        Python   → ython
    PowerShell → owerhell    Edge浏览器 → dge浏览器
    GitHub     → itub        ModelScope → modelcope

⭐ 正确的处理位置是【转写之后】，不是去改解码器：
   音频 → CTC 解码 → 转写文本 → 【本模块：热词匹配】 → 参数

  （试过在解码时加 logit 偏置，实测有害：
    词中偏置触发不了——丢的是首字母，末尾串永远不是热词前缀；
    词首偏置无法预知下一个词是不是英文，会在每个汉字后都触发，
    往中文里注入垃圾字母。α=4 时命中率从 95% 掉到 40%。）

═══ 为什么用 LCS 比而不是编辑距离 ═══

丢首字母这个失败模式下，编辑距离会打平，LCS 比能分开：
    'ode'       vs 'node'       编辑距离 1   LCS 比 0.750
    'modelcope' vs 'modelscope' 编辑距离 1   LCS 比 0.900
  → 只看编辑距离会误判成 node（实测踩过），LCS 比能选对 modelscope。
"""
import re
from dataclasses import dataclass

ASCII_RE = re.compile(r"[a-z0-9][a-z0-9\-\.]*")


# ══════════════════════════════════════════════════════════════
#  基础度量
# ══════════════════════════════════════════════════════════════
def lev(a, b):
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


def lcs_len(a, b):
    """最长公共子序列长度"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def norm(s):
    """归一化：小写 + 只留 ASCII 字母数字连字符点

    ⭐ 英文名字里混进中文几乎一定是识别错误
       「lede了r」→ "leder"（不清洗的话距离 blender 是 5，救不回来）
    """
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch.isascii())
    return re.sub(r"[^a-z0-9\-\.]", "", s)


def similarity(hyp, cand):
    """综合相似度 ∈ [0,1]，越大越像

    三个分量：
      lcs    LCS / max(len(hyp), len(cand))   ← ⭐ 必须用 max 做分母
             用 len(cand) 做分母会给超短候选白送满分：
             「打开itub」→ 候选「B站」归一化成 "b"，LCS 1/1 = 1.0 直接满分。
             用 max 后 1/4 = 0.25，正确排除。
      ned    1 - lev/max(len)                  ← 对「字母替换」敏感
      anchor 子串/前后缀关系加分（丢首字母的典型）
    """
    h, c = norm(hyp), norm(cand)
    if not h or not c:
        return 0.0
    L = max(len(h), len(c))
    lcs = lcs_len(h, c) / L
    ned = 1.0 - lev(h, c) / L
    anchor = 0.0
    if h in c or c in h:
        anchor = 1.0
    elif c.startswith(h) or c.endswith(h) or h.startswith(c) or h.endswith(c):
        anchor = 0.85
    return max(lcs, ned, anchor)


# ══════════════════════════════════════════════════════════════
#  匹配器
# ══════════════════════════════════════════════════════════════
@dataclass
class Entry:
    kind: str          # launch_app / open_url / open_file
    id: str
    name: str
    aliases: list
    target: str        # path 或 url


@dataclass
class Match:
    entry: Entry | None
    score: float
    matched_form: str | None    # 命中的是 name 还是某个 alias
    hyp_span: str | None        # 转写里对应的片段


class Matcher:
    def __init__(self, cfg, threshold=0.72):
        self.cfg = cfg
        self.threshold = threshold
        self.entries = []
        for kind, lst, key in (("launch_app", cfg.apps, "path"),
                               ("open_url", cfg.urls, "url"),
                               ("open_file", cfg.files, "path")):
            for e in lst:
                self.entries.append(Entry(kind, e["id"], e["name"],
                                          list(e.get("aliases", [])), e[key]))
        # 每个条目所有可匹配的形式（name + aliases），预归一化
        self._forms = []
        for e in self.entries:
            fs = []
            for f in [e.name] + e.aliases:
                n = norm(f)
                if n:
                    fs.append((n, f))
            self._forms.append(fs)

    # ────────────────────────────────────────────
    def best(self, hyp, kinds=None, threshold=None):
        """在闭集里找与 hyp 最像的条目。

        kinds: 限定只在这些意图里找（比如模型判了 open_url 就只看网址闭集）
               ⚠️ 但【不要】因为模型判 launch_app 就只看 apps ——
                  实测「打开modelcope」模型判 launch_app，正确答案在 urls 里。
                  默认 kinds=None 表示全闭集搜索。
        """
        thr = self.threshold if threshold is None else threshold
        h = norm(hyp)
        if not h:
            return Match(None, 0.0, None, None)
        best, bs, bf = None, 0.0, None
        for e, fs in zip(self.entries, self._forms):
            if kinds and e.kind not in kinds:
                continue
            for n, orig in fs:
                s = similarity(h, n)
                if s > bs:
                    best, bs, bf = e, s, orig
        if best is not None and bs >= thr:
            return Match(best, bs, bf, hyp)
        return Match(None, bs, bf, hyp)

    # ────────────────────────────────────────────
    def spans(self, transcript):
        """把转写切成 span 列表：[('zh', 文本), ('en', 文本), ...]"""
        out, buf, mode = [], [], None
        for ch in transcript:
            m = "en" if (ch.isascii() and (ch.isalnum() or ch in "-.")) else "zh"
            if mode is None:
                mode = m
            elif m != mode:
                out.append((mode, "".join(buf)))
                buf = []
                mode = m
            buf.append(ch)
        if buf:
            out.append((mode, "".join(buf)))
        return out

    def match_transcript(self, transcript, kinds=None, threshold=None):
        """对整条转写做热词匹配。

        对每个 ASCII span 单独匹配；中文 span 也整体试一次
        （中文名字也可能被识别错，比如「产品需求文档」→「产产品需求文档」）。
        返回 (最佳 Match, 全部候选列表)
        """
        cands = []
        for mode, span in self.spans(transcript):
            if mode != "en" or len(span) < 2:
                continue
            m = self.best(span, kinds, threshold)
            if m.entry is not None:
                cands.append((m.score, span, m))
        cands.sort(key=lambda x: -x[0])
        return (cands[0][2] if cands else Match(None, 0.0, None, None)), cands


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    from pathlib import Path
    HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(HERE))
    from actions import Config

    cfg = Config(HERE / "intent_config.json")
    M = Matcher(cfg)
    print(f"闭集条目 {len(M.entries)} 个")

    # ⭐ 实测失败的转写原样拿来测
    CASES = [
        ("打开lender", "blender"), ("启动llama", "ollama"),
        ("启动owerhell", "powershell"), ("启动ython", "python"),
        ("打开team", "steam"), ("打开rae", "trae"),
        ("启动lash", "clash"), ("打开ode", "node"),
        ("打开ulkan测试", "vulkan"), ("启动dge浏览器", "edge"),
        ("打开itub", "github"), ("打开uggingace", "huggingface"),
        ("打开modelcope", "modelscope"), ("打开lede了r", "blender"),
        ("开lndar", "blender"), ("打开cdsn", "csdn"),
        ("打开", None), ("今天天气怎么样", None), ("帮我记一下", None),
    ]
    print()
    print("=" * 86)
    print("  转写后热词匹配（LCS 比 + 归一化编辑距离 + 锚定）")
    print("=" * 86)
    ok = 0
    for text, want in CASES:
        m, cands = M.match_transcript(text)
        got = m.entry.name if m.entry else None
        if want is None:
            good = (m.entry is None)
        else:
            good = bool(m.entry) and (
                norm(want) in norm(m.entry.name) or norm(want) in norm(m.entry.id)
                or norm(m.entry.name) in norm(want))
        ok += int(bool(good))
        mark = "✅" if good else "❌"
        print(f"  {mark} {text:20s} → {str(got):14s} (score {m.score:.3f}, "
              f"form={m.matched_form!r})  期望 {want}")
    print()
    print(f"  通过 {ok}/{len(CASES)}")
