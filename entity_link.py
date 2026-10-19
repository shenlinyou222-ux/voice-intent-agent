# -*- coding: utf-8 -*-
"""L3 实体链接：把 ASR 转写里的参数，链接到【闭集实体】上。

为什么需要（实测血债）：
  「直呼」->知乎（同音）、「记事板」->记事本（近形）、「第四个」->DeepSeek（经验误听）、
  「打開知乎」->繁体（已由 L0 归一化解决）、「豆瓣」->豆包（**必须拒绝**：2字中文差一字=不同词）

四路融合打分（业界做法：规则词典 + 拼音 + 编辑距离 + 语义，各自有独立置信度）：
  ① 归一化精确（大小写/空格/标点无关）      -> 1.00
  ② 拼音完全相同                           -> 0.92
  ③ 拼音差 1 且首字母同（名字 >=3 字）        -> 0.82
  ④ 编辑距离 1（名字 >=3 字；2字中文只允许精确）-> 0.80
  ⑤ 经验误听表（从日志里学到的固定错法）      -> 0.75
  ⑥ 语义余弦（可选，调 embedding 服务）      -> 0.6~0.95
取最高分；低于阈值（默认 0.75）判"链接失败"（宁可拒识也不乱开 ✗）
"""
from difflib import SequenceMatcher

try:
    from pypinyin import lazy_pinyin
except Exception:
    lazy_pinyin = None

# ⑥ 语义余弦（可选）：只在前几路都失败、且名字较短时才调，避免常态延迟
EMB_URL = "http://172.20.71.101:30802/v1/embeddings"
EMB_MODEL = "dmeta-small"

# ⑤ 经验误听表（来自真实日志；只收"确定性高、不会与正常词冲突"的）
CONFUSION = {
    "第四个": "deepseek",
    "第四个网页": "deepseek网页",
    "克劳德": "claude",
    "油管": "youtube",
    "开源中国": "开源中国",
}


def _py(s):
    if not lazy_pinyin:
        return ""
    try:
        return "".join(lazy_pinyin(s)).lower()
    except Exception:
        return ""


def _norm(s):
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def _lev(a, b):
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _emb_cos(texts, cands):
    """语义余弦：返回 {(text,cand): cos}；失败返回 {}（不阻塞主流程）"""
    try:
        import json as _j
        import urllib.request as _u
        import numpy as np
        body = _j.dumps({"model": EMB_MODEL, "input": list(texts) + list(cands)}).encode()
        req = _u.Request(EMB_URL, data=body, headers={"Content-Type": "application/json"})
        with _u.urlopen(req, timeout=3) as r:
            d = _j.loads(r.read())["data"]
        V = np.asarray([x["embedding"] for x in d], dtype=np.float32)
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        nt = len(texts)
        return {(t, c): float(V[i] @ V[nt + k])
                for i, t in enumerate(texts) for k, c in enumerate(cands)}
    except Exception:
        return {}


def link(arg, candidates, thr=0.75, use_embed=False):
    """把 arg 链接到 candidates（闭集实体的 name+aliases）里的某一个。

    candidates: [(key, display_name), ...]   key 用于打分，display_name 返回给上层
    -> (display_name | None, score, how)
    """
    a = (arg or "").strip()
    if not a:
        return None, 0.0, "empty"
    an, ap = _norm(a), _py(a)
    best = (None, 0.0, "none")
    for key, name in candidates:
        k = str(key)
        kn, kp = _norm(k), _py(k)
        s, how = 0.0, ""
        if an and an == kn:                       # ① 归一化精确
            s, how = 1.00, "exact"
        elif a.lower() == k.lower():
            s, how = 1.00, "exact"
        if s < 0.92 and ap and kp and ap == kp:   # ② 拼音相同（直呼->知乎）
            s, how = 0.92, "pinyin="
        # ③ 拼音差 1：⚠️ 必须同时要求【名字】>=3 字 —— 之前判断的是拼音长度
        #    （douban=6），于是「豆瓣」又被吸到「豆包」✗（用户早就骂过这个）。
        #    2 字中文名字只允许拼音【完全相同】（②）。
        if (s < 0.82 and ap and kp and len(k) >= 3 and ap[0] == kp[0]
                and _lev(ap, kp) <= 1):
            s, how = 0.82, "pinyin~1"
        if s < 0.80:
            # ④ 编辑距离：>=3 字允许 1；2 字中文只允许精确（豆瓣 != 豆包）
            short_cjk = (len(k) <= 2 and all("\u4e00" <= ch <= "\u9fff" for ch in k))
            if not kn:
                pass
            # ④a 距离 1 且【两边都 >=3 字】——真正的近似（记事板->记事本）
            elif len(an) >= 3 and len(kn) >= 3 and _lev(an, kn) <= 1:
                s, how = 0.80, "lev1"
            # ④b 输入更长（多出前后缀字）：知乎网->知乎、记事本吧->记事本 —— 安全 ✓
            elif len(an) > len(kn) >= 2 and _lev(an, kn) <= 1:
                s, how = 0.78, "lev1-suffix"
            # ④c 归一化后完全相同（标点/空格差异）
            elif not short_cjk and _lev(an, kn) == 0:
                s, how = 0.90, "exact-norm"
        if s < 0.75 and a in CONFUSION and CONFUSION[a] == k.lower():
            s, how = 0.75, "confusion"            # ⑤ 经验误听（第四个->DeepSeek）
        if s > best[1]:
            best = (name, s, how)
    # ⑥ 语义兜底（可选，慢）
    if best[1] < thr and use_embed and len(a) <= 8:
        cos = _emb_cos([a], [str(k) for k, _ in candidates])
        for (t, k), v in cos.items():
            if v > best[1]:
                name = next((n for kk, n in candidates if str(kk) == k), None)
                if name:
                    best = (name, float(v), "embed")
    if best[1] >= thr:
        return best
    return None, best[1], best[2] + "(below_thr)"
