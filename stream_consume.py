# -*- coding: utf-8 -*-
"""流式文本 → 可执行指令序列（A 路消费器）

职责（方案 §〇之二 里"文本域唯一裁决者"的落地）：
  ① 增量喂入：把 AgentStream 的 delta 增量喂进 Splitter（触发词定边界）
  ② 段形状检查：strict_shape()（触发词在段首 + 参数紧随 + 尾巴≤5字）
  ③ A 路判定：只认**规则层直通** + 只留**可撤销动作**
  ④ 去重（两个都必须）：
     · 段去重：同一句里文本相同的段只处理一次
       —— 实测必需：收尾（final commit）会把最后几秒**重解一遍**，
         重复段会让同一动作执行两次（11 段 vs 离线 8 段就是这么来的）
     · 动作去重：同样的 (intent, arg) 在同一句里只执行一次
  ⑤ 记账：本句 A 路决定执行了哪些动作（供与 B 路权威对账）

刻意不做：不判模型意图（那条等 B 路）、不执行动作（执行由调用方给 execute 开关）。
"""
from intent_split import Splitter, strict_shape
from _dual_path_sim import verdict, REVERSIBLE
from _dual_replay import akey


class StreamConsumer:
    def __init__(self, cfg, triggers, boundary, names, on_log=None):
        self.cfg = cfg
        self.BD = boundary
        self.NM = names
        self.sp = Splitter(triggers, boundary=boundary, names=names)
        self.on_log = on_log or (lambda m: None)
        self.closed_texts = []          # 已闭合段文本（去重）
        self.n_skipped_dup = 0
        self.actions = []               # [(intent, arg, resolved)] 本句 A 路已决定的动作

    def feed(self, text) -> list:
        """喂一段增量文本，返回本次【新决定】的动作列表"""
        out = []
        for seg in self.sp.push(text or ""):
            t = (seg.text or "").strip()
            if not t:
                continue
            if t in self.closed_texts:               # ④ 段去重（收尾重解）
                self.n_skipped_dup += 1
                continue
            self.closed_texts.append(t)
            ok_shape, tr, nm, left = strict_shape(t, self.BD, self.NM)
            if not ok_shape:
                self.on_log(f"   · 段形状不良，不执行：「{t[:24]}」（触发={tr or '—'} 尾巴={left}）")
                continue
            v = verdict(self.cfg, None, t, allow_knn=False)     # ③ 只认规则直通
            if not (v["ok"] and v["intent"] in REVERSIBLE):
                continue
            key = akey(v["intent"], v["arg"], v.get("resolved"))
            if key in {akey(i, a, r) for i, a, r in self.actions}:   # ④ 动作去重
                self.n_skipped_dup += 1
                continue
            self.actions.append((v["intent"], v["arg"], v.get("resolved")))
            out.append({"text": t, "intent": v["intent"], "arg": v["arg"],
                        "resolved": v.get("resolved"), "why": v["why"]})
        return out

    def flush(self) -> list:
        """收尾：把缓冲里剩下的段也走一遍同样的判定"""
        out = []
        for seg in self.sp.flush():
            t = (seg.text or "").strip()
            if not t or t in self.closed_texts:
                self.n_skipped_dup += 1
                continue
            self.closed_texts.append(t)
            ok_shape, tr, nm, left = strict_shape(t, self.BD, self.NM)
            if not ok_shape:
                continue
            v = verdict(self.cfg, None, t, allow_knn=False)
            if not (v["ok"] and v["intent"] in REVERSIBLE):
                continue
            key = akey(v["intent"], v["arg"], v.get("resolved"))
            if key in {akey(i, a, r) for i, a, r in self.actions}:
                self.n_skipped_dup += 1
                continue
            self.actions.append((v["intent"], v["arg"], v.get("resolved")))
            out.append({"text": t, "intent": v["intent"], "arg": v["arg"],
                        "resolved": v.get("resolved"), "why": v["why"]})
        return out

    def reset(self):
        """一句说完了 → 清空（下一次按住重新开始）"""
        self.sp = Splitter(self.sp.triggers, boundary=self.BD, names=self.NM)
        self.closed_texts, self.actions = [], []
