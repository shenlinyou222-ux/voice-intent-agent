# -*- coding: utf-8 -*-
"""双路离线重放 —— 用【真机录下来的那句 15 秒长句】验证改好的判据（不需要人再说话）。

数据源：_dual/*_dual.wav（用户 21:09 按住 14.52s 说的那句 8 指令长句，15.54s）

A 路（流式前缀）：按 1s 切片，每片把【已累积音频】重转一次 → 切分 → 段形状检查
                 → 只认规则直通 + 只留可撤销动作 + **连续 2 片一致才允许执行**（稳定片段）
B 路（整段权威）：全部音频转一次 → 切分 → 段形状检查 + 全管线
                 → **形状良好段 ≥2 时不被"整句太长"否决**（这是这次要验证的核心改动）
                 → 形状良好段 ≤1 且整句超长 → 否决全部（保住 33 字非命令的防线）
对账：A 已执行 vs B 权威 → 一致 / A 多出(需回滚) / B 多出(需补) / 冲突

用法：python -X utf8 _dual_replay.py [wav]
"""
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from intent_split import (Splitter, triggers_from_cfg, boundary_from_cfg,  # noqa: E402
                          names_from_cfg, strict_shape)
from _cut_align import asr as asr_post                                    # noqa: E402
from _dual_path_sim import verdict, REVERSIBLE                            # noqa: E402
import voice_agent as va                                                  # noqa: E402
from actions import Config                                                # noqa: E402
import json                                                               # noqa: E402
import re                                                                 # noqa: E402


def akey(it, arg, resolved=None):
    """对账用的动作键：优先用解析出的闭集 id（DeepSeek/Deepseek 是同一个），
       否则用归一化后的参数（去大小写与标点）—— 否则同一动作会被算成三个。"""
    base = resolved if resolved else arg
    return (it, re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(base).lower()))

SR = 16000
CFGJ = json.loads((HERE / "intent_config.json").read_text(encoding="utf-8"))
TR = triggers_from_cfg(CFGJ)
BD = boundary_from_cfg(CFGJ)
NM = names_from_cfg(CFGJ)
SEC = 1.0


def split(text):
    return [s.text for s in Splitter(TR, boundary=BD, names=NM).segments(text)]


def a_actions(cfg, text):
    """A 路：形状良好 + 规则直通 + 可撤销"""
    out = []
    for seg in split(text):
        ok_shape, tr, nm, left = strict_shape(seg, BD, NM)
        if not ok_shape:
            continue
        v = verdict(cfg, None, seg, allow_knn=False)
        if v["ok"] and v["intent"] in REVERSIBLE:
            out.append((v["intent"], v["arg"], f"{tr}+{nm}", v.get("resolved")))
    return out


def b_actions(cfg, embed, text):
    """B 路：整段文本 → 切分 → 形状检查 + 全管线；长度门控只在"形状良好≤1"时否决"""
    segs = split(text)
    good, bad = [], []
    for seg in segs:
        ok_shape, tr, nm, left = strict_shape(seg, BD, NM)
        v = verdict(cfg, embed, seg, allow_knn=True)
        (good if ok_shape else bad).append((seg, tr, nm, left, v))
    if len(good) <= 1:
        w = verdict(cfg, embed, text, allow_knn=True)      # 退回整句口径（含长度门控）
        return ([(w["intent"], w["arg"], "整句", w.get("resolved"))] if w["ok"] else []), good, bad, w
    acts = [(v["intent"], v["arg"], f"{tr}+{nm}", v.get("resolved"))
            for _, tr, nm, _, v in good if v["ok"] and v["intent"] in REVERSIBLE]
    return acts, good, bad, {"ok": True, "why": f"整段切出 {len(good)} 条形状良好指令"}


def main():
    wavs = sorted((HERE / "_dual").glob("*_dual.wav"))
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else (wavs[-1] if wavs else None)
    if not p or not p.exists():
        print(f"  找不到录音（{HERE/'_dual'}/*_dual.wav）")
        return 2
    with wave.open(str(p), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    cfg = Config(None)
    embed = None
    try:
        from intent_embed import EmbedIntent
        embed = EmbedIntent(
            url=str(cfg.behavior.get("intent_embed_url")
                    or "http://172.20.71.101:30802/v1/embeddings"),
            index=str(HERE / str(cfg.behavior.get("intent_index", "intent_index.json"))),
            thr=float(cfg.behavior.get("intent_embed_thr") or 0.30),
            margin=float(cfg.behavior.get("intent_embed_margin") or 0.0))
    except Exception as e:
        print(f"  ⚠️ kNN 不可用（{e}）")

    print(f"\n  录音 {p.name}  {len(x)/SR:.2f}s")
    print(f"\n  ══ A 路：每 {SEC:.0f}s 一片，累积重转（只认规则直通 + 可撤销 + 稳定2片）══")
    seen = {}
    executed = []
    for k in range(1, int(len(x) / SR / SEC) + 1):
        audio = x[:int(k * SEC * SR)]
        text = asr_post(audio, gain=8.0)
        acts = a_actions(cfg, text)
        tag = []
        for it, arg, why, res in acts:
            key = akey(it, arg, res)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] == 2:                       # 连续 2 片一致 → 允许执行
                executed.append((it, arg, why, res))
                tag.append(f"✅执行 {it}({arg})")
            elif seen[key] == 1:
                tag.append(f"…观察 {it}({arg})")
        print(f"     +{k*SEC:4.1f}s 「{text[:64]}」")
        if tag:
            print(f"              {'  '.join(tag)}")

    print(f"\n  ══ B 路：整段权威 ══")
    ftext = asr_post(x, gain=8.0)
    b, good, bad, gate = b_actions(cfg, embed, ftext)
    print(f"     整段文本：「{ftext}」")
    print(f"     切 {len(good)+len(bad)} 段：形状良好 {len(good)} / 不良 {len(bad)}")
    for seg, tr, nm, left, v in good:
        print(f"       ✅ 「{seg[:34]}」 触发词={tr or '—'} 参数={nm or '—'} 尾巴={left} "
              f"→ {v['intent']}({v['arg']})" if v["ok"] else
              f"       ✅ 「{seg[:34]}」 形状良好但判定不执行（{v['why'][:26]}）")
    for seg, tr, nm, left, v in bad:
        print(f"       ⛔ 「{seg[:34]}」 形状不良（触发词={tr or '—'} 尾巴={left}）")
    print(f"     总闸：{'过' if gate.get('ok') else '不过 —— ' + str(gate.get('why'))[:40]}")

    print(f"\n  ══ 对账 ══")
    sa = {akey(i, a, r) for i, a, _, r in executed}
    sb = {akey(i, a, r) for i, a, _, r in b}
    print(f"     A 已执行 {len(executed)} 条：{sorted(sa)}")
    print(f"     B 权威   {len(b)} 条：{sorted(sb)}")
    if sa == sb:
        print(f"     ⚖ 一致 ✅")
    else:
        print(f"     ⚖ A 多出（需回滚）：{sorted(sa - sb)}")
        print(f"     ⚖ B 多出（需补执行）：{sorted(sb - sa)}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
