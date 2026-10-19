# -*- coding: utf-8 -*-
"""双路对账模拟（离线）—— 验证「A 提前切分执行 + B 整段不切当权威，错了回滚」这条设计。

两条路都跑在同一批【真实录音文本】上（441 条）：
  A（提前执行，流式切分路）
     切分器切段 → 每段只认【规则层直通】（G1，0 调度延迟）
     → **且只执行可撤销动作**（open_*/search 类；close_*/清空/听写模式 不可逆，一律等 B）
  B（权威，整段不切路 = 今天的路径）
     整句过一次总闸（plausible：太长/不像人话 → 直接拒）
     → 过闸则全管线判一条（规则层 → kNN + 置信 + 闭集）
     按设计：B 是权威，A 与 B 不一致 → 回滚 A（可撤销）+ 按 B 补执行

⚠️ 离线能测的与不能测的（必须说清，别再出"0.2s"那种错）
  · 能测：A（完整文本切分 + G1 + 可撤销白名单）vs B（整句不切）的【判定口径差异】
          —— 这就是"需回滚 / 需补执行"的条数和清单
  · 不能测：A 用的是**流式前缀文本**（更碎、还会被 delta 改写），
          离线用完整文本喂 A 是【乐观上界】；真实的前缀差异只能在线测（阶段2 只打印不执行）
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import voice_agent as va                                        # noqa: E402
from actions import Config                                      # noqa: E402
from intent_split import (Splitter, triggers_from_cfg,          # noqa: E402
                          boundary_from_cfg, names_from_cfg)

# 可撤销（能关掉/撤回）→ 允许 A 提前执行；反之只能等 B
REVERSIBLE = {"open_url", "launch_app", "open_file", "web_search",
              "open_web", "exact_search", "file_search"}
IRREVERSIBLE = {"close_app", "clear_input", "dictation_on", "dictation_off"}


def verdict(cfg, embed, text, allow_knn=True):
    """生产口径判定 → dict(ok, intent, arg, why)

    ⚠️ 总闸（plausible）必须最先过：生产里规则层也后置在它之后
       （配置注释：规则同样要过 plausible，噪声过不来；长度门控就在这里）。
    """
    try:
        ok_p, why_p = va.plausible(text, cfg)
    except Exception:
        ok_p, why_p = True, ""
    if not ok_p:
        return {"ok": False, "intent": "?", "arg": "", "why": f"总闸不过：{why_p}"}
    try:
        q = va.tool_query(cfg, text)
    except Exception as e:
        return {"ok": False, "intent": "?", "arg": "", "why": f"tool_query异常 {e}"}
    _act = q.get("action")
    # ⚠️ 判"规则层命中"只看 action，**不能**要求 target 非空：
    #    tool_query 对 open_web/clear_input/dictation_* 这几类不建 target，
    #    要求 target 会把它们误判成"规则层没命中" → 落到 kNN → 变成别的意图
    #    （实测：「搜索Deepseek网页版」因此被报成 launch_app）
    if _act and _act not in ("?", "不执行"):
        return {"ok": True, "intent": q["intent"], "arg": q.get("arg") or "",
                "resolved": q.get("resolved"),
                "why": f"规则层 {_act} → {q.get('target') or q.get('arg')}"}
    if embed is not None and allow_knn:
        try:
            r = embed.predict(text)
        except Exception as e:
            return {"ok": False, "intent": "?", "arg": "", "why": f"kNN异常 {e}"}
        it, conf = r["intent"], float(r["conf"])
        if it != "unknown" and conf >= float(embed.thr):
            ok_p, why_p = va.plausible(text, cfg)
            if not ok_p:
                return {"ok": False, "intent": it, "arg": "",
                        "why": f"总闸不过：{why_p}"}
            try:
                arg = va.resolve(it, text, cfg)
                hit = va.resolve_closed(it, arg, cfg)
            except Exception:
                arg, hit = "", None
            if it in ("launch_app", "open_url", "open_file") and not hit:
                return {"ok": False, "intent": it, "arg": arg,
                        "why": f"kNN {it} 但闭集里没有参数 → 拒识"}
            return {"ok": True, "intent": it, "arg": arg or "",
                    "why": f"kNN {it} conf={conf:.3f}"}
        return {"ok": False, "intent": it, "arg": "", "why": f"kNN {it} conf={conf:.3f} 低于 thr"}
    return {"ok": False, "intent": q.get("intent", "unknown"), "arg": "",
            "why": q.get("reason") or "不执行"}


def main():
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
        print(f"  ⚠️ kNN 不可用（{e}）—— 只跑规则层，数字会偏保守")

    cfgj = json.loads((HERE / "intent_config.json").read_text(encoding="utf-8"))
    TR = triggers_from_cfg(cfgj)
    BD = boundary_from_cfg(cfgj)
    NM = names_from_cfg(cfgj)

    rows = [json.loads(l) for l in (HERE / "utt" / "index.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    texts = [r.get("text", "") for r in rows if r.get("src") == "mic" and r.get("text")]

    stat = Counter()
    rollback, fill, conflict, agree = [], [], [], []
    for t in texts:
        segs = [s.text for s in Splitter(TR, boundary=BD, names=NM).segments(t)]
        # A：每段只认规则直通 + 只执行可撤销动作
        a_all = [verdict(cfg, embed, s, allow_knn=False) for s in segs]
        a = [v for v in a_all if v["ok"] and v["intent"] in REVERSIBLE]
        a_blocked = [v for v in a_all if v["ok"] and v["intent"] not in REVERSIBLE]
        # B：整句不切（今天的路径，允许 kNN）
        b = verdict(cfg, embed, t, allow_knn=True)

        if a and not b["ok"]:
            stat["需回滚（B 说不执行）"] += 1
            rollback.append((t, a, b))
        elif a and b["ok"]:
            if (a[0]["intent"], a[0]["arg"]) == (b["intent"], b["arg"]):
                if len(a) > 1:
                    stat["一致但 A 多出同句多条"] += 1
                else:
                    stat["一致"] += 1
                agree.append((t, a, b))
            elif any((x["intent"], x["arg"]) == (b["intent"], b["arg"]) for x in a):
                stat["A 多出（需撤多余）"] += 1
                rollback.append((t, a, b))
            else:
                stat["冲突（放 A 的、执行 B 的）"] += 1
                conflict.append((t, a, b))
        elif not a and b["ok"]:
            stat["需补执行（A 没做，B 补）"] += 1
            fill.append((t, a, b))
        else:
            stat["两边都不动"] += 1
        if a_blocked:
            stat["A 因不可逆被拦下（等 B）"] += 1

    n = len(texts)
    print(f"\n  ══ 双路对账（{n} 条真实录音文本）══")
    for k, v in stat.most_common():
        print(f"     {k:<26} {v:>4} 条  ({v/n*100:4.1f}%)")

    def show(title, rows_, k=8):
        print(f"\n  ══ {title}：{len(rows_)} 条 ══")
        for t, a, b in rows_[:k]:
            aa = " | ".join(f"{x['intent']}({x['arg']})" for x in a) or "—"
            print(f"     「{t[:42]}」")
            print(f"        A 提前执行：{aa}")
            print(f"        B 权威　　：{'✅ ' + b['intent'] + '(' + b['arg'] + ')' if b['ok'] else '⛔ ' + b['why'][:34]}")

    show("需回滚（A 执行了、B 说不执行）", rollback)
    show("冲突（A 与 B 不是同一条）", conflict)
    show("需补执行（A 没做、B 补上）", fill, 5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
