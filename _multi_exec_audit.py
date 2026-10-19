# -*- coding: utf-8 -*-
"""多指令切分的【风险画像】—— 切成片段后，会不会多出"今天不会执行"的动作。

不执行任何动作：只用生产侧的两个纯判定入口
  · 规则层：voice_agent.tool_query(cfg, text)   ← parse→resolve→resolve_closed，零漂移
  · kNN 层：intent_embed.EmbedIntent.predict()  ← 走已在跑的 30802 embed 服务
判"可执行"与生产一致：
  · 规则层命中闭集/有目标 → 可执行（rule_direct=true，模型置信度不参与否决）
  · 否则 kNN：intent != unknown 且 conf >= thr → 可执行

关注三类：
  A 新增执行：整句不可执行，但至少一个片段可执行   ← **这是误判风险的来源**
  B 动作改变：整句可执行，但片段执行的是别的动作
  C 无损：片段与整句一致（纯切分，不改变行为）

用法：E:\\voice_agent_build\\Scripts\\python.exe -X utf8 _multi_exec_audit.py [--all]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import voice_agent as va                                     # noqa: E402
from actions import Config                                   # noqa: E402
from intent_split import (Splitter, triggers_from_cfg, boundary_from_cfg,  # noqa: E402
                          names_from_cfg)

CFGJ = json.loads((HERE / "intent_config.json").read_text(encoding="utf-8"))
TRIGGERS = triggers_from_cfg(CFGJ)
BOUNDARY = boundary_from_cfg(CFGJ)
NAMES = names_from_cfg(CFGJ)
RUN_ALL = "--all" in sys.argv
G1 = "--g1" in sys.argv        # 护栏：分条执行只认规则层直通（kNN-only 片段不执行）


def judge(cfg, embed, text, allow_knn=True):  # noqa: D401
    """返回 (可执行?, 意图, 动作说明) —— 判据与生产对齐：

    · 规则层：tool_query（内含闭集检查 hit）→ 有目标才算可执行（rule_direct=true）
    · kNN 层：intent != unknown 且 conf >= thr，**且** 再过两道生产门：
        - plausible()：句子里得有已知词（触发词/闭集名），否则是识别乱码
        - 闭集意图（launch_app/open_url/open_file）必须有 resolve_closed 命中
          ← 少了这道就会把"参数不在闭集里"的句子误判成可执行（第一版审计就踩了）
    """
    try:
        ok_p, why_p = va.plausible(text, cfg)
    except Exception:
        ok_p, why_p = True, ""
    if not ok_p:
        return False, "?", f"总闸不过：{why_p}"
    try:
        q = va.tool_query(cfg, text)
    except Exception as e:
        return False, "?", f"⚠️ tool_query异常 {type(e).__name__}: {e}"
    _act = q.get("action")
    if _act and _act != "不执行" and q.get("target"):
        return True, q["intent"], f"{_act} → {q.get('target')}"
    if embed is not None and allow_knn:
        try:
            r = embed.predict(text)
        except Exception as e:
            return False, "?", f"kNN异常 {type(e).__name__}"
        it, conf = r["intent"], float(r["conf"])
        if it != "unknown" and conf >= float(embed.thr):
            try:
                ok_plausible, why_p = va.plausible(text, cfg)
            except Exception:
                ok_plausible, why_p = True, ""
            if not ok_plausible:
                return False, it, f"kNN {it} conf={conf:.3f} 但不像人话：{why_p}"
            if it in ("launch_app", "open_url", "open_file"):
                try:
                    arg = va.resolve(it, text, cfg)
                    hit = va.resolve_closed(it, arg, cfg)
                except Exception:
                    hit = None
                if not hit:
                    return False, it, f"kNN {it} conf={conf:.3f} 但闭集里没有参数 → 生产会拒识"
            return True, it, f"kNN {it} conf={conf:.3f}"
    return False, q.get("intent", "unknown"), (q.get("reason") or "不执行")


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
        print(f"  kNN 层就绪：{len(embed.labels)} 示例 / {len(embed.intents)} 类 / "
              f"thr={embed.thr}")
    except Exception as e:
        print(f"  ⚠️ kNN 层不可用（{type(e).__name__}: {e}）→ 只审计规则层")

    rows = [json.loads(l) for l in (HERE / "utt" / "index.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    texts = [r.get("text", "") for r in rows if r.get("src") == "mic" and r.get("text")]

    # 先算每条会被切成几段
    multi = []
    for t in texts:
        segs = Splitter(TRIGGERS, boundary=BOUNDARY, names=NAMES).segments(t)
        if len(segs) > 1:
            multi.append((t, [s.text for s in segs]))
    print(f"\n  真实录音文本 {len(texts)} 条，其中会被切多段的 {len(multi)} 条")
    targets = multi if RUN_ALL else multi

    A = B = C = 0
    detail = []
    for t, segs in targets:
        w_ok, w_int, w_do = judge(cfg, embed, t)
        s_res = [judge(cfg, embed, s, allow_knn=not G1) for s in segs]
        s_ok = [x for x in s_res if x[0]]
        if not w_ok and s_ok:
            A += 1
        elif w_ok and any(x[1] != w_int for x in s_ok):
            B += 1
        else:
            C += 1
        detail.append((t, segs, (w_ok, w_int, w_do), s_res))

    print(f"\n  ══ 汇总（只看会被切多段的 {len(targets)} 条）══  护栏 G1 = {G1}")
    print(f"     A 新增执行（整句不执行 → 片段执行了）：{A} 条   ← 误判风险来源")
    print(f"     B 动作改变（整句执行别的）：{B} 条")
    print(f"     C 行为不变：{C} 条")

    print(f"\n  ══ 逐条明细 ══")
    for t, segs, w, s_res in detail:
        print(f"\n  「{t[:60]}」")
        print(f"     整句：{'✅可执行' if w[0] else '⛔不执行'}  {w[1]}  {w[2]}")
        for s, (ok, it, do) in zip(segs, s_res):
            print(f"     片段：{'✅' if ok else '  '} 「{s[:34]}」 → {it}  {do}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
