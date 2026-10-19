# -*- coding: utf-8 -*-
"""训练【纯文本头】768→8 —— 给多意图的"逐分句 embedding"用。

为什么要单独一个头：
  现有 head 吃 concat(c1, c2) = 1536 维，c1 来自【整段音频】（V1 软查表池化）。
  一句话切成 N 个分句后，音频没法跟着切（V1 是整段音频的特征），
  所以逐分句判定只能走纯文本路：c2 = B(E_ALL[分句 tokens]) 的 CLS → 768 维。
  这个头就是为这件事训的，不碰现有 1536 维头（单句路径完全不变）。

训练数据（关键：要让"分句"这个分布被覆盖）：
  ① 正向：数据集里每条 text → c2 → 它的 intent（1422 train / 609 val）
  ② ⭐ 多意图增强：把 1~3 条命令 text 和 0~2 条闲聊片段拼成一句长话，
     再过 split_clauses 切回分句 —— 每个分句拿它【原本的标签】，
     闲聊片段拿 unknown。这样"切开后的短分句"就落在训练分布里了，
     而不是只在完整单句上训练（否则上线就会遇到没见过短分句）。
  ③ 闲聊/unknown：数据集自带的 unknown 行（「今天天气怎么样」等）
     + 常见口语碎片（「我们看一下吧」「可以」「嗯」）

用法：
    python train_text_head.py            # 训练 + 报告 val 准确率 → model/head_text.pt
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, split_clauses          # noqa: E402
from model_runner import IntentModel               # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "model" / "head_text.pt"

# 闲聊碎片（真实日志里出现过的 + 常见口语），标签一律 unknown
CHATTER = ["我们看一下吧", "可以", "可以，可以", "嗯", "谢谢", "现在",
           "稍等一下", "好的", "就这样", "我知道了", "开始我们一天的学习",
           "机器还是太太变态了", "我们一起一天的学习"]


def build_samples(rows, cfg):
    """→ [(clause_text, intent_label)]，含多意图拼接增强"""
    out = [(r["text"], r["intent"]) for r in rows]
    lab = {r["text"]: r["intent"] for r in rows}
    cmds = [r["text"] for r in rows if r["intent"] != "unknown"]
    if not cmds:
        return out

    rng = np.random.default_rng(0)
    for _ in range(len(rows) * 2):
        pick = list(rng.choice(len(cmds), size=int(rng.integers(1, 4)),
                              replace=False))
        parts = [cmds[pi] for pi in pick]
        labels = [lab[cmds[pi]] for pi in pick]
        if rng.random() < 0.6:              # 掺一条闲聊碎片
            parts.append(CHATTER[int(rng.integers(0, len(CHATTER)))])
            labels.append("unknown")
        # 用【线上同一个】分句器切回来
        joined = "，".join(parts)
        li = 0
        for c in split_clauses(joined, cfg):
            found = None
            for j in range(li, len(parts)):
                if parts[j] and (parts[j] in c or c in parts[j]):
                    found = j
                    break
            if found is None:               # 分句器合并出来的碎片 → 当闲聊
                out.append((c, "unknown"))
            else:
                out.append((c, labels[found]))
                li = found + 1
    return out


def feats(model, texts, bs=32):
    """c2 = B(E_ALL[ids]) 的 CLS → (N, 768)"""
    vs = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        with torch.no_grad():
            for t in chunk:
                ids = model.tok(t, truncation=True, max_length=64)["input_ids"]
                c2 = model._b_cls(model.E_ALL[ids])
                vs.append(c2.detach().cpu().numpy())
    return np.stack(vs).astype(np.float32)


def main():
    t0 = time.time()
    cfg = Config(str(HERE / "intent_config.json"))
    rows = [json.loads(l) for l in
            open(HERE / "intent_dataset.jsonl", encoding="utf-8")]
    tr_rows = [r for r in rows if r.get("split") == "train"]
    va_rows = [r for r in rows if r.get("split") == "val"]
    print(f"数据集 {len(rows)}：train {len(tr_rows)} / val {len(va_rows)}")

    print("加载 int8 模型（只用到 tokenizer / E_ALL / B encoder）...")
    model = IntentModel(model_dir=str(HERE / "model"), device="cpu", int8=True,
                        verbose=False)
    INTENTS = model.INTENTS
    print(f"intents = {INTENTS}")

    tr = build_samples(tr_rows, cfg)
    va = build_samples(va_rows, cfg)
    print(f"样本（含多意图增强）：train {len(tr)} / val {len(va)}")
    from collections import Counter
    print("  标签分布:", Counter(l for _, l in tr).most_common())

    print("抽特征（c2, 768 维）...")
    t1 = time.time()
    Xtr = feats(model, [t for t, _ in tr])
    Xva = feats(model, [t for t, _ in va])
    ytr = np.array([INTENTS.index(l) for _, l in tr])
    yva = np.array([INTENTS.index(l) for _, l in va])
    print(f"  {Xtr.shape} / {Xva.shape}  用时 {time.time()-t1:.1f}s")

    head = nn.Linear(Xtr.shape[1], len(INTENTS))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xva_t, yva_t = torch.tensor(Xva), torch.tensor(yva)
    best, best_state = -1.0, None
    for ep in range(400):
        head.train()
        opt.zero_grad()
        loss = lossf(head(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == 399:
            head.eval()
            with torch.no_grad():
                acc = float((head(Xva_t).argmax(1) == yva_t).float().mean())
            if acc > best:
                best, best_state = acc, {k: v.clone() for k, v in
                                         head.state_dict().items()}
    if best_state:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pv = head(Xva_t)
        acc = float((pv.argmax(1) == yva_t).float().mean())
    print(f"\nval 准确率 {acc*100:.2f}%（best {best*100:.2f}%）")

    # 分意图细看
    with torch.no_grad():
        pred = pv.argmax(1).numpy()
    print(f"{'意图':<14}{'正确/总数':>10}")
    for i, name in enumerate(INTENTS):
        m = yva == i
        if m.sum():
            print(f"{name:<14}{(pred[m] == i).sum():>6}/{m.sum():<5}")

    torch.save({"w": head.weight.data, "b": head.bias.data,
                "intents": INTENTS, "val_acc": acc,
                "n_train": len(tr), "n_val": len(va)}, str(OUT))
    print(f"\n已存 {OUT}（{OUT.stat().st_size/1024:.0f} KB）"
          f"  总用时 {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
