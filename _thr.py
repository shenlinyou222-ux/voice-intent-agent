import json, random, urllib.request
from pathlib import Path
import numpy as np
HERE = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
EMB = "http://172.20.71.101:30802/v1/embeddings"

def embed(txts):
    out = []
    for i in range(0, len(txts), 32):
        body = json.dumps({"model": "dmeta-small", "input": txts[i:i+32]}).encode()
        req = urllib.request.Request(EMB, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out += [d["embedding"] for d in json.loads(r.read())["data"]]
    return np.array(out, dtype=np.float32)

def norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

rows = [json.loads(l) for l in (HERE / "intent_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
# ⭐ 用语料自带的 split 字段（比随机切分更接近真实分布）
tr = [r for r in rows if r.get("split") == "train"] or rows[:1625]
va = [r for r in rows if r.get("split") != "train"] or rows[1625:]
print("train=%d val=%d  (用 split 字段)" % (len(tr), len(va)))
labels = sorted({r["intent"] for r in rows})
Etr = norm(embed([r["text"] for r in tr]))
tl = [r["intent"] for r in tr]
# 原型 = 每类质心 + 每类示例向量（kNN 用）
C, Cn, ex, exl = [], [], [], []
for lab in labels:
    idx = [i for i, l in enumerate(tl) if l == lab]
    if not idx: continue
    C.append(norm(Etr[idx].mean(0, keepdims=True))[0]); Cn.append(lab)
    for i in idx: ex.append(Etr[i]); exl.append(lab)
Cm, Ex = np.stack(C), np.stack(ex)
V = norm(embed([r["text"] for r in va]))
vl = [r["intent"] for r in va]
sc = V @ Cm.T
km = V @ Ex.T
def run(thr, margin, use_knn):
    pred = []
    for i in range(len(vl)):
        if use_knn:
            j = int(km[i].argmax()); best, lab = float(km[i][j]), exl[j]
            srt = np.sort(km[i])[::-1]
        else:
            j = int(sc[i].argmax()); best, lab = float(sc[i][j]), Cn[j]
            srt = np.sort(sc[i])[::-1]
        second = float(srt[1]) if len(srt) > 1 else 0.0
        pred.append("unknown" if (best < thr or (best - second) < margin) else lab)
    return sum(1 for p, g in zip(pred, vl) if p == g) / len(vl), pred
print("\n  === 阈值扫描（原型 / kNN）===")
best = (0, None)
for use_knn in (False, True):
    for thr in (0.30, 0.35, 0.40, 0.45, 0.50):
        for margin in (0.00, 0.02, 0.05):
            a, _ = run(thr, margin, use_knn)
            if a > best[0]: best = (a, ("kNN" if use_knn else "质心", thr, margin))
    print("  %-4s 最佳: %.4f @ %s" % ("kNN" if use_knn else "质心",
          max(run(t, m, use_knn)[0] for t in (0.30,0.35,0.40,0.45,0.50) for m in (0.0,0.02,0.05)),
          "见下表"))
print("\n  ★ 全局最佳: acc=%.4f  方案=%s thr=%.2f margin=%.2f" % (best[0], *best[1]))
a, pred = run(best[1][1], best[1][2], best[1][0] == "kNN")
print("\n  逐类召回（最佳配置）:")
for lab in labels:
    idx = [i for i, g in enumerate(vl) if g == lab]
    if idx: print("    %-14s n=%-4d recall=%.2f" % (lab, len(idx), sum(pred[i]==lab for i in idx)/len(idx)))
bad = [(va[i]["text"][:30], vl[i], pred[i]) for i in range(len(va)) if pred[i] != vl[i]][:8]
print("\n  错例（前 8）:")
for t, g, p in bad: print("    真=%-12s 判=%-12s 「%s」" % (g, p, t))
