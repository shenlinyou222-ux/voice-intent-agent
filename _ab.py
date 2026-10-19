import json, random, sys, urllib.request
from collections import Counter
from pathlib import Path
import numpy as np
HERE = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
EMB = "http://172.20.71.101:30802/v1/embeddings"
rows = [json.loads(l) for l in (HERE / "intent_dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
print("语料 %d 条  字段=%s" % (len(rows), list(rows[0].keys())))
tk = next(k for k in rows[0] if k.lower() in ("text", "query", "sentence", "utt"))
ik = next(k for k in rows[0] if k.lower() in ("intent", "label", "category", "y"))
labels = sorted({r[ik] for r in rows})
print("意图类别(%d): %s" % (len(labels), labels))

def embed(txts):
    out = []
    for i in range(0, len(txts), 32):
        body = json.dumps({"model": "dmeta-small", "input": txts[i:i+32]}).encode()
        req = urllib.request.Request(EMB, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            js = json.loads(r.read())
        out += [d["embedding"] for d in js["data"]]
    return np.array(out, dtype=np.float32)

random.seed(42); random.shuffle(rows)
n_val = max(80, len(rows) // 5)
val, train = rows[:n_val], rows[n_val:]
print("train=%d  val=%d" % (len(train), len(val)))
tr_t = [r[tk] for r in train]; tr_l = [r[ik] for r in train]
va_t = [r[tk] for r in val];   va_l = [r[ik] for r in val]

def norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

Etr = norm(embed(tr_t))
C, Cn = [], []
for lab in labels:
    idx = [i for i, l in enumerate(tr_l) if l == lab]
    if idx:
        C.append(norm(Etr[idx].mean(0, keepdims=True))[0]); Cn.append(lab)
Cm = np.stack(C)
V = norm(embed(va_t))
sims = V @ Cm.T
pred_b = [Cn[i] for i in sims.argmax(1)]
conf_b = sims.max(1)

sys.path.insert(0, str(HERE))
pred_a = []
try:
    from model_runner import IntentModel
    m = IntentModel(model_dir=str(HERE / "model"), verbose=False)
    for t in va_t:
        try:
            pred_a.append(m.predict_text(t)["intent"])
        except Exception:
            pred_a.append("ERR")
except Exception as e:
    print("  现有模型加载失败:", type(e).__name__, str(e)[:80])

def acc(ps):
    return sum(1 for p, g in zip(ps, va_l) if p == g) / max(1, len(va_l))

print("\n  === 总准确率（val %d 条）===" % len(va_l))
if pred_a:
    print("  A 现有：词嵌入+8分类线性头   acc = %.4f" % acc(pred_a))
print("  B 新：dmeta 句向量 + 原型质心  acc = %.4f" % acc(pred_b))
print("\n  逐类召回（n / A / B）:")
for lab in labels:
    idx = [i for i, g in enumerate(va_l) if g == lab]
    if not idx: continue
    ra = sum(1 for i in idx if pred_a and pred_a[i] == lab) / len(idx) if pred_a else float("nan")
    rb = sum(1 for i in idx if pred_b[i] == lab) / len(idx)
    print("    %-16s n=%-4d A=%.2f  B=%.2f" % (lab, len(idx), ra, rb))
print("\n  B 的低置信样本（cos<0.45，最该交给规则/兜底）:")
low = [(va_t[i][:26], va_l[i], pred_b[i], round(float(conf_b[i]), 3)) for i in range(len(va_t)) if conf_b[i] < 0.45]
for t, g, p, c in low[:8]:
    print("    cos=%.3f  真=%-12s 判=%-12s 「%s」" % (c, g, p, t))
print("    低置信占比: %.1f%%" % (100.0 * len(low) / len(va_t)))
