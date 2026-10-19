# -*- coding: utf-8 -*-
"""句向量意图分类（业界 semantic-router / kNN-NLU 的等价实现）。

设计要点：
  · 零训练：加意图 = 加示例 -> 重建索引（毫秒级），**不碰模型权重**，不用重训
  · kNN 而非质心：同一意图有多种说法（模态），质心会把它们平均掉（实测 0.913 -> 0.964）
  · 开集检测用【阈值】：低于 thr 判 unknown（Rasa FallbackClassifier 的 nlu_threshold）
  · 置信度用【最近 vs 次近不同标签的间隔】（Rasa ambiguity_threshold）
  · 向量来源必须与生产同源（集群 vllm-embed / dmeta-embedding-zh-small）
"""
import json
import time
import urllib.request

import numpy as np

DEFAULT_URL = "http://172.20.71.101:30802/v1/embeddings"
DEFAULT_MODEL = "dmeta-small"

# 失败冷却：embedding 服务不可用时，不要每次都卡满超时（实测会卡 21s 并刷屏）
_FAIL_UNTIL = 0.0
_FAIL_COOLDOWN_S = 60.0


class EmbedIntent:
    def __init__(self, url=DEFAULT_URL, model=DEFAULT_MODEL, index=None,
                 thr=0.30, margin=0.0, timeout=10):
        self.url, self.model = url, model
        self.thr, self.margin, self.timeout = float(thr), float(margin), timeout
        self.vecs = None            # (N, 768) 已归一化
        self.labels = []            # (N,)
        self.intents = []           # 去重后的意图名
        self.last_ms = 0.0
        self.last_conf = 0.0
        if index:
            self.load(index)

    # ── 向量 ──
    def embed(self, texts, batch=32):
        global _FAIL_UNTIL
        if time.time() < _FAIL_UNTIL:
            raise RuntimeError("embedding 服务冷却中（上次连接失败，%.0fs 后重试）"
                               % (_FAIL_UNTIL - time.time()))
        out = []
        for i in range(0, len(texts), batch):
            body = json.dumps({"model": self.model, "input": list(texts[i:i + batch])}).encode()
            req = urllib.request.Request(self.url, data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out += [d["embedding"] for d in json.loads(r.read())["data"]]
            except Exception:
                _FAIL_UNTIL = time.time() + _FAIL_COOLDOWN_S
                raise
        a = np.asarray(out, dtype=np.float32)
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

    # ── 索引 ──
    def build(self, pairs):
        """pairs: [(text, intent), ...]  -> 建索引并返回是否成功"""
        texts = [t for t, _ in pairs]
        self.labels = [i for _, i in pairs]
        self.vecs = self.embed(texts)
        seen = []
        for i in self.labels:
            if i not in seen:
                seen.append(i)
        self.intents = seen
        return True

    def save(self, path, ndigits=5):
        d = {"model": self.model, "url": self.url, "intents": self.intents,
             "labels": self.labels,
             "vecs": [[round(float(x), ndigits) for x in v] for v in self.vecs]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        return path

    def load(self, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.model, self.url = d.get("model", self.model), d.get("url", self.url)
        self.intents, self.labels = d["intents"], d["labels"]
        self.vecs = np.asarray(d["vecs"], dtype=np.float32)
        return self

    # ── 预测 ──
    def predict(self, text):
        import time
        t0 = time.perf_counter()
        v = self.embed([text])[0]
        sims = self.vecs @ v
        order = np.argsort(sims)[::-1]
        best_i = int(order[0])
        best, best_lab = float(sims[best_i]), self.labels[best_i]
        second = 0.0
        for j in order[1:]:
            if self.labels[int(j)] != best_lab:
                second = float(sims[int(j)])
                break
        self.last_ms = (time.perf_counter() - t0) * 1000
        self.last_conf = best
        # ⭐ 开集：太低或与次近标签太接近 -> unknown（不是命令）
        if best < self.thr or (best - second) < self.margin:
            return {"intent": "unknown", "conf": best, "margin": best - second,
                    "raw": best_lab, "ms": self.last_ms}
        return {"intent": best_lab, "conf": best, "margin": best - second,
                "raw": best_lab, "ms": self.last_ms}
