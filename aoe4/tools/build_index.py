# -*- coding: utf-8 -*-
"""为 AoE4 profile 建【embedding 意图索引】（用户要的"那一套"）。

为什么要它：规则直通能覆盖「三农」这种，但口语千变万化（"来点农民""整三个工人"…），
embedding kNN 才是稳的。索引 = 把 intents.jsonl 里每条示例编码成向量，运行时取最近邻。

它做的事（和 openra/tools/build_profile_index.py 同构）：
  1. 读 profiles/<id>/intents.jsonl（意图示例语料）
  2. 调后台 embedding 服务（dmeta-small @ 172.20.71.101:30802）编码
  3. 写 index.json（model/url/intents/labels/vecs），回填 profile.json 的 index_version

用法：
    python aoe4/tools/build_index.py                 # 默认 aoe4-v1
    python aoe4/tools/build_index.py --dry           # 只校验语料，不联网
    python aoe4/tools/build_index.py --url http://.../v1/embeddings
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
AOE4_DIR = HERE.parent
PROFILES = AOE4_DIR / "profiles"
DEFAULT_URL = "http://172.20.71.101:30802/v1/embeddings"
DEFAULT_MODEL = "dmeta-small"


def embed(texts: list[str], url: str, model: str, batch: int = 32, timeout: int = 60) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"model": model, "input": list(texts[i:i + batch])}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())["data"]
        out += [d["embedding"] for d in data]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="aoe4-v1")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    prof_dir = PROFILES / a.id
    corpus = prof_dir / "intents.jsonl"
    if not corpus.exists():
        print(f"没有语料: {corpus}")
        return 1

    pairs: list[tuple[str, str]] = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        pairs.append((obj["text"], obj["intent"]))

    texts = [p[0] for p in pairs]
    labels = sorted({p[1] for p in pairs})
    print(f"  语料 {len(texts)} 条，意图 {len(labels)} 个: {labels}")
    for t, i in pairs[:8]:
        print(f"    {t}  ->  {i}")

    if a.dry:
        print("  --dry：不联网，校验通过 ✓")
        return 0

    try:
        vecs = embed(texts, a.url, a.model)
    except Exception as e:
        print(f"  ❌ 编码失败（embedding 服务没起？）: {type(e).__name__}: {e}")
        print(f"     服务地址: {a.url}")
        return 2

    # ⭐ 必须归一化：索引里的向量与查询向量都归一化后，点积才是余弦相似度（0~1）。
    #    不归一化时相似度会到 10+，最近邻完全被向量模长主导（实测所有话都命中同一类）。
    import math
    normed = []
    for v in vecs:
        n = math.sqrt(sum(float(x) * float(x) for x in v)) or 1.0
        normed.append([float(x) / n for x in v])
    vecs = normed

    dim = len(vecs[0]) if vecs else 0
    # 注意（和 intent_embed.EmbedIntent 的约定一致）：
    #   labels  = 每条示例对应的意图（长度 == 示例数）
    #   intents = 去重后的意图名列表
    idx = {
        "model": a.model,
        "url": a.url,
        "dim": dim,
        "labels": [i for _, i in pairs],
        "intents": labels,
        "texts": texts,
        "vecs": vecs,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    blob = json.dumps({"texts": texts, "labels": labels}, ensure_ascii=False, sort_keys=True)
    version = f"{a.id}-idx-{len(texts)}x{dim}-{hashlib.sha1(blob.encode()).hexdigest()[:12]}"
    idx["index_version"] = version
    (prof_dir / "index.json").write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")

    pj = prof_dir / "profile.json"
    if pj.exists():
        raw = json.loads(pj.read_text(encoding="utf-8"))
        raw["index_version"] = version
        pj.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  ✅ 索引写好了: {prof_dir / 'index.json'}  ({dim} 维, {len(texts)} 条)")
    print(f"     index_version = {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
