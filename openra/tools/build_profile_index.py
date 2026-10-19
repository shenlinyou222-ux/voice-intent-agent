# -*- coding: utf-8 -*-
"""为某个 profile 建【embedding 意图索引】（= 用户说的 "后面 embedding 的重训练"）。

它做的事：
  1. 读 profiles/<id>/intents.jsonl（意图示例语料）
  2. 调后台 embedding 服务（默认 dmeta-small @ 172.20.71.101:30802）把每条示例编码
  3. 存成 kNN 索引 index.json —— 与 voice_agent/intent_embed.py 完全同格式（model/url/intents/labels/vecs）
  4. 把 index_version / 各版本戳回填进 profile.json

为什么这是"重训练"但不碰权重：
  A/B 主干（Qwen3-ASR audio_tower + 语义编码器）全程冻结，意图是靠【向量近邻】分类的。
  所以换域 = 换语料重建索引（秒级），不需要反向传播。
  索引文件自带版本戳，新旧两版可以同时存在、随时 A/B。

用法：
  python openra/tools/build_profile_index.py                       # 默认 openra-v1
  python openra/tools/build_profile_index.py --id openra-v1 --dry  # 只校验语料，不联网
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPENRA_DIR = HERE.parent
PROFILES = OPENRA_DIR / "profiles"


def embed(texts: list[str], url: str, model: str, batch: int = 32, timeout: int = 60) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"model": model, "input": list(texts[i:i + batch])}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())["data"]
        out += [d["embedding"] for d in data]
    return out


def normalize(vecs: list[list[float]]) -> list[list[float]]:
    out = []
    for v in vecs:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="openra-v1")
    ap.add_argument("--dry", action="store_true", help="只校验语料与版本，不调用 embedding 服务")
    args = ap.parse_args()

    pdir = PROFILES / args.id
    profile = json.loads((pdir / "profile.json").read_text(encoding="utf-8"))
    corpus = [json.loads(line) for line in (pdir / "intents.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    texts = [c["text"] for c in corpus]
    labels = [c["intent"] for c in corpus]
    intents: list[str] = []
    for lab in labels:
        if lab not in intents:
            intents.append(lab)

    print(f"profile      : {args.id}")
    print(f"hotwords_ver : {profile['hotwords_version']}")
    print(f"intent_ver   : {profile['intent_version']}")
    print(f"corpus       : {len(corpus)} 条 / {len(intents)} 个意图")

    if args.dry:
        print("dry-run：跳过 embedding 服务调用")
        return

    url = profile["embed"]["url"]
    model = profile["embed"]["model"]
    print(f"embedding    : {model} @ {url}")
    vecs = normalize(embed(texts, url, model))
    dim = len(vecs[0])

    built_at = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    index_version = f"{args.id}-idx-{len(corpus)}x{dim}-{profile['intent_version'].split('-')[-1]}"
    index = {
        "profile_id": args.id,
        "index_version": index_version,
        "hotwords_version": profile["hotwords_version"],
        "intent_version": profile["intent_version"],
        "built_at": built_at,
        "model": model,
        "url": url,
        "intents": intents,
        "labels": labels,
        "vecs": [[round(float(x), 5) for x in v] for v in vecs],
    }

    out = pdir / profile.get("index_file", "index.json")
    out.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

    profile["index_version"] = index_version
    profile["index_built_at"] = built_at
    profile["index_dim"] = dim
    profile["index_samples"] = len(corpus)
    (pdir / "profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"index        : {out}  ({dim} 维, {len(corpus)} 条)")
    print(f"index_ver    : {index_version}")

    # 自检：语料自己跑一遍 kNN，看是否自洽（应该接近 100%）
    hits = 0
    for i, v in enumerate(vecs):
        sims = [sum(a * b for a, b in zip(v, w)) for w in vecs]
        sims[i] = -2
        best = max(range(len(sims)), key=lambda j: sims[j])
        hits += labels[best] == labels[i]
    print(f"自检(kNN 留一) : {hits}/{len(vecs)} = {hits / len(vecs):.1%}")


if __name__ == "__main__":
    main()
