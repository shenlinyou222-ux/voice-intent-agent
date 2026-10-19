# -*- coding: utf-8 -*-
"""打一个【纯文本路】的精简产物：B(int8) + word_emb + head_text。

砍掉的东西（就是用户说的"音频塔只留一个"）：
    A_module  音频塔 int8      198.0 MB   ← 砍（ASR 侧 mmproj 里有一个就够）
    ctc_head  自训 CTC 头       23.8 MB   ← 砍（软查表 bridge 没有消费者了）
    E_PAD     软查表词表        20 MB     ← 砍（同上）
保留：
    Benc_module  dmeta 编码器 int8   85.5 MB
    word_emb     (21128,768) fp32    64.9 MB
    head_text    768→8（已训好，val 94.58% 干净集）
    tok/词表      极小
→ 约 150 MB（原 375.7 MB）
"""
import io
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
MD = HERE / "model"
SRC = MD / "intent_int8b.pt"
OUT = MD / "intent_text.pt"


def obj_mb(o):
    b = io.BytesIO()
    torch.save(o, b)
    return b.tell() / 1e6


def main():
    t0 = time.time()
    print(f"读 {SRC.name}（{SRC.stat().st_size/1e6:.1f} MB）...")
    ck = torch.load(str(SRC), map_location="cpu", weights_only=False)
    print("原产物子对象:", {k: f"{obj_mb(v):.1f}MB" if (hasattr(v, 'numel') or hasattr(v, 'state_dict')) else type(v).__name__
                        for k, v in ck.items()})

    ht = torch.load(str(MD / "head_text.pt"), map_location="cpu",
                    weights_only=False)
    print(f"head_text: 768→{ht['w'].shape[0]}  val_acc={ht.get('val_acc', 0)*100:.2f}%")

    out = {
        "fmt": "text_only_v1",
        "intents": ck["intents"],
        # B：量化模块 + 它的配置/嵌入（_init_text 用 0 层壳 + 直接挂 encoder）
        "Benc_module": ck["Benc_module"],
        "bert_config": ck["bert_config"],
        "bert_emb": ck["bert_emb"],
        "word_emb": ck["word_emb"],
        # 文本头（替代原来的 1536 维 concat 头）
        "head_text_w": ht["w"],
        "head_text_b": ht["b"],
        "val_acc_text": float(ht.get("val_acc", 0.0)),
        # 老字段保留兼容（有些地方会读 tau/R）
        "tau": ck.get("tau", 0.1),
        "R": ck.get("R", 3),
        "shared_chars": ck.get("shared_chars", []),
    }
    torch.save(out, str(OUT))
    print(f"\n写出 {OUT.name}  {OUT.stat().st_size/1e6:.1f} MB"
          f"（原 {SRC.stat().st_size/1e6:.1f} MB，"
          f"省 {(SRC.stat().st_size-OUT.stat().st_size)/1e6:.1f} MB）")
    print(f"用时 {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
