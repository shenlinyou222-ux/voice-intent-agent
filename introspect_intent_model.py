# -*- coding: utf-8 -*-
"""意图模型的组件账单（干净版）：每个组件多少参数、多少 MB、干什么用。

量化模块的 state_dict 里是 _packed_params 元组，不能按张量统计 ——
改成【把每个子对象单独 torch.save 到内存，量真实字节数】。
"""
import io
import json
from pathlib import Path

import torch

HERE = Path(__file__).parent
MD = HERE / "model"


def obj_bytes(o):
    b = io.BytesIO()
    torch.save(o, b)
    return b.tell()


def main():
    print("=" * 78)
    print("① 线上产物 intent_int8b.pt —— 按子对象量真实字节")
    print("=" * 78)
    ck = torch.load(str(MD / "intent_int8b.pt"), map_location="cpu",
                    weights_only=False)
    total = 0
    for k in sorted(ck.keys()):
        v = ck[k]
        if hasattr(v, "numel") or hasattr(v, "state_dict"):
            n = obj_bytes(v)
            total += n
            kind = "量化模块" if hasattr(v, "state_dict") else "张量"
            shape = tuple(v.shape) if hasattr(v, "shape") else ""
            print(f"  {k:16s} {kind:8s} {shape!s:24s} {n/1e6:8.1f} MB")
    print(f"  {'（其余为标量/词表）':16s} {'':8s} {'':24s} "
          f"{(MD/'intent_int8b.pt').stat().st_size/1e6:8.1f} MB  ← 文件总大小")
    print(f"  子对象合计 {total/1e6:.1f} MB")

    print()
    print("=" * 78)
    print("② fp32 原始包 bundle.safetensors —— 各组件的参数量（干净形状）")
    print("=" * 78)
    from safetensors import safe_open
    groups = {}
    with safe_open(str(MD / "bundle.safetensors"), framework="pt") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            n = 1
            for d in sl.get_shape():
                n *= d
            if k.startswith("audio_tower"):
                g = "A 音频塔 (Qwen3-ASR audio_tower)"
            elif k.startswith("multi_modal_projector"):
                g = "A 投影层 (proj1/proj2)"
            elif k.startswith("bert"):
                g = "B dmeta 编码器 (12 层)"
            elif k.startswith("language_model") or k.startswith("model."):
                g = "ASR 语言模型(解码器) 的权重"
            elif k.startswith("ctc"):
                g = "ctc_head 自训 CTC 头"
            else:
                g = "其它"
            gs = groups.setdefault(g, [0, 0])
            gs[0] += n
            gs[1] += 1
    tot = 0
    for g, (n, c) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
        print(f"  {g:38s} {n/1e6:8.2f}M 参数  fp32≈{n*4/1e6:7.1f} MB ({c} 张量)")
        tot += n
    print(f"  {'合计':38s} {tot/1e6:8.2f}M 参数  fp32≈{tot*4/1e6:7.1f} MB")

    print()
    print("=" * 78)
    print("③ 小件 intent.pt（head / ctc_head / 词表）")
    print("=" * 78)
    ck2 = torch.load(str(MD / "intent.pt"), map_location="cpu", weights_only=False)
    for k, v in ck2.items():
        if hasattr(v, "shape"):
            print(f"  {k:16s} {tuple(v.shape)!s:22s} "
                  f"{v.numel()*v.element_size()/1e6:7.2f} MB")
        elif isinstance(v, list):
            print(f"  {k:16s} list[{len(v)}]")
        else:
            print(f"  {k:16s} {v}")

    print()
    print("=" * 78)
    print("④ ASR 侧 GGUF（解码器 + 音频塔各在哪）")
    print("=" * 78)
    GD = Path(r"C:\Users\user\Desktop\ds harness\aigc_demo_framework\models\qwen3-asr-gguf")
    for f in sorted(GD.glob("*.gguf")):
        role = ("解码器(LLM)" if "Qwen3-ASR-0.6B" in f.name and "mmproj" not in f.name
                else "音频塔+投影(mmproj)" if "mmproj" in f.name else "?")
        print(f"  {f.name:46s} {f.stat().st_size/1e6:7.1f} MB  {role}")


if __name__ == "__main__":
    main()
