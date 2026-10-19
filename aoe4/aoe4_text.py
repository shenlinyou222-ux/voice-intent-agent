# -*- coding: utf-8 -*-
"""AoE4 语音意图的【文本模式】测试/驱动：绕过麦克风与 ASR，直接把一句话走完整条链路。

用法:
    python aoe4_text.py "造两个农民"          # 真执行（写命令文件）
    python aoe4_text.py "造两个农民" --dry     # 只看意图，不执行
    python aoe4_text.py --list                # 列出可用动作
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import actions  # noqa: E402
from actions import Config, dispatch, parse  # noqa: E402

CFG = ROOT / "aoe4" / "profiles" / "aoe4-v1" / "intent_config.aoe4.json"


def main() -> int:
    args = [a for a in sys.argv[1:]]
    dry = "--dry" in args
    args = [a for a in args if a != "--dry"]

    cfg = Config(str(CFG))

    if "--list" in args:
        print("AoE4 已注册动作:")
        for k in sorted(actions.ACTIONS):
            if k.startswith("aoe4"):
                print("   ", k)
        return 0

    text = " ".join(args).strip()
    if not text:
        print(__doc__)
        return 0

    hit = parse(text, cfg)
    print(f"  输入: {text}")
    print(f"  意图: {hit}")
    if not hit:
        print("  （没命中 —— 交给模型/拒识）")
        return 1
    intent, arg = hit
    ok, msg, info = dispatch(intent, arg, cfg, dry=dry)
    print(f"  执行: ok={ok}  {msg}")
    if info:
        print(f"  详情: {info}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
