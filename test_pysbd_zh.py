# -*- coding: utf-8 -*-
"""业界现成轮子 vs 我手写的分句：在真实日志句子上对比。

目的：回答"这是分句问题，业界标准做法是啥" —— 先把现成轮子（pysbd）拉进来，
在同一批真实转写上跑，用数据决定用哪个/怎么组合，而不是凭感觉手写。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, split_clauses            # noqa: E402

CASES = [
    "打开知乎，我们看一下吧。",
    "启动我们《DeepSeek Harmony》的讲解版PDF讲解版，开始我们一天的学习。",
    "打开计算器，然后打开知乎",
    "打开记事本打开哔哩哔哩",
    "启动Steam，顺便搜一下resnet50",
    "搜索文件季度报告",
    "精确查找测试报告",
    "帮我打开微信",
    "现在打开远程桌面。",
    "启动我们的启动我们的GLM吧。",
]


def show_pkg():
    import pysbd
    d = os.path.dirname(pysbd.__file__)
    print(f"pysbd {getattr(pysbd, '__version__', '?')} @ {d}")
    for root, dirs, files in os.walk(d):
        rel = os.path.relpath(root, d)
        if rel == ".":
            print("   顶层:", sorted(f for f in files if f.endswith('.py')))
        else:
            print(f"   {rel}/:", sorted(files))


def main():
    try:
        import pysbd
    except Exception as e:
        print(f"没装 pysbd: {e}")
        return 1
    show_pkg()

    cfg = Config("intent_config.json")
    seg = None
    for lang in ("zh", "en"):
        try:
            seg = pysbd.Segmenter(language=lang, clean=False)
            print(f"\n可用语言: {lang}（Segmenter 构造成功）")
            break
        except Exception as e:
            print(f"语言 {lang} 不可用: {type(e).__name__}: {e}")
    if seg is None:
        return 1

    print("\n" + "=" * 78)
    print(f"{'原句':<44} {'pysbd':<12} {'手写':<10} 一致?")
    print("=" * 78)
    same = 0
    for t in CASES:
        try:
            p = seg.segment(t)
        except Exception as e:
            p = [f"<err {type(e).__name__}>"]
        m = split_clauses(t, cfg)
        eq = len(p) == len(m)
        same += eq
        print(f"{t:<44} {len(p):<12} {len(m):<10} {'✅' if eq else '❌'}")
        if not eq:
            print(f"    pysbd : {p}")
            print(f"    手写  : {m}")
    print(f"\n段数一致 {same}/{len(CASES)}")
    print("\n注：pysbd 是按【标点】切句子（SBD），它不做")
    print("    「然后/顺便」这类无标点顺承词的切分，也不认识触发词——")
    print("    这两层本来就得自己补，业界也是这么分层的。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
