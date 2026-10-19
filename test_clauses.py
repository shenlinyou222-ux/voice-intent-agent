# -*- coding: utf-8 -*-
"""分句回归测试：真实日志里的句子必须按"意图边界"切开。

用例全部来自本次运行的 run_src.log（原始转写）+ 手工边界用例。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, split_clauses          # noqa: E402

CFG = Config("intent_config.json")

# (原始转写, 期望分句)
CASES = [
    # —— 真实日志里的句子 ——
    ("打开知乎，我们看一下吧。", ["打开知乎", "我们看一下吧"]),
    ("打开直呼，我们看一下吧。", ["打开直呼", "我们看一下吧"]),
    ("启动我们《DeepSeek Harmony》的讲解版PDF讲解版，开始我们一天的学习。",
     ["启动我们《DeepSeek Harmony》的讲解版PDF讲解版", "开始我们一天的学习"]),
    ("可以，可以启动。", ["可以", "可以启动"]),
    # 结巴重复（ASR 把「启动我们的」重复了两遍）→ 切开反而更好：
    # 前半段取不到闭集参数会被丢掉，后半段「启动我们的GLM吧」能解析出 GLM
    ("启动我们的启动我们的GLM吧。", ["启动我们的", "启动我们的GLM吧"]),
    # —— 多命令（这是多意图真正要解决的问题）——
    ("打开计算器，然后打开知乎", ["打开计算器", "然后打开知乎"]),
    ("打开记事本打开哔哩哔哩", ["打开记事本", "打开哔哩哔哩"]),
    ("启动Steam，顺便搜一下resnet50",
     ["启动Steam", "顺便搜一下resnet50"]),
    # —— 不能误切 ——
    ("搜索文件季度报告", ["搜索文件季度报告"]),
    ("精确查找测试报告", ["精确查找测试报告"]),
    ("打开知乎", ["打开知乎"]),
    ("帮我打开微信", ["帮我打开微信"]),
    ("", []),
]


def main():
    bad = 0
    for raw, want in CASES:
        got = split_clauses(raw, CFG)
        ok = got == want
        if not ok:
            bad += 1
        print(f"{'✅' if ok else '❌'} {raw!r}")
        print(f"     得到 {got}")
        if not ok:
            print(f"     期望 {want}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} 通过")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
