# -*- coding: utf-8 -*-
"""规则层回归：文件搜索 vs 网页搜索的边界（端到端 18 条错例钉住）。

现场：这些句子只有 web_search 触发词，但宾语明确是文件 →
      原来全判 web_search（标签是 file_search）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, parse          # noqa: E402

CFG = Config("intent_config.json")

CASES = [
    # 漏网的（应判 file_search）
    ("找一下文件推理加速", "file_search"),
    ("找一下文件量化方法", "file_search"),
    ("搜索配置文件这个文件", "file_search"),
    ("搜索日志文件这个文件", "file_search"),
    ("找一下文件python教程", "file_search"),
    ("搜索git命令这个文件", "file_search"),
    ("搜索文件季度报告", "file_search"),          # 连写触发词，本来就对
    ("查找文件", "file_search"),
    # 不能误判成 file_search 的
    ("搜索python教程", "web_search"),
    ("查查日志文件。", "web_search"),             # 「日志文件」是搜索词，不是文件宾语
    ("搜索量化方法", "web_search"),
    ("打开文件管理器", "launch_app"),
    ("打开密码本文件", "open_file"),
    ("精确查找测试报告", "exact_search"),
    ("打开知乎", "open_url"),
    ("帮我搜一下文件git命令", "file_search"),
]


def main():
    bad = 0
    for text, want in CASES:
        got, arg = parse(text, CFG)
        ok = got == want
        if not ok:
            bad += 1
        print(f"{'✅' if ok else '❌'} {text!r:24s} → {got:12s} "
              f"参数={arg!r:16s} 期望={want}")
    n = len(CASES)
    print(f"\n{n - bad}/{n} 通过")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
