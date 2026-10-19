# -*- coding: utf-8 -*-
"""ASR 错例纠错回归测试（同音字 / 英文词级）。

用例全部来自真实日志（run_src.log）+ 必须【不能误命中】的反向用例。
反向用例尤其重要：开错程序比不开更糟（历史上「帮我打开微信」被 snap 成微博）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, resolve, resolve_closed   # noqa: E402

CFG = Config("intent_config.json")

# (意图, 转写, 期望命中名 或 None)
CASES = [
    # ── 现场错例：必须救回来 ──
    ("open_url", "打开直呼", "知乎"),                    # zhihu == zhihu
    ("open_url", "打开直呼，我们看一下吧。", "知乎"),
    ("launch_app", "启动DVC Harness的。", "DeepSeek"),    # 词元 harness
    ("launch_app", "启动deepseek", "DeepSeek"),
    # ⚠️ 2026-10-17 定案：删除 2 条过时用例「打开记事板/启动记事板 → 记事本」
    #    —— 记事本条目已从闭集移除（intent_config.json _comment_pruned），
    #    新接线的 resolve_by_rank 对「启动记事板」top-K 最高仅 0.5 → 拒识，安全。
    # ⚠️ 这一条的意图必须用 open_url：模型会把「打开modelcope」判成 launch_app，
    #    但规则 parse 在三个闭集里比优劣、全串命中 modelscope → open_url，
    #    线上有【规则优先】把它改回 open_url/ModelScope（见 handle()）。
    #    这里直接按规则给的意图调 resolve，才是线上真实路径。
    ("open_url", "打开modelcope", "ModelScope"),          # 真实日志原文，防回归
    # ── 反向用例：绝对不能命中（否则会乱开程序）──
    ("launch_app", "启动电视。", None),
    ("launch_app", "启动白色。", None),
    ("launch_app", "启动我们的。", None),
    ("launch_app", "启动骑士版。", None),
    ("open_url", "打开可以可以启动。", None),
    ("launch_app", "试一试打开吧那个", None),
    ("launch_app", "帮我打开微信。", None),               # 微信不在闭集（且不许snap成微博）
    ("launch_app", "嗯。", None),
]


def main():
    bad = 0
    for intent, text, want in CASES:
        arg = resolve(intent, text, CFG)
        hit = resolve_closed(intent, arg, CFG)
        name = hit["name"] if hit else None
        ok = (name == want)
        if not ok:
            bad += 1
        print(f"{'✅' if ok else '❌'} {text!r:26s} → 参数={arg!r:14s} "
              f"命中={name!r:12s} 期望={want!r}")
    n = len(CASES)
    print(f"\n{n - bad}/{n} 通过")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
