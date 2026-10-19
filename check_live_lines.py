# -*- coding: utf-8 -*-
"""日志里那些真实转写 → parse 判定（含同音容错）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, parse, resolve     # noqa: E402

cfg = Config("intent_config.json")
LINES = [
    ("新建绘画。", "想说「新建会话」"),
    ("DeepSeek新建绘画。", "「DeepSeek 新建会话」"),
    ("Edge新建绘画。", "「Edge 新建会话」"),
    ("新建会话，跑一下测试", "带任务的正确说法"),
    ("让DSH跑一下测试", ""),
    ("启动一下DSH", "「启动 DSH」"),
    ("请切换回画。", "「切换会话」（未实现）"),
    ("时间回放。", "?"),
]
print(f"{'转写':<22}{'→ 意图':<12}{'参数':<20}备注")
print("-" * 74)
for t, note in LINES:
    ri, ra = parse(t, cfg)
    # 走一遍 handle 里用的取参数逻辑（dsh 用 parse 的 arg）
    if ri in ("dsh_run", "dsh_new"):
        arg = ra
    else:
        arg = resolve(ri, t, cfg) if ri != "unknown" else ""
    print(f"{t!r:<22}{ri:<12}{arg!r:<20}{note}")
