# -*- coding: utf-8 -*-
"""对比「社区简体中文汉化补丁」与本地 OpenRA 引擎的 Fluent 键覆盖度。

汉化补丁只适配 playtest-20260222，我们这份是更新的 bleed（playtest-20261011）。
Fluent 的行为：**缺键不回退英文，而是把键名原样显示**（界面出现 options-xxx.yyy）。
所以必须逐文件算键差：
   · 补丁缺的键   = 打上后会看到「键名」的地方
   · 补丁多的键   = 引擎已经删掉/改名，属于陈旧文案（无害）

用法: python openra/tools/zhcn_coverage.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PATCH = Path(r"E:\OpenRA_simplified_chinese\files")
GAME = Path(r"E:\OpenRA")

# Fluent 顶层 message 键：行首的 identifier（可带 . 和 -），后面跟 = 或换行后的属性
RE_KEY = re.compile(r"^([A-Za-z][A-Za-z0-9_.\-]*)\s*=", re.M)


def keys_of(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    return set(RE_KEY.findall(text))


def main():
    payload = [p for p in PATCH.rglob("*") if p.is_file()]
    ftl = [p for p in payload if p.suffix == ".ftl"]
    print(f"汉化补丁载荷: {len(payload)} 个文件（其中 .ftl {len(ftl)} 个）\n")

    missing_total = 0
    stale_total = 0
    same_total = 0
    absent_files = []
    per_file = []

    for p in sorted(ftl):
        rel = p.relative_to(PATCH)
        target = GAME / rel
        if not target.exists():
            absent_files.append(str(rel))
            continue
        mine = keys_of(target)
        theirs = keys_of(p)
        missing = mine - theirs          # 引擎有、补丁没有 → 会显示键名
        stale = theirs - mine            # 补丁有、引擎没有 → 无用
        missing_total += len(missing)
        stale_total += len(stale)
        same_total += len(mine & theirs)
        per_file.append((str(rel), len(mine), len(theirs), len(missing), sorted(missing)))

    per_file.sort(key=lambda x: -x[3])
    print("键覆盖最好的/最差的文件（按缺键数排序，只看缺键>0 的）:")
    print(f"{'文件':<52}{'引擎键':>7}{'补丁键':>7}{'缺键':>6}")
    for rel, n_mine, n_theirs, n_missing, _ in per_file:
        if n_missing:
            print(f"{rel:<52}{n_mine:>7}{n_theirs:>7}{n_missing:>6}")

    print()
    print(f"本地引擎里没有对应文件的载荷: {len(absent_files)} 个")
    for a in absent_files[:10]:
        print(f"   - {a}")

    total_mine = same_total + missing_total
    print()
    print(f"键总计: 引擎 {total_mine} / 命中 {same_total} / 缺 {missing_total} / 陈旧 {stale_total}")
    if total_mine:
        print(f"覆盖率: {same_total / total_mine:.1%}")

    print("\n缺键最多的 5 个文件里的具体键（打上后会原样显示）:")
    for rel, _, _, n_missing, missing in per_file[:5]:
        if not n_missing:
            continue
        print(f"  {rel}: {', '.join(missing[:12])}{' …' if len(missing) > 12 else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
