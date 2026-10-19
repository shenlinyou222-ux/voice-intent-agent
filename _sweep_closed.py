# -*- coding: utf-8 -*-
"""83 别名完备性横扫 —— 现实口径版(2026-10-18)。

★ ASR 转写只会给出汉字:说话人念拉丁名,转写出来的是其权威中文读法的
  谐音字(deepseek→"迪普西克")。所以:
    · 纯拉丁别名 → 口令 = 打开 + 权威中文读法(复用 _check_translit 的映射)
    · 汉字/混合别名 → 口令 = 打开 + 别名本身
  每条口令必须命中该别名所属条目,否则记 MISS。

用法:
    E:\\voice_agent_build\\Scripts\\python.exe -X utf8 _sweep_closed.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _check_translit import CHECK  # noqa: E402
from actions import (  # noqa: E402
    Config, resolve_by_rank, closed_alias_union)

CFG = Config(Path(__file__).parent / "intent_config.json")
BY_ALIAS = {"launch_app": CFG.app_by_alias, "open_url": CFG.url_by_alias,
            "open_file": CFG.file_by_alias}


def main():
    bad, kinds = 0, {}
    for al in closed_alias_union(CFG):
        lat = al.isascii() and al.isalpha()
        if lat and al in CHECK:
            ut, kind = "打开" + CHECK[al], "拉丁→中文读法"
        elif lat:
            ut, kind = "打开" + al, "拉丁→拉丁(无读法映射,仅鲁棒性)"
        else:
            ut, kind = "打开" + al, "汉字"
        kinds[kind] = kinds.get(kind, 0) + 1
        it, ar, s1, s2 = resolve_by_rank(ut, CFG)
        ok = None
        if it is not None and ar is not None:
            ent = BY_ALIAS.get(it, {}).get(ar)
            own = BY_ALIAS.get(it, {}).get(al.lower())
            ok = ent is not None and ent is own
        if not ok:
            bad += 1
            print(f"MISS [{kind}] {ut!r:20s} (别名={al!r}) → "
                  f"{it} {ar!r} {s1:.3f}/{s2:.3f}")
    print()
    print("口径分布:", kinds)
    print(f"完备性横扫: {kinds.get('汉字', 0) + kinds.get('拉丁→中文读法', 0)}"
          f" 现实条 + {kinds.get('拉丁→拉丁(无读法映射,仅鲁棒性)', 0)} 鲁棒性条,"
          f" 缺失 {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
