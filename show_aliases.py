# -*- coding: utf-8 -*-
"""看一眼闭集条目的别名结构（纠错层要用）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
d = json.loads(Path("intent_config.json").read_text(encoding="utf-8"))
for g in ("apps", "urls", "files"):
    print(f"--- {g} ---")
    for e in d[g]:
        print(f"  {e['name']:16s} aliases={e.get('aliases')}")
