# -*- coding: utf-8 -*-
r"""把社区简体中文汉化补丁【合并】进本地 OpenRA 构建（不覆盖 mod.yaml）。

为什么不能直接跑补丁自带的 install.bat：
  · 它只适配 playtest-20260222，而我们这份是更新的 bleed（playtest-20261011）
  · 它会整体覆盖 mods/*/mod.yaml，而我们的 mod.yaml 更新
    （例如我们的 ra|chrome/ingame-debug.yaml 在它那里是 common|chrome/ingame-debug.yaml，
      我们还有 load-game-browser.yaml / ingame-debug.ftl / ZoodRangmah.ttf 等新条目）
    → 直接覆盖会少加载新东西

本脚本的做法（可回滚）：
  1. 162 个翻译 .ftl + 139 个战役简报 map.ftl + installer/chrome/missions 文案：照抄覆盖
  2. mod.yaml 只做【外科手术】：Font: FreeSans/FreeSansBold/ZoodRangmah → common|SimHei.ttf
  3. 本机中文字体复制成 mods/common/SimHei.ttf
  4. ⭐ 缺键回填：引擎新增、补丁没有的键，用英文原文补进去
     （Fluent 缺键不回退英文，会把键名原样显示；实测缺 120 个键）
  5. 覆盖前全部备份到 <游戏目录>\_zhcn-backup\<时间戳>\ + backup.json

用法：
  python openra/tools/install_zhcn.py --dry-run     # 预演
  python openra/tools/install_zhcn.py               # 安装
  python openra/tools/install_zhcn.py --uninstall   # 还原最近一次
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

PATCH_DEFAULT = Path(r"E:\OpenRA_simplified_chinese\files")
GAME_DEFAULT = Path(r"E:\OpenRA")
FONT_CANDIDATES = ["simhei.ttf", "Deng.ttf", "msyh.ttc", "simsun.ttc"]
FONT_DEST_REL = Path("mods/common/SimHei.ttf")

# 只做字体替换的 mod.yaml（不整体覆盖）
MOD_YAML_FONT_SWAPS = [
    ("Font: common|FreeSans.ttf", "Font: common|SimHei.ttf"),
    ("Font: common|FreeSansBold.ttf", "Font: common|SimHei.ttf"),
    ("Font: ra|ZoodRangmah.ttf", "Font: common|SimHei.ttf"),
    ("Font: cnc|ZoodRangmah.ttf", "Font: common|SimHei.ttf"),
    ("Font: d2k|ZoodRangmah.ttf", "Font: common|SimHei.ttf"),
]

RE_MSG_START = re.compile(r"^([A-Za-z][A-Za-z0-9_.\-]*)\s*=", re.M)
BACKFILL_MARK = "# ── [zhcn-backfill] 引擎比汉化包新，下面这些键用英文原文补上，避免界面显示键名 ──"


def parse_blocks(text: str) -> dict[str, str]:
    """把 .ftl 拆成 {key: 原始块文本}（块 = 从 key 行到下一个顶层 key 之前）。"""
    blocks: dict[str, str] = {}
    lines = text.splitlines(keepends=True)
    cur_key = None
    buf: list[str] = []
    for line in lines:
        m = RE_MSG_START.match(line)
        if m:
            if cur_key:
                blocks[cur_key] = "".join(buf)
            cur_key = m.group(1)
            buf = [line]
        elif cur_key is not None:
            buf.append(line)
    if cur_key:
        blocks[cur_key] = "".join(buf)
    return blocks


def backfill(translated: str, english: str) -> tuple[str, list[str]]:
    """把英文里有、译文里没有的键补到译文末尾。返回 (新文本, 补的键列表)。"""
    zh_blocks = parse_blocks(translated)
    en_blocks = parse_blocks(english)
    missing = [k for k in en_blocks if k not in zh_blocks]
    if not missing:
        return translated, []
    out = translated
    if not out.endswith("\n"):
        out += "\n"
    out += f"\n{BACKFILL_MARK}\n"
    for k in missing:
        out += en_blocks[k]
        if not out.endswith("\n"):
            out += "\n"
    return out, missing


def find_font(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for name in FONT_CANDIDATES:
        p = Path(r"C:\Windows\Fonts") / name
        if p.exists():
            return p
    return None


def newest_backup(game: Path) -> Path | None:
    root = game / "_zhcn-backup"
    if not root.exists():
        return None
    dirs = sorted([d for d in root.iterdir() if d.is_dir()], reverse=True)
    return dirs[0] if dirs else None


def do_uninstall(game: Path) -> int:
    bdir = newest_backup(game)
    if not bdir:
        print(f"没有找到备份目录 {game / '_zhcn-backup'}")
        return 1
    meta = json.loads((bdir / "backup.json").read_text(encoding="utf-8"))
    restored = removed = 0
    for entry in meta["entries"]:
        target = game / entry["path"]
        if entry["state"] == "overwritten":
            src = bdir / entry["path"]
            if src.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                restored += 1
        elif entry["state"] == "added":
            if target.exists():
                target.unlink()
                removed += 1
    print(f"已还原 {restored} 个文件，删除 {removed} 个新增文件（备份目录 {bdir}）")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default=str(PATCH_DEFAULT))
    ap.add_argument("--game", default=str(GAME_DEFAULT))
    ap.add_argument("--font", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--no-backfill", action="store_true", help="不做缺键英文回填")
    args = ap.parse_args()

    game = Path(args.game).resolve()
    patch = Path(args.patch).resolve()

    if args.uninstall:
        return do_uninstall(game)

    if not patch.exists():
        print(f"找不到汉化包载荷目录: {patch}")
        return 1

    font = find_font(args.font)
    if not font:
        print("找不到本机中文字体（simhei/Deng/msyh/simsun），可用 --font 指定")
        return 1

    payload = [p for p in patch.rglob("*") if p.is_file()]
    plan_copy: list[Path] = []      # 直接覆盖
    plan_mod_yaml: list[Path] = []  # 只做字体替换
    for p in payload:
        rel = p.relative_to(patch)
        if rel.name == "mod.yaml":
            plan_mod_yaml.append(rel)
        else:
            plan_copy.append(rel)

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = game / "_zhcn-backup" / stamp
    entries: list[dict] = []
    backfilled_total = 0
    backfilled_files = 0
    missing_targets: list[str] = []

    print(f"汉化包: {patch}")
    print(f"游戏目录: {game}")
    print(f"字体: {font}  ->  {FONT_DEST_REL}")
    print(f"载荷: 直接覆盖 {len(plan_copy)} 个 + 外科手术 mod.yaml {len(plan_mod_yaml)} 个")
    print(f"备份: {backup_dir}")
    if args.dry_run:
        print("（--dry-run：不写任何文件）")

    if not args.dry_run:
        backup_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 直接覆盖的载荷 ──
    for rel in plan_copy:
        src = patch / rel
        dst = game / rel
        state = "overwritten" if dst.exists() else "added"
        if state == "overwritten" and not args.dry_run:
            bdst = backup_dir / rel
            bdst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, bdst)

        text = src.read_bytes()
        if rel.suffix == ".ftl" and dst.exists() and not args.no_backfill:
            zh = src.read_text(encoding="utf-8", errors="replace")
            en = dst.read_text(encoding="utf-8", errors="replace")
            zh2, added = backfill(zh, en)
            if added:
                backfilled_total += len(added)
                backfilled_files += 1
                text = zh2.encode("utf-8")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(text)
        entries.append({"path": str(rel).replace("\\", "/"), "state": state})
        if not dst.parent.exists() and state == "added":
            missing_targets.append(str(rel))

    # ── 2. mod.yaml：只换字体 ──
    for rel in plan_mod_yaml:
        dst = game / rel
        if not dst.exists():
            print(f"  ! 跳过 {rel}（本地没有这个文件）")
            continue
        if not args.dry_run:
            bdst = backup_dir / rel
            bdst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, bdst)
        text = dst.read_text(encoding="utf-8")
        new = text
        changed = []
        for old, repl in MOD_YAML_FONT_SWAPS:
            if old in new:
                new = new.replace(old, repl)
                changed.append(old.split("|")[-1])
        if changed and not args.dry_run:
            dst.write_text(new, encoding="utf-8")
        entries.append({"path": str(rel).replace("\\", "/"), "state": "overwritten"})
        print(f"  mod.yaml 字体替换: {rel}  ({', '.join(changed) or '无需替换'})")

    # ── 3. 字体 ──
    font_dst = game / FONT_DEST_REL
    state = "overwritten" if font_dst.exists() else "added"
    if state == "overwritten" and not args.dry_run:
        bdst = backup_dir / FONT_DEST_REL
        bdst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(font_dst, bdst)
    if not args.dry_run:
        font_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(font, font_dst)
    entries.append({"path": str(FONT_DEST_REL).replace("\\", "/"), "state": state})

    # ── 4. 备份清单 ──
    if not args.dry_run:
        (backup_dir / "backup.json").write_text(json.dumps({
            "created": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target": str(game),
            "patch": str(patch),
            "patchVersion": "1.3.1 (upstream: playtest-20260222)",
            "font": str(font),
            "entries": entries,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"覆盖/新增文件 : {len(entries)}")
    print(f"缺键英文回填 : {backfilled_total} 个键（分布在 {backfilled_files} 个文件）")
    print(f"字体          : {font.name} -> {FONT_DEST_REL}")
    print(f"备份          : {backup_dir}")
    print()
    print("还原命令: python openra/tools/install_zhcn.py --uninstall")
    return 0


if __name__ == "__main__":
    sys.exit(main())
