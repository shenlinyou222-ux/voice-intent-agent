# -*- coding: utf-8 -*-
"""回归测试：ASR 一定会带句末标点/语气词，参数解析必须扛得住。

现场（13:21 日志）：
    「启动电视。」  → 参数 '电视。'  在闭集里找不到 → 拒识
    「启动白色。」  → 参数 '白色。'  在闭集里找不到 → 拒识
    「启动我们的。」→ 参数 '我们的。' 在闭集里找不到 → 拒识
参数尾巴上挂一个「。」，闭集查找就必然失败 —— 这是会【误伤本来正确的命令】
的那类 bug，必须用测试钉住。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, parse, resolve, resolve_closed   # noqa: E402

CFG = Config("intent_config.json")

# (期望意图, 转写)  —— 全是"正确命令 + ASR 常见尾巴"
# ⚠️ 2026-10-22：记事本 / 腾讯视频 已被有意从闭集裁剪（见 intent_config.json），
#    原用例只会测出「闭集里没有」；这里换成仍在闭集的 任务管理器 / GitHub，
#    测试目的（句末标点/语气词不能弄脏参数）完全不变。
CASES = [
    ("launch_app", "启动任务管理器。"),
    ("launch_app", "启动任务管理器"),
    ("launch_app", "启动任务管理器！"),
    ("launch_app", "启动任务管理器，"),
    ("launch_app", "启动媒体播放器。"),
    ("launch_app", "打开任务管理器。"),
    ("launch_app", "启动Blender。"),
    ("launch_app", "启动Blender吧。"),
    ("launch_app", "启动steam。"),
    ("open_url", "打开GitHub。"),
    ("open_url", "启动GitHub。"),          # GitHub 是网址条目 → 期望 open_url
    ("open_url", "打开知乎。"),
    ("open_url", "打开B站。"),
    ("open_url", "打开b站吧。"),
    ("launch_app", "打开命令行。"),
    # ⚠️ 2026-10-22：files 闭集当前为空（密码本等条目已裁剪），
    #    原 open_file 两条只能测出「闭集里没有」，已换成 open_url 条目。
    ("launch_app", "把任务管理器打开。"),
    ("launch_app", "帮我打开微信。"),          # 微信不在闭集 → 应拒识（反向用例）
]


def main():
    bad = 0
    for intend, t in CASES:
        ri, ra = parse(t, CFG)
        r = resolve(intend, t, CFG)
        hit = resolve_closed(intend, r, CFG)
        want_hit = "微信" not in t
        ok = bool(hit) == want_hit
        if not ok:
            bad += 1
        mark = "✅" if ok else "❌"
        print(f"{mark} {t!r:20s} parse→{ri:11s} arg={ra!r:12s} "
              f"resolve→{r!r:14s} 命中={bool(hit)}")
    n = len(CASES)
    print(f"\n{n - bad}/{n} 通过")

    # ② 现场那几句：参数里不允许残留标点（拒识是对的，但日志必须是干净的）
    print("\n现场句子（都该拒识，但参数不能带标点）:")
    ODD = ["启动电视。", "启动白色。", "启动我们的。", "骑士版。",
           "可以，可以启动。", "试一试，打开吧。那个。"]
    for t in ODD:
        ri, ra = parse(t, CFG)
        # ⚠️ parse 返回 unknown 时，arg 按设计就是【原文】（日志要看清用户说了啥），
        #    那种情况不算脏参数 —— 只校验"解析出意图"的那些。
        dirty = ri != "unknown" and ra.endswith(
            ("。", "，", "！", "？", "、", "；", "：", ".", ","))
        if dirty:
            bad += 1
        print(f"{'❌ 脏参数' if dirty else '✅ 干净  '} {t!r:22s} "
              f"parse→{ri:11s} arg={ra!r}")

    print(f"\n合计 {n + len(ODD) - bad}/{n + len(ODD)} 通过")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
