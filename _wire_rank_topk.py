# -*- coding: utf-8 -*-
"""「保证不漏」接线路冒烟:resolve_by_rank(v2 统一标注+模板树,2026-10-18)。

★ 现实口径:ASR 转写只会给出**汉字**(说话人说 deepseek,转写就是
  "迪普西克"这类谐音字)。所以主电池全部用【中文谐音转写】口令 ——
  别名侧的拉丁名走"权威中文读法"白名单转成同一拼音音节空间去对齐。

三区:
  正向  = 真实中文转写应命中(动作/词缀/噪声槽由结构消化)。
  拉丁残余 = ASR 极偶尔保留的拉丁词(AI/APP/错拼)——仅鲁棒性兼容,非主口径。
  反向  = 方案 4.4 的 11 条 + 口语噪声,绝不能达到采纳门槛。

用法:
    E:\\voice_agent_build\\Scripts\\python.exe -X utf8 _wire_rank_topk.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from actions import Config, resolve_by_rank  # noqa: E402

CFG = Config(Path(__file__).parent / "intent_config.json")

# (转写, 期望意图 or None, 期望别名 or None)
POS = [
    # ── 知乎(真机错例 zhihu==zhihu 直呼)──
    ("打开直呼", "open_url"),
    ("打开直呼，我们看一下吧。", "open_url"),   # 跳语气词 6 音节
    ("打开直呼网页版", "open_url"),            # 词缀点名,模板消费词缀
    ("打开指乎", "open_url"),                  # 知→指 谐音字
    ("打开指乎网站", "open_url"),
    # ── DeepSeek = 迪普西克(拉丁名按权威中文读法对齐,转写全中文)──
    ("打开迪普西克", "launch_app"),
    ("启动迪普西克", "launch_app"),
    ("打开地普西克", "launch_app"),            # 迪→地 谐音字
    ("打开迪普细壳", "launch_app"),            # 西克→细壳 谐音字
    ("打开迪普西克网页版", "open_url"),         # 词缀纠察:网页版→deepseek网页
    ("打开迪普西克网页", "open_url"),
    # ── DVC Harness = 迪威西·哈尼斯(真机转写)──
    ("启动迪威西哈尼斯的。", "launch_app"),
    ("打开哈尼斯", "launch_app"),
    # ── Clash / Ollama / GitHub / Gitee ──
    ("打开克拉什", "launch_app"),
    ("打开油田欧拉玛", "launch_app"),          # 真机:前缀噪声"油田"+欧拉玛
    ("打开欧拉玛", "launch_app"),
    ("打开吉特哈布", "open_url"),              # github=吉特哈布
    ("打开麻云", "open_url"),                  # gitee=码云,码→麻 谐音字
    # ── youtube = 油管 / +网页版;同条目合并 ──
    ("打开油管", "open_url"),
    ("打开油管网页版", "open_url"),
    ("打开优图网页版", "open_url"),            # 优图=拉丁读法变体写出的汉字
    # ── qwen 家族(真应用三条目)+维基百科+豆包 ──
    ("打开千问", "launch_app"),
    ("打开千问办公", "launch_app"),            # 族内容词"办公"
    ("打开通义千问", "launch_app"),
    ("打开通义办公", "launch_app"),
    ("打开维基百科", "open_url"),              # 维基+百科 族字面别名
    ("打开豆包", "open_url"),
    ("打开豆包爱", "open_url"),                # 豆包ai,爱=ai 同音节
    ("打开edge浏览器", "launch_app"),          # edge+浏览器 族
]
# 设计内拒判(不算失败):
#   · 打开modelcope    → ASR 稀偶发保留拉丁的错拼残余;音素层管拼音不管
#     字符拼写,整链由 resolve() 的 _snap(编辑距离)兜(test_asr_correct ✓)。
#   · 打开莫德勒口坡    → 中文插入音节跨脱靶线(斯扣→口坡),模板消费不了 →
#     不猜,拒判交确认/老路 —— 反向安全优先。
ABS = ["打开modelcope", "打开莫德勒口坡"]
NEG = [
    "启动电视。", "启动白色。", "启动我们的。", "启动骑士版。",
    "打开可以可以启动。", "试一试打开吧那个", "帮我打开微信。",
    "嗯。", "今天天气不错", "我们继续下一步", "这个东西怎么弄",
    "打开油田网页版", "打开股沟学术",          # 网页版点名但无音节骨架/假别名谐音
]
# 拉丁残余(ASR 极偶尔保留的拉丁词;仅鲁棒性兼容,非主口径)
LEG = [
    "启动deepseek", "打开DeepSeek", "打开DeepSeek网页版",
    "打开youtube", "打开dockerhub", "打开豆包ai",
]


def main():
    bad = 0
    print("── 正向（应被 top-K 救回）──")
    for t, want_intent in POS:
        it, ar, s1, s2 = resolve_by_rank(t, CFG)
        ok = (it == want_intent) and ar is not None
        bad += (not ok)
        print(f"{'✅' if ok else '❌'} {t!r:24s} → {it} {ar!r:16s}"
              f"  分数 {s1:.3f} / 次 {s2:.3f}")
    print("── 拉丁残余（ASR 极偶发保留拉丁词，仅鲁棒性兼容）──")
    for t in LEG:
        it, ar, s1, s2 = resolve_by_rank(t, CFG)
        ok = it is not None
        bad += (not ok)
        print(f"{'✅' if ok else '❌'} {t!r:24s} → {it} {ar!r:16s}"
              f"  分数 {s1:.3f} / 次 {s2:.3f}")
    print("── 设计内拒判（老路已兜，top-K 保持不动）──")
    for t in ABS:
        it, ar, s1, s2 = resolve_by_rank(t, CFG)
        ok = it is None
        bad += (not ok)
        print(f"{'✅' if ok else '❌'} {t!r:24s} → 拒判(最高 {s1:.3f} / 次 {s2:.3f})"
              if ok else
              f"❌ {t!r:24s} → 误命中 {it} {ar!r}  分数 {s1:.3f}")
    print("── 反向（必须 None，一个都不许猜）──")
    for t in NEG:
        it, ar, s1, s2 = resolve_by_rank(t, CFG)
        ok = it is None
        bad += (not ok)
        print(f"{'✅' if ok else '❌'} {t!r:24s} → 拒判(最高 {s1:.3f} / 次 {s2:.3f})"
              if ok else
              f"❌ {t!r:24s} → 误命中 {it} {ar!r}  分数 {s1:.3f} / 次 {s2:.3f}")
    n = len(POS) + len(NEG) + len(LEG)
    print(f"\n{n - bad}/{n} 通过")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
