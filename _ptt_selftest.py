# -*- coding: utf-8 -*-
"""PttGate 离线自测 —— 用脚本化的假按键喂合成流，断言段边界。

不需要人参与：`FakeKey` 替掉真键盘，按"第几块按下/松开"注入边沿。
覆盖的都是实测踩过的坑：
  ① 按下就开口（段头 20ms 处就到 47% 电平）→ 两头裁剪不能把起音削掉
  ② 先开口再按下（比按键早 ~300ms 说话）→ 预滚动必须够长（v1 demo 只有 300ms → 缺 300ms）
  ③ 说完立刻松手 → 段尾余量必须够（实测挖掉 500ms 会掉「录进去」）
  ④ 全程没按键 → 一个段都不能产出（"不要一直监听"）
  ⑤ 误碰一下（100ms）→ 丢弃
  ⑥ 段与段之间的预滚动必须【一直在滚】，与上一段无关（VAD 路的血债就是这里被 clear）
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from ptt import PttGate, SR, trim_edges  # noqa: E402

BLOCK = 480
rng = np.random.default_rng(11)


class FakeKey:
    """pos = 第几块时返回 'down'/'up'"""
    name = "FAKE"

    def __init__(self, edges):
        self.edges = dict(edges)
        self.i = 0

    def poll(self):
        ev = self.edges.get(self.i)
        self.i += 1
        return ev


def noise(n, level=0.0006):
    return (rng.normal(0, 1, n).astype(np.float32)) * (level / 1.0)


def speech(n, level=0.0030):
    """用噪声包一层 200Hz 调制，冒充有起伏的语音（只为能量判据服务）"""
    t = np.arange(n) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t)
    return (rng.normal(0, 1, n).astype(np.float32) * env) * (level / 0.6)


def build(speech_at, total_s=12.0, level=0.0030, amb=0.0006):
    """在 speech_at=(起秒, 止秒) 处放语音，其余放环境噪声"""
    n = int(total_s * SR)
    x = noise(n, amb)
    a, b = int(speech_at[0] * SR), int(speech_at[1] * SR)
    x[a:b] = speech(b - a, level)
    return x


def run(x, edges, **kw):
    """把流按 30ms 块喂进门；edges 用【块号】指定按下/松开"""
    g = PttGate(key_obj=FakeKey(edges), **kw)
    got = []
    for i in range(0, len(x) - BLOCK + 1, BLOCK):
        got += g.push(x[i:i + BLOCK])
    return g, got


def overlaps(seg, x, at, tol_ms=20):
    """段里有没有覆盖到 at=(起秒,止秒) 的语音（用互相关能量比近似：直接查该段是否含该区间）"""
    # 段是从 x 里切出来的连续片段，用归一化互相关峰值定位
    a = np.asarray(seg, dtype=np.float64)
    a = a - a.mean()
    best, bestlag = -1e9, 0
    step = 160                                    # 10ms 步进，够用
    for lag in range(0, len(x) - len(seg) + 1, step):
        b = x[lag:lag + len(seg)].astype(np.float64)
        b = b - b.mean()
        d = (np.linalg.norm(a) * np.linalg.norm(b))
        if d <= 0:
            continue
        c = float(np.dot(a, b) / d)
        if c > best:
            best, bestlag = c, lag
    off = bestlag / SR
    return off, best


def check(name, cond, detail=""):
    print(f"    {'✅' if cond else '❌'} {name}" + (f"   {detail}" if detail else ""))
    return bool(cond)


def main():
    ok = True
    print("\n  ══ PttGate 离线自测 ══")
    kw = dict(preroll_ms=1000, head_keep_ms=150, tail_keep_ms=300, min_ms=250)

    # ① 按下就开口：语音 5.00~7.00s，5.00s 按下（0.3 块精度），7.00s 松开
    x = build((5.0, 7.0))
    g, got = run(x, {int(5.0 / 0.03): "down", int(7.0 / 0.03): "up"}, **kw)
    print(f"\n  ① 按下就开口（语音 5.00~7.00s，按键 5.00/7.00s）")
    ok &= check("切出 1 段", len(got) == 1, f"实得 {len(got)}")
    if got:
        off, c = overlaps(got[0], x, (5.0, 7.0))
        print(f"       段长 {len(got[0])/SR:.2f}s  对齐偏移 {off:.2f}s  相关 {c:.3f}")
        ok &= check("段起点 ≤ 5.00s（含 1000ms 预滚动）", off <= 5.01, f"偏移 {off:.2f}s")
        ok &= check("段终点 ≥ 6.99s（语音尾部完整）",
                    off + len(got[0]) / SR >= 6.99, f"终点 {off + len(got[0])/SR:.2f}s")
        ok &= check("没有把起音削掉（段头裁到语音前 150ms 内）", off <= 4.86, f"偏移 {off:.2f}s")

    # ② 先开口再按下：语音 4.60~6.50s，5.00s 才按下 → 预滚动 1000ms 必须兜住
    x = build((4.6, 6.5))
    g, got = run(x, {int(5.0 / 0.03): "down", int(6.5 / 0.03): "up"}, **kw)
    print(f"\n  ② 先开口再按下（语音从 4.60s 起，5.00s 才按下 = 早说 400ms）")
    ok &= check("切出 1 段", len(got) == 1, f"实得 {len(got)}")
    if got:
        off, c = overlaps(got[0], x, (4.6, 6.5))
        print(f"       段长 {len(got[0])/SR:.2f}s  对齐偏移 {off:.2f}s  相关 {c:.3f}")
        ok &= check("段起点 ≤ 4.60s（早说的那 400ms 一个字都没丢）", off <= 4.61, f"偏移 {off:.2f}s")
    # 对照：预滚动只有 300ms 时必然丢（复现 v1 demo 的缺口）
    g2, got2 = run(x, {int(5.0 / 0.03): "down", int(6.5 / 0.03): "up"},
                   **{**kw, "preroll_ms": 300})
    if got2:
        off2, _ = overlaps(got2[0], x, (4.6, 6.5))
        print(f"       对照（预滚动 300ms）：对齐偏移 {off2:.2f}s → "
              f"丢了 {(off2-4.6)*1000:.0f}ms 语音")
        ok &= check("对照确实会丢（证明这个旋钮是必要的）", off2 > 4.61, f"{off2:.2f}s")

    # ③ 说完立刻松手：语音 5.00~6.90s，6.90s 松开
    x = build((5.0, 6.9))
    g, got = run(x, {int(5.0 / 0.03): "down", int(6.9 / 0.03): "up"}, **kw)
    print(f"\n  ③ 说完立刻松手（语音到 6.90s，6.90s 松开）")
    ok &= check("切出 1 段", len(got) == 1, f"实得 {len(got)}")
    if got:
        off, _ = overlaps(got[0], x, (5.0, 6.9))
        print(f"       段长 {len(got[0])/SR:.2f}s  对齐偏移 {off:.2f}s  "
              f"结尾覆盖到 {off + len(got[0])/SR:.2f}s")
        ok &= check("结尾没被裁掉（覆盖到 ≥6.89s）", off + len(got[0]) / SR >= 6.89)
        # ⚠️ 松手即停时音频就结束在松手那一刻：尾部【没有】300ms 可留。
        #    所以"末个语音帧后 300ms 余量"只在松手后仍录到静音时才吃得到。
        #    反过来也说明：裁尾必须是"有静音才裁"，不能无条件砍掉 X ms。
        ok &= check("尾部余量被松手时刻封顶（不越界）",
                    off + len(got[0]) / SR <= 7.0, f"{off + len(got[0])/SR:.2f}s")

    # ④ 全程没按键（环境噪声里还有一段"说话"）→ 一个段都不许出
    x = build((5.0, 7.0), level=0.01)             # 响亮的语音
    g, got = run(x, {}, **kw)
    print(f"\n  ④ 全程不按键（但环境里有一句响亮的话）")
    ok &= check("0 段（不按就不听）", len(got) == 0, f"实得 {len(got)}")

    # ⑤ 误碰 100ms
    x = build((5.0, 5.1))
    g, got = run(x, {int(5.0 / 0.03): "down", int(5.1 / 0.03): "up"}, **kw)
    print(f"\n  ⑤ 误碰 100ms")
    ok &= check("丢弃（<250ms）", len(got) == 0, f"实得 {len(got)}")

    # ⑥ 上一段刚结束就再说（预滚动必须一直在滚）
    n = int(12 * SR)
    x = noise(n, 0.0006)
    x[int(3.0 * SR):int(4.6 * SR)] = speech(int(1.6 * SR), 0.003)     # 第一句
    x[int(6.0 * SR):int(7.6 * SR)] = speech(int(1.6 * SR), 0.003)     # 第二句（隔 1.4s）
    edges = {int(3.6 / 0.03): "down", int(4.6 / 0.03): "up",      # 第一句：按下时已经在说
             int(6.0 / 0.03): "down", int(7.6 / 0.03): "up"}
    g, got = run(x, edges, **kw)
    print(f"\n  ⑥ 上一段刚结束后紧接着再说（两段，各带 1 个早开口）")
    ok &= check("切出 2 段", len(got) == 2, f"实得 {len(got)}")
    for k, seg in enumerate(got):
        off, c = overlaps(seg, x, (0, 0))
        print(f"       段{k}: {len(seg)/SR:.2f}s  对齐偏移 {off:.2f}s  相关 {c:.3f}")
    if len(got) == 2:
        o0, _ = overlaps(got[0], x, (0, 0))
        o1, _ = overlaps(got[1], x, (0, 0))
        ok &= check("第一段起点 ≤3.00s（早说 600ms 也在）", o0 <= 3.01, f"{o0:.2f}s")
        ok &= check("第二段起点 ≤6.00s（没被上一段影响）", o1 <= 6.01, f"{o1:.2f}s")

    # ⑦ trim_edges 的纯函数行为
    print(f"\n  ⑦ trim_edges 纯函数")
    seg, info = trim_edges(np.concatenate([noise(int(0.5 * SR)), speech(int(1.0 * SR)),
                                           noise(int(1.0 * SR))]), 150, 300)
    print(f"       info={info}")
    ok &= check("裁掉头 500ms→留 150ms", 330 <= info["cut_head_ms"] <= 400,
                f"cut_head={info['cut_head_ms']}ms")
    ok &= check("裁掉尾 1000ms→留 300ms", 680 <= info["cut_tail_ms"] <= 720,
                f"cut_tail={info['cut_tail_ms']}ms")

    print(f"\n  {'🎉 全部通过' if ok else '⚠️ 有失败项'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
