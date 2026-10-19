# -*- coding: utf-8 -*-
"""按住说话（PTT）的按键检测 —— 只用 GetAsyncKeyState 轮询，不装全局钩子。

为什么不用 WH_KEYBOARD_LL 钩子（钩子能"吞键"，本可以做到"短按=空格、长按=说话"）：
  · 右 Ctrl / 鼠标侧键本来就不产生字符，没有任何东西需要吞；
  · 钩子必须自己起消息循环，而且要过滤 LLKHF_INJECTED
    —— app 自己就是用 SendInput 打字的，不过滤就会【自己触发自己】；
  · 轮询零依赖、无管理员要求、任何线程都能调。

⚠️ 空格不能走这条路：它在中文输入法里是"上屏候选"，
   吞掉 = 按住期间打不了字；不吞 = 按住就往输入框灌空格。
"""
import collections
import ctypes

import numpy as np

SR = 16000                       # 和 voice_agent.SAMPLE_RATE / vad_silero.SR 一致
FRAME_MS = 20                    # 两头裁剪用的帧长

VK = {
    "RCTRL": 0xA3, "RCONTROL": 0xA3,
    "LCTRL": 0xA2, "RALT": 0xA5, "LALT": 0xA4,
    "CAPSLOCK": 0x14, "SPACE": 0x20,
    "MBUTTON": 0x04, "XBUTTON1": 0x05, "XBUTTON2": 0x06,
    "F8": 0x77, "F9": 0x78, "F10": 0x79,
}

_User32 = ctypes.windll.user32


class HoldKey:
    """按住检测：poll() 返回 'down' / 'up' / None

    confirm：连续 confirm 次采样一致才认（去抖）。音频块 30ms 一块，
    confirm=1 时最坏晚 30ms 发现按下 —— 由预滚动缓冲兜住。
    """

    def __init__(self, key="RCTRL", confirm=1):
        name = str(key).upper().strip()
        if name not in VK:
            raise ValueError(f"不认识的键 {key!r}，可选：{', '.join(sorted(VK))}")
        self.name, self.vk = name, VK[name]
        self.confirm = max(1, int(confirm))
        self._down = False
        self._n = 0

    def raw_down(self):
        return bool(_User32.GetAsyncKeyState(self.vk) & 0x8000)

    def poll(self):
        r = self.raw_down()
        if r == self._down:
            self._n = 0
            return None
        self._n += 1
        if self._n < self.confirm:
            return None
        self._down, self._n = r, 0
        return "down" if r else "up"


# ══════════════════════════════════════════════════════════════
#  段头/段尾裁剪 + 增益（PTT 门用）
# ══════════════════════════════════════════════════════════════
def frame_env(x, h=None):
    """20ms 帧 RMS 包络"""
    h = h or int(SR * FRAME_MS / 1000)
    n = len(x) // h
    return np.sqrt((x[:n * h].reshape(n, h) ** 2).mean(axis=1)) if n else np.zeros(0)


def trim_edges(x, head_keep_ms=150, tail_keep_ms=300, ratio=6.0, floor=0.0008):
    """两头都裁：段头只留「首个语音帧前 head_keep_ms」，段尾只留「末个语音帧后 tail_keep_ms」。

    · 环境底取整段帧能量的 p10 —— **不能**假设"段头就是静音"：
      按下就开口时（实测段头 20ms 处就到 47% 电平）段头就是语音，
      用段头估环境底会把阈值抬到语音之上、反过来把话裁掉。
    · 必须裁尾：Qwen3-ASR 在静音上会编词（实测 30s 近静音被转成
      「《中国好声音》第二季，其实我看是观众演唱。」）。
    · 必须留 head_keep_ms 段头：剪掉起音前的爬升会连首音素一起削掉。
    返回 (裁剪后音频, info)
    """
    h = int(SR * FRAME_MS / 1000)
    n = len(x) // h
    info = {"first_ms": -1, "last_ms": -1, "amb": 0.0, "thr": 0.0,
            "cut_head_ms": 0, "cut_tail_ms": 0}
    if n == 0:
        return x[:0], info
    e = frame_env(x, h)
    amb = float(np.percentile(e, 10))
    thr = max(amb * ratio, floor)
    info.update(amb=round(amb, 6), thr=round(thr, 6))
    idx = np.where(e >= thr)[0]
    if len(idx) == 0:
        return x[:0], info
    a, b = int(idx[0]), int(idx[-1])
    info["first_ms"], info["last_ms"] = a * FRAME_MS, b * FRAME_MS
    s = max(0, a * FRAME_MS - int(head_keep_ms))
    t = min(n * FRAME_MS, b * FRAME_MS + FRAME_MS + int(tail_keep_ms))
    info["cut_head_ms"] = s
    info["cut_tail_ms"] = max(0, (n * FRAME_MS) - t)
    return x[s // FRAME_MS * h:t // FRAME_MS * h], info


def match_engine_gain(x, target_rms=0.03, max_gain=8.0, limiter=0.95):
    """复刻 vad_silero.PreGain 的定标（target/p95，夹在 [1, max_gain]，再加 limiter）。

    为什么要复刻：现状送 ASR 的是 VAD 引擎的输出（PreGain 常年 ×8）。
    PTT 路没有引擎，若不给同样量级的增益，ASR 输入会比现在低 8 倍、
    存档电平也和历史语料对不上。实测 Qwen3-ASR 对电平不敏感
    （同一批 wav ×1 与 ×8 转写内容一致），所以这是为了"和现状一致"，不是救精度。
    """
    e = frame_env(x, int(SR * 0.02))
    ref = float(np.percentile(e, 95)) if len(e) else 0.0
    g = float(min(max(target_rms / max(ref, 1e-6), 1.0), max_gain))
    pk = float(np.abs(x).max()) if x.size else 0.0
    if pk > 1e-9 and pk * g > limiter:
        g = limiter / pk
    return np.clip(np.asarray(x, dtype=np.float32) * g, -1.0, 1.0), g


# ══════════════════════════════════════════════════════════════
#  PTT 门：可直接当 voice_agent 的 self.seg 用（同形状接口）
# ══════════════════════════════════════════════════════════════
class PttGate:
    """按住说话的门 —— 段边界由按键决定，不靠概率检测。

        段起点 = 按下时刻 − preroll_ms（环形缓冲一直在滚，与上一段无关）
        段终点 = 松开，再裁掉尾部静音（防 ASR 在静音上编词）

    血债对照（2026-10-14 实测，同一句话两条路）：
      · VAD 路 utt/162207_mic_dictation_就好像可以了.wav
        段头 0.0229/0.0280 = 段内 RMS 的 2.35×，转写「就好像可以了。」→ 吃掉「感」
        （同一句 162212 转写完整「感觉好像可以了吧。」= 两条拼接、重复一条）
      · PTT 路 162207_ptt3_raw.wav 段头 0.08×，转写「感觉好像可以了。」完整单条
    机制：VAD 收段时把 pre_buffer 清空，之后每块只补 1 块；上一段刚结束、
    紧接着又说话时预滚动不足 640ms，而触发还要等「5 块平滑 + hits」≈200~300ms
    → 段起点落到起音之后 = 从说话中间切进来。

    接口刻意与 SileroVADEngine 同形状（state/level/gain_now/last_prob/flush），
    这样 voice_agent.handle() 与电平表一行都不用改。
    """
    is_ptt = True
    frame_len = 480              # handle() 用它分"峰值帧"，与 30ms 块一致

    def __init__(self, key="RCTRL", preroll_ms=1000, min_ms=250, min_speech_ms=120,
                 head_keep_ms=150, tail_keep_ms=300, trim_ratio=6.0,
                 trim_floor=0.0008, gain=None, block_ms=30, key_obj=None):
        self.key = key_obj if key_obj is not None else HoldKey(key)
        self.key_name = getattr(self.key, "name", str(key))
        self.preroll_ms = float(preroll_ms)
        self.min_ms = float(min_ms)
        # ⚠️ 护栏必须量【按住时长】和【检测到的语音跨度】两个：
        #    量"裁过之后的段长"是错的 —— 两头各补了 head/tail 余量，
        #    误碰 100ms 会被补成 ~550ms 而漏过（自测 ⑤ 实测踩中）。
        self.min_speech_ms = float(min_speech_ms)
        self.head_keep_ms = float(head_keep_ms)
        self.tail_keep_ms = float(tail_keep_ms)
        self.trim_ratio = float(trim_ratio)
        self.trim_floor = float(trim_floor)
        self.gain_fixed = gain                      # None = 每段自适应（复刻 PreGain）
        self._gain_last = float(gain) if gain else 8.0
        self._ring = collections.deque(maxlen=max(1, int(round(preroll_ms / block_ms))))
        self._cap = None
        self._rms = collections.deque(maxlen=100)    # 最近 3s 的块 RMS（原始）
        self.n_segments = 0
        self.last_hold_ms = 0.0
        self.last_trim = {}
        # 兼容字段（日志/旧代码按名字取，缺一个就可能 AttributeError 杀掉常开进程）
        self.last_prob = 0.0
        self.last_seg_prob = 1.0                     # PTT = 用户按下的，一定是想说话
        self.threshold = 0.0
        self.rms_floor = self.rms_ceiling = 0.0
        self.voice_margin = 0.0
        self.energy_threshold = 0.0

    # ── 与 VAD 引擎同形状的接口 ──
    def set_noise_floor(self, floor):
        self._noise_floor = float(floor)             # PTT 不需要噪声底，仅接口兼容

    def reset(self):
        self._ring.clear()
        self._cap = None

    @property
    def state(self):
        return "speech" if self._cap is not None else "idle"

    @property
    def level(self):
        """电平表用：最近 3s 峰值块 RMS ×当前增益（与 VAD 路的量纲一致）"""
        return max(self._rms) * self._gain_last if self._rms else 0.0

    @property
    def gain_now(self):
        return self._gain_last

    def push(self, blk):
        """喂一块（app 是 30ms/480 点），返回收尾好的段（0 或 1 个）"""
        x = np.asarray(blk, dtype=np.float32).ravel()
        self._rms.append(float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0)
        out = []
        ev = self.key.poll()
        if ev == "down":
            self._cap = list(self._ring) + [x]        # 带上按下之前的预滚动
        elif ev == "up" and self._cap is not None:
            self._cap.append(x)
            seg = self._finish()
            if seg is not None:
                out.append(seg)
        elif self._cap is not None:
            self._cap.append(x)
        else:
            self._ring.append(x)
        return out

    def flush(self):
        """退出时若还按着，把当前这段收尾（不让它丢）"""
        return self._finish() if self._cap is not None else None

    def _finish(self):
        raw = np.concatenate(self._cap) if self._cap else None
        self._cap = None
        if raw is None or raw.size == 0:
            return None
        self.last_hold_ms = len(raw) / SR * 1000
        seg, info = trim_edges(raw, self.head_keep_ms, self.tail_keep_ms,
                               self.trim_ratio, self.trim_floor)
        info["hold_ms"] = round(self.last_hold_ms)
        speech_ms = (info["last_ms"] - info["first_ms"]) if info["first_ms"] >= 0 else 0
        info["speech_ms"] = speech_ms
        if self.last_hold_ms < self.min_ms:
            info["dropped"] = f"按住太短 {self.last_hold_ms:.0f}<{self.min_ms:.0f}ms"
            self.last_trim = info
            return None
        if speech_ms < self.min_speech_ms:
            info["dropped"] = f"没检测到足够语音 {speech_ms}<{self.min_speech_ms:.0f}ms"
            self.last_trim = info
            return None
        if self.gain_fixed is None:
            seg, g = match_engine_gain(seg)
        else:
            g = float(self.gain_fixed)
            pk = float(np.abs(seg).max()) if seg.size else 0.0
            if pk > 1e-9 and pk * g > 0.95:
                g = 0.95 / pk
            seg = np.clip(seg * g, -1.0, 1.0).astype(np.float32)
        self._gain_last = g
        info["gain"] = round(float(g), 2)
        self.last_trim = info
        self.n_segments += 1
        return seg


if __name__ == "__main__":
    import time
    k = HoldKey("RCTRL")
    print("  按右 Ctrl 试（Ctrl+C 退出）：")
    try:
        while True:
            ev = k.poll()
            if ev:
                print(f"    {time.strftime('%H:%M:%S')}  {ev}   raw={k.raw_down()}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
