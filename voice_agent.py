# -*- coding: utf-8 -*-
"""语音意图识别 —— 常开流式监听 → 意图识别 → 自动执行
（意图开关：intent_config.json → intent_registry，每条 enabled；改完自动热加载）

用法：
    voice_agent.exe                     常开监听麦克风（Ctrl+C 或关窗口退出）
    voice_agent.exe --list              只列出所有可执行命令
    voice_agent.exe --dry-run           不真的执行，只打印会做什么
    voice_agent.exe --wav a.wav b.wav   不用麦克风，直接跑音频文件（测试用）
    voice_agent.exe --config xxx.json   换配置文件
    voice_agent.exe --device 2          指定麦克风设备号

架构（V5 集成）：
    麦克风 → VAD 分句 → [A(18层) → ctc_head → V1帧级混合 + V3分词查表 → B → 意图头]
           → 分句规则解析参数 → Python 动作
"""
from urllib.parse import urlparse
import argparse
import json
import os
import re
import signal
import sys
import time
import traceback
from pathlib import Path

# ⚠️ 冻结（PyInstaller）后 stdout 默认是 Windows 本地编码（cp1252），
#    打印中文会直接 UnicodeEncodeError 崩掉（实测 --list 就死在这）。
#    必须强制 UTF-8。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ⚠️ 2026-10-21：日志落盘不要再走 Windows PowerShell 5.1 的管道
#   （`& python ... | Tee-Object` / `ForEach-Object` 会用 OEM 代码页解码子进程
#    的 UTF-8 输出，所有中文变成 '?'，日志直接没法看）。
#   改为进程内 tee：stdout 照常进控制台（句柄直连，显示正常），同时由 Python
#   自己按 UTF-8 追加到 VOICE_AGENT_TEE 指定的文件。
_tee_path = os.environ.get("VOICE_AGENT_TEE")
if _tee_path:
    class _Tee:
        def __init__(self, stream, path):
            self._stream = stream
            self._f = None
            try:
                self._f = open(path, "a", encoding="utf-8", buffering=1)
            except Exception:
                pass

        def write(self, s):
            if self._f is not None:
                try:
                    self._f.write(s)
                except Exception:
                    pass
            return self._stream.write(s)

        def flush(self):
            for f in (self._f, self._stream):
                if f is not None:
                    try:
                        f.flush()
                    except Exception:
                        pass

        def __getattr__(self, name):
            return getattr(self._stream, name)

    sys.stdout = _Tee(sys.stdout, _tee_path)

# ⭐ 冻结（PyInstaller）后 __file__ 指向 _internal\，不是 exe 所在目录，
#   所以配置/模型必须按 sys.executable 找。不处理的话会报
#   FileNotFoundError: ...\dist\voice_agent\_internal\intent_config.json
if getattr(sys, "frozen", False):
    HERE = Path(sys.executable).resolve().parent
else:
    HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np

from actions import (Config, parse, parse_multi, resolve, resolve_closed, dispatch,
                     resolve_by_rank,
                     list_commands, ACTIONS)
from vad import Segmenter

SAMPLE_RATE = 16000
BLOCK = 480          # 30ms

def _disp_width(s: str) -> int:
    """显示宽度：CJK 算 2 列，其余算 1（用来把横幅的右边框对齐）。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _banner_box(lines, width: int = 58) -> str:
    out = ["   ╔" + "═" * width + "╗"]
    for ln in lines:
        pad = max(0, width - 3 - _disp_width(ln))
        out.append("   ║   " + ln + " " * pad + "║")
    out.append("   ╚" + "═" * width + "╝")
    return "\n".join(out)


# [OPENCODE-A2] 产品改名：帝国时代4 语音指挥 → **语音意图识别**
#   横幅改成"产品名 + 当前意图域"；具体哪些意图参与匹配由 intent_config.json
#   的 intent_registry（每条 enabled）决定，改完自动热加载。
_BANNERS = {
    "desktop": [
        "语音意图识别  ——  常开监听（直接说话，不用按键）",
        "意图域：office ｜ 可说：打开B站 / 启动Blender",
    ],
    "ops": [
        "语音意图识别  ——  常开监听（直接说话，不用按键）",
        "可说：查磁盘 / 查集群 / 重启nginx（会先问一遍）",
        "只读直通 · 变更两步确认 · 危险动词不进闭集",
    ],
    "aoe4": [
        "语音意图识别  ——  常开监听（直接说话，不用按键）",
        # [OPENCODE-Z] 横幅写"现在能说的话"（拼音路由细节只在 debug 里出现）
        "可说：打开B站、命令列表、切换领域（游戏/运维/日常）",
        "说「语音校准」说一句话即可校准；非命令话按听写开关处理",
    ],
    "openra": [
        "语音意图识别  ——  意图域：openra（按住右 Ctrl）",
        "可说：建造 / 放置 / 编队 / 探索（见 openra/语音指令表.md）",
    ],
}


def banner_for(cfg) -> str:
    """按当前 profile 的域选横幅（desktop / ops / aoe4 / openra）。"""
    try:
        from actions import ops_mode, aoe4_mode
        if ops_mode(cfg):
            return _banner_box(_BANNERS["ops"])
        if aoe4_mode(cfg):
            return _banner_box(_BANNERS["aoe4"])
        if cfg.behavior.get("openra_mode"):
            return _banner_box(_BANNERS["openra"])
    except Exception:
        pass
    return _banner_box(_BANNERS["desktop"])


# 日志标签：统一走 dlog（线程安全、TAG 对齐、emoji 自动转 TAG）
from dlog import log                              # noqa: E402


# ══════════════════════════════════════════════════════════════
#  「像不像人话」校验 —— 挡掉噪声被识别成的乱码
#
#  实测踩过的坑：麦克风收到噪声时，模型会输出
#      「核核核核」「t核核mbo核核核核」「核thdbu核plpicu核u」
#  这类乱码，而且【置信度高达 0.78~0.99】—— 光靠 conf_threshold 挡不住。
#  更糟的是它触发了 web_search → 浏览器播放声音 → 麦克风又听到 → 正反馈，
#  段长从 1s 一路涨到 15s（撞上 max_speech_ms）。
#
#  三个判据（都不需要模型，纯统计）：
#    ① 长度：命令都很短，>30 字一定不是
#    ② 重复率：同一个字占比 >45% = 乱码（「核核核核」）
#    ③ 中英交替密度：汉字/字母每两个字就翻一次 = 乱码（「t核核mbo核核」）
# ══════════════════════════════════════════════════════════════


def plausible(text, cfg=None, max_len=30, max_repeat=0.45, max_flip=0.40):
    """判断转写像不像一句人话

    四个判据（前三个纯统计，第四个查词表）：
      ① 长度：命令都很短，>30 字一定不是
      ② 重复率：同一个字占比 >45% = 乱码（「核核核核」）
      ③ 中英交替密度：汉字/字母每两个字就翻一次 = 乱码
      ④ ⭐ 必须含【至少一个已知词】（触发词 或 闭集名字/别名）
         这是最强的一条：真实命令必然含「打开/搜索/启动/精确」或某个闭集名字，
         乱码一个都不会有。
         实测 ①②③ 都挡不住的 '核thdbu核plpicu核u' / 'pabntubh核'，
         ④ 一次全挡掉。
    """
    from collections import Counter
    import re as _re
    t = (text or "").strip()
    if len(t) < 2:
        return False, f"太短（{len(t)} 字）"
    if len(t) > max_len:
        return False, f"太长（{len(t)} 字 > {max_len}，命令不会这么长）"
    top, n = Counter(t).most_common(1)[0]
    if n / len(t) > max_repeat:
        return False, f"「{top}」重复率 {n/len(t)*100:.0f}% > {max_repeat*100:.0f}%"
    flips = len(_re.findall(r"[a-z][\u4e00-\u9fff]|[\u4e00-\u9fff][a-z]", t))
    if flips / len(t) > max_flip:
        return False, f"中英交替 {flips} 次 / {len(t)} 字，过密"
    if cfg is not None:
        known = set()
        for v in cfg.triggers.values():
            known.update(v)
        for group in (cfg.apps, cfg.urls, cfg.files):
            for e in group:
                known.add(e["name"])
                known.update(e.get("aliases", []))
        # ⭐ 游戏模式：把版本化 profile 的热词/单位别名也算「已知词」，
        #    否则「两个矿车」这种没触发词的口令会被这条判据丢掉（实测踩过）
        try:
            import actions as _A
            known.update(_A._game_known_words(cfg))
        except Exception:
            pass
        tl = t.lower()
        if not any(k and k.lower() in tl for k in known):
            # ⚠️ 措辞修正：这句话【不是】乱码判据，而是"不是命令"判据。
            #    实测踩中：「那么现在做这个意图识别，怎么做呢？」转写完全正确，
            #    只是没有触发词/闭集名字 → 被判"一定是识别乱码"，日志误导。
            #    行为（拒识）是对的，说明必须准确。
            return False, "没有触发词/闭集名字（不是命令，或识别错了）"
    return True, ""


def _akey(intent, arg, resolved=None):
    """流式对账用的动作键：优先用解析出的闭集 id（DeepSeek/Deepseek 是同一个），
    否则用归一化后的参数 —— 否则同一动作会被算成三个（实测踩过）。"""
    base = resolved if resolved else arg
    return (intent, re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(base).lower()))


# ══════════════════════════════════════════════════════════════
class Agent:
    def __init__(self, cfg: Config, model_dir=None, device="cpu", dry=None):
        self.cfg = cfg
        self.dry = cfg.behavior.get("dry_run", False) if dry is None else dry
        self.model = None
        self.model_dir = model_dir
        self.device = device
        b = cfg.behavior
        self.seg = Segmenter(
            silence_ms=b.get("vad_silence_ms", 500),
            min_speech_ms=b.get("vad_min_speech_ms", 200),
            energy_threshold=b.get("vad_energy_threshold", 0.0025),
            max_speech_ms=b.get("vad_max_speech_ms", 12000),
            aggressiveness=b.get("vad_aggressiveness", 1),
            use_webrtcvad=b.get("use_webrtcvad", False),
            min_run_ms=b.get("vad_min_run_ms", 150),
            floor_mult=b.get("vad_floor_mult", 1.5),
            ceil_mult=b.get("vad_ceil_mult", 8.0),
            voice_margin=b.get("vad_voice_margin", 2.0),
            noise_floor=b.get("vad_noise_floor", 0.0008),
            noise_down=b.get("vad_noise_down", 0.25),
            noise_up=b.get("vad_noise_up", 0.002),
        )
        # 分句后端：silero = 标准神经网络 VAD + 前置 AGC（业界标准链路）
        #           energy = 旧的手调阈值（保留作回退/对照）
        #           ptt    = 不判 VAD：段边界由【按住右 Ctrl】决定（见下面 trigger_mode）
        self.vad_backend = str(b.get("vad_backend", "energy")).lower()
        # ⭐⭐ 触发方式：ptt（按住说话）| vad（常开监听，旧行为）
        #   ptt 的理由（2026-10-14 实测，同一句话两条路）：
        #     VAD 路的段起点靠"概率检测 + 预滚动"，而 StateMachine 收段时把 pre_buffer
        #     清空、之后每块只补 1 块 —— 上一段刚收完、紧接着又说话时，预滚动不足，
        #     触发又要等「5 块平滑 + hits」≈200~300ms → 段起点落到起音【之后】。
        #     血债：utt/162207_mic_dictation_就好像可以了.wav 段头 = 段内 RMS 的 2.35×
        #     （整段最响就在开头），转写「就好像可以了。」吃掉「感」；同一句话 PTT 段
        #     段头 0.08×、转写「感觉好像可以了。」完整。
        #   ptt 顺带解决：不按就不听（噪声/对话不再切段、不再往输入框打）、
        #     松手即送 ASR（省掉 704ms 静音等待，实测 142~155ms 出字）。
        #   回滚：intent_config.json 里 trigger_mode 改回 "vad"（一行）。
        self.trigger_mode = str(b.get("trigger_mode", "vad")).lower()
        if self.trigger_mode == "ptt":
            from ptt import PttGate
            self.vad_backend = "ptt"
            self.seg = PttGate(
                key=b.get("ptt_key", "RCTRL"),
                preroll_ms=b.get("ptt_preroll_ms", 1000),
                min_ms=b.get("ptt_min_ms", 250),
                min_speech_ms=b.get("ptt_min_speech_ms", 120),
                head_keep_ms=b.get("ptt_head_keep_ms", 150),
                tail_keep_ms=b.get("ptt_tail_keep_ms", 300),
            )
        elif self.vad_backend == "silero":
            from vad_silero import VADFactory
            self.seg = VADFactory.get_vad_engine(
                "silero_vad",
                target_rms=float(b.get("vad_target_rms", 0.05)),
                max_gain=float(b.get("vad_max_gain", 80.0)),
                prob_threshold=float(b.get("vad_prob_threshold", 0.3)),
                db_threshold=float(b.get("vad_db_threshold", 40)),
                required_hits=int(b.get("vad_required_hits", 1)),
                required_misses=int(b.get("vad_required_misses", 16)),
                smoothing_window=int(b.get("vad_smoothing_window", 2)),   # ★ 平滑窗（越大越稳但越慢）
                max_speech_ms=int(b.get("vad_max_speech_ms", 12000)),
                pre_roll_ms=float(b.get("vad_preroll_ms", 640)),          # ★ 预滚（段起点往前带多少音频）
            )
        # ASR 后端：ctc = 自训 ctc_head 贪心解码；own = Qwen3-ASR 自己的解码器
        self.asr_backend = b.get("asr_backend", "ctc")
        # ⭐ 意图来源：text（默认，纯文本路：ASR 文本 → dmeta embedding → head_text）
        #            audio（旧路，concat(音频塔 c1, 文本 c2) → 1536 头）
        #    砍掉音频塔后 intent_from 必须是 text；只有加载了完整产物才能用 audio。
        self.intent_from = str(b.get("intent_from", "text")).lower()
        self.quantize_int8 = bool(b.get("quantize_int8", False))
        self.asr = None
        # 动态加载状态（平时不占内存，说话才装，闲久了还回去）
        self.lazy = bool(b.get("lazy_load", False))
        # ⚠️ 空闲多久卸载模型。默认原来是 120s（2 分钟）—— 用户实测受不了：
        #    闲置一会儿再说，模型正好在重装（IntentModel 构造实测 17s，
        #    全程 19.6s），而这段时间【音频队列会满】→ 实测一次丢 244 块
        #    = 7.3 秒音频，于是「说的话开头没了」，看起来像"转写不工作"。
        #    2026-10-17 用户指定：改成 500 分钟（30000s），等于不卸。
        #    各 profile 的 behavior.idle_unload_s 也同步改了。
        self.idle_unload_s = float(b.get("idle_unload_s", 30000))
        self.preload_on_onset = bool(b.get("preload_on_onset", True))
        import threading as _th
        self._load_lock = _th.Lock()
        self._loading = None
        self._last_use = time.time()
        self._loaded_once = False
        self._asr_ms = 0.0          # 最近一句的 ASR 耗时/token 数（打日志用）
        self._asr_tok = 0
        self.stats = {"heard": 0, "executed": 0, "rejected": 0, "failed": 0}

    def vad_summary(self) -> str:
        """启动时打印【实际生效】的 VAD 参数（不是配置里写了什么）。"""
        seg = self.seg
        sm = getattr(seg, "sm", None)
        _pre = float(getattr(seg, "preroll_ms", 0.0) or 0.0)
        _pre_n = int(getattr(sm, "pre_buffer_chunks", 0) or 0) if sm is not None else 0
        if sm is not None:
            gain = getattr(seg, "gain", None)
            tgt = getattr(gain, "target", None)
            gmax = getattr(gain, "max_gain", None)
            _pct = getattr(gain, "pct_cfg", None)
            return (f"backend=silero | prob {sm.prob_threshold:.2f} / db {sm.db_threshold:.0f} / "
                    f"hits {sm.required_hits} / misses {sm.required_misses}(≈{sm.required_misses * 32}ms) / "
                    f"平滑 {sm.smoothing_window} | 预滚 {_pre:.0f}ms({_pre_n}块) | "
                    f"AGC 目标 {tgt} 上限 ×{gmax}"
                    + (f" / pct {_pct:.0f}" if _pct is not None else ""))
        return (f"backend={self.vad_backend} | 静音 {getattr(seg, 'silence_frames', '?')} 帧 / "
                f"最短语音 {getattr(seg, 'min_speech_frames', '?')} 帧"
                + (f" | 预滚 {_pre:.0f}ms" if _pre else ""))

    def trim_summary(self) -> str:
        """启动时打印【实际生效】的段首裁剪参数（用户 2026-10-21 要求可见）。"""
        b = self.cfg.behavior
        frac = float(b.get("ahead_frac", 0.2))
        cap_cfg = int(b.get("ahead_cap_ms", 640))
        margin = int(b.get("onset_margin_ms", 100))
        pre = float(getattr(self.seg, "preroll_ms", 0.0) or 0.0)
        if self.vad_backend == "ptt":
            return f"PTT | 预滚 {pre:.0f}ms（按下前）"
        trim = int(b.get("ahead_trim_ms", max(0, int(pre) - cap_cfg)))
        cap = max(0, int(pre) - trim)
        return (f"VAD 预滚 {pre:.0f}ms | 头部保留 = min[{cap}ms, 总长×{frac:.2f}, "
                f"语音起点−{margin}ms] | ahead_trim {trim}ms / cap {cap_cfg}ms / "
                f"frac {frac:.2f} / onset_margin {margin}ms")

    def _apply_int8(self):
        """把【解码器 + A】动态量化成 int8。

        实测（quant_ab.py，24 条 val，文本固定）：
          A int8        → 意图 24/24 无损
          + 解码器 int8 → 意图 24/24 无损，转写去标点后一致，速度 1.83×
          + dmeta-B int8→ 意图掉到 13/24（头是在 fp32 特征上训的线性层，需重训）
        ⚠️ 只量化 Linear/Conv1d，【不动 Embedding 词表】：
           词表与 lm_head 是 tied 的同一张 fp32 表，量化 Embedding 会打断 tie，
           lm_head 继续持有旧表 → 反而多占 622 MB。
        ⚠️ A 是与意图模型共享的，必须量化它本尊再回填给解码器，否则又变两份。
        """
        import gc
        import torch
        from torch.ao.quantization import quantize_dynamic, default_dynamic_qconfig
        qlin = {torch.nn.Linear: default_dynamic_qconfig}
        qconv = {torch.nn.Linear: default_dynamic_qconfig,
                 torch.nn.Conv1d: default_dynamic_qconfig}
        t0 = time.time()
        try:
            self.asr.model.model.language_model = quantize_dynamic(
                self.asr.model.model.language_model, qlin, dtype=torch.qint8)
            qa = quantize_dynamic(self.model.A, qconv, dtype=torch.qint8)
            self.model.A = qa
            self.asr.model.model.audio_tower = qa       # 重新指向同一实例（只一份）
            gc.collect()
            log(f"🎛 int8 量化完成（解码器 Linear + A，词表保持 fp32）"
                f"  {time.time()-t0:.1f}s")
        except Exception as e:
            log(f"⚠️ int8 量化失败，保持 fp32: {type(e).__name__}: {e}")

    def _trim_working_set(self):
        """把已经空闲的页交还系统（Windows EmptyWorkingSet）。

        为什么需要：torch 的 CPU 分配器释放后不把页还给 OS —— 实测「先分配 fp32
        再换掉」和「pickle 反序列化」留下的页会一直挂在 RSS 上。这里只是把空闲页
        送进 standby（提交量不变，再次访问按需调回），所以 RSS 立刻下降。
        """
        try:
            import ctypes
            k = ctypes.WinDLL("kernel32")
            p = ctypes.WinDLL("psapi")
            k.GetCurrentProcess.restype = ctypes.c_void_p
            p.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
            p.EmptyWorkingSet.restype = ctypes.c_int
            p.EmptyWorkingSet(k.GetCurrentProcess())
            return True
        except Exception as e:
            log(f"   （工作集整理失败: {type(e).__name__}: {e}）")
            return False

    # ── 动态加载：平时不装，说话时才装，闲久了还回去 ──
    def ensure_loaded(self, where=""):
        """确保意图模型 + ASR 就绪（幂等；并发调用只装一次）

        分阶段计时：实测出现过"第一句语音等 20+ 秒"的情况，而 import/读盘本身
        只要 2 秒 —— 没有阶段计时就只能猜。这里每一段都报 ms。
        """
        with self._load_lock:
            _t = time.time()
            if self.model is None:
                self.load_model()
                log(f"   · 意图模型 {time.time()-_t:.2f}s", "SYS", sub=True)
            _t = time.time()
            if self.asr is None and self.asr_backend in ("own", "gguf", "http", "vllm"):
                self.asr = self._make_asr()
                self.asr.load()
                log(f"   · ASR({self.asr_backend}) {time.time()-_t:.2f}s",
                    "SYS", sub=True)
            _t = time.time()
        self._last_use = time.time()
        if _t and (time.time() - _t) > 0.05:
            log(f"   · 加载收尾 {time.time()-_t:.2f}s", "SYS", sub=True)

    def preload_async(self, why=""):
        """VAD 刚检测到开口 → 后台线程先把模型装上，等话说完正好就绪"""
        if self.model is not None and (self.asr is not None or self.asr_backend == "ctc"):
            return
        if self._loading is not None and self._loading.is_alive():
            return
        import threading

        def _work():
            try:
                t0 = time.time()
                self.ensure_loaded()
                log(f"   ⚡ 后台预加载完成（{why}，{time.time()-t0:.1f}s）")
            except Exception as e:
                log(f"   ⚠️ 预加载失败: {type(e).__name__}: {e}")

        self._loading = threading.Thread(target=_work, daemon=True)
        self._loading.start()
        log(f"   ⚡ 检测到说话 → 后台预加载中（{why}）")

    def unload_idle(self, idle_s):
        """闲够 idle_s 秒 → 卸载模型并把工作集交还系统"""
        if self.model is None and self.asr is None:
            return
        if time.time() - self._last_use < idle_s:
            return
        n_asr = self.asr is not None
        try:
            if self.asr is not None:
                self.asr.unload()
        except Exception:
            pass
        self.asr = None
        self.model = None
        import gc
        gc.collect()
        self._trim_working_set()
        self._loaded_once = True
        log(f"   💤 空闲 {idle_s:.0f}s → 已卸载模型（ASR={'有' if n_asr else '无'}）")

    def _hotwords(self):
        """热词上下文文本（两条 ASR 路共用；gguf 走 system 消息，http 走 prompt）"""
        if not self.cfg.behavior.get("asr_hotwords", True):
            return ""
        from actions import hotword_context
        extra = []
        try:
            import json as _j
            meta = _j.loads((Path(self.model_dir) / "meta.json")
                            .read_text(encoding="utf-8"))
            extra = list(meta.get("en_words", []))
        except Exception:
            pass
        extra = extra + list(self.cfg.behavior.get("asr_hotwords_extra", []))
        ctx = hotword_context(
            self.cfg, extra=extra,
            max_chars=int(self.cfg.behavior.get("asr_hotword_chars", 400)),
            ascii_only=bool(self.cfg.behavior.get("asr_hotword_ascii_only", False)))
        # ⚠️ ascii_only 只判断"含 ASCII"，于是「Edge浏览器」「B站」「qwen办公」也进来了 ✗
        #   用户要求：热词 prompt 只放【少量英文专有名词】-> 这里后置过滤成纯 ASCII。
        if self.cfg.behavior.get("asr_hotword_ascii_only"):
            _parts = [x for x in (ctx or "").split("、") if x and x.isascii()]
            ctx = "、".join(_parts[:40])
            log(f"   ASR 热词（纯英文 {len(_parts[:40])} 个）：{ctx[:70]}", "ASR")
        return ctx

    def _make_asr(self):
        """按 asr_backend 造 ASR：
             http = 调 app 那条 llama-server（同模型/同参数，**推荐**：零本进程开销）
             gguf = 进程内 llama.cpp 自己跑 GGUF（占 1.5GB）
             own  = torch 加载官方权重（3,130MB）
        """
        if self.asr_backend == "vllm":
            # ★ vLLM(GPU, 集群 30805)：chat/completions + input_audio（没有 transcriptions 端点 ✗）
            #   实测 ~105ms（0.6B + CUDA Graph）vs CPU 进程内 1.7B 430ms
            from asr_vllm import VLLMASR
            _eng = VLLMASR(
                url=str(self.cfg.behavior.get("asr_vllm_url")
                        or "http://127.0.0.1:30805/v1/chat/completions"),
                model=str(self.cfg.behavior.get("asr_vllm_model") or "qwen3-asr"),
                max_tokens=int(self.cfg.behavior.get("asr_vllm_max_tokens", 64)),
                timeout=float(self.cfg.behavior.get("asr_vllm_timeout", 30.0)),
                hotwords=self.cfg.behavior.get("asr_vllm_hotwords") or None,
                hotword_bias=float(self.cfg.behavior.get("asr_vllm_hotword_bias", 6.0)),
                language=str(self.cfg.behavior.get("asr_vllm_language") or "Chinese"),
                bad_words=self.cfg.behavior.get("asr_vllm_bad_words") or None)
            log(f"   ASR 后端 = vLLM(GPU) {_eng.url}", "ASR", sub=True)
            return _eng
        if self.asr_backend == "gguf":
            from asr_gguf import GGUFASR
            # ⭐ 热词【只放名词/专有名词/昵称】—— 用户明确要求：
            #   "只需要特定的昵称和专有名词…是他妈名词不要把什么乱七八糟的
            #    也当做 token 前缀树的热词表"
            #   所以：闭集条目名（都是名词）全留；别名只留 ≥3 字的（丢掉 'hf'
            #   这种过短会乱偏置的）；动词/泛词（启动/打开/搜索/文件/网址）
            #   一律不进 —— asr_hotwords_extra 整个不用了。
            nouns, seen = [], set()
            for _g in (self.cfg.apps, self.cfg.urls, self.cfg.files):
                for _e in _g:
                    _cand = [(_e.get("name") or "")] + list(_e.get("aliases") or [])
                    for _w in _cand:
                        _w = (_w or "").strip()
                        if len(_w) < 3:          # 名字短于 3 字的也留（知乎/B站）
                            if _w != (_e.get("name") or "").strip():
                                continue
                        if not _w or _w.lower() in seen:
                            continue
                        seen.add(_w.lower())
                        nouns.append(_w)
            # ⚠️ 不再用 meta.json 的 en_words 当热词来源！
            #    那是训练时的静态英文词表（notepad、calc、cmd…），里面混着
            #    用户根本用不上的东西 —— 删掉「记事本」app 条目后 notepad
            #    还会作为热词存在，纯属覆盖过多（用户："把这个傻逼notepad删掉
            #    我他妈根本用不上"）。热词【只跟随当前闭集】：配置里有什么，
            #    热词里就有什么。
            if self.cfg.behavior.get("asr_hotwords", True) and nouns:
                log(f"   ASR 热词（名词前缀树，{len(nouns)} 个）："
                    + "、".join(nouns[:18]) + "…", "ASR", sub=True)
            _eng = GGUFASR(n_threads=int(self.cfg.behavior.get("asr_threads", 4)),
                           n_ctx=int(self.cfg.behavior.get("asr_n_ctx", 512)),
                           language=self.cfg.behavior.get("asr_language", "Chinese"),
                           context=None,       # ⭐ prompt 不用了（已实测 x2.3 延迟 0 收益）
                           # ⭐ 权重可由配置切换（0.6B / 1.7B 都在本机）
                           gguf=self.cfg.behavior.get("asr_gguf") or None,
                           mmproj=self.cfg.behavior.get("asr_mmproj") or None,
                           use_audio_tags=bool(self.cfg.behavior.get(
                               "asr_audio_tags", False)))
            # 前缀树热词 + 权重（NeMo boosting_tree_alpha 的等价物）
            _eng.hotwords = (nouns if self.cfg.behavior.get("asr_hotwords", True)
                             else [])
            _eng.hotwords_score = float(self.cfg.behavior.get("hotwords_score", 2.0))
            _eng._trie = None
            return _eng
        if self.asr_backend == "http":
            # ⭐ ASR 走 HTTP：WSL k3s 里的 llama-asr（GPU 上跑量化 1.7B Q4_K_M, -ngl 99）
            #   实测 82ms/句（CPU 版 702ms），文本与 CPU 逐字一致。
            #   日志【不带 sub=True】—— 之前 sub 日志在普通模式被隐藏，害我查了很久。
            from asr_http import HTTPASR
            ctx = self._hotwords()
            _url = str(self.cfg.behavior.get("asr_http_url")
                       or "http://127.0.0.1:30803/v1/audio/transcriptions")
            if self.cfg.behavior.get("asr_http_resolve_wsl_ip") and "{wsl_ip}" in _url:
                try:
                    import subprocess as _sp
                    _r = _sp.run(["wsl", "-d", str(self.cfg.behavior.get("wsl_distro")
                                                   or "Ubuntu-24.04"),
                                  "-e", "bash", "-c", "hostname -I"],
                                 capture_output=True, text=True, timeout=25)
                    _ip = (_r.stdout or "").split()[0]
                    _url = _url.replace("{wsl_ip}", _ip)
                    log(f"   ASR 端点：WSL IP = {_ip}", "ASR")
                except Exception as _e:
                    log(f"   ⚠️ 解析 WSL IP 失败: {type(_e).__name__}: {_e}", "WARN")
            _a = HTTPASR(url=_url,
                         timeout=float(self.cfg.behavior.get("asr_http_timeout") or 120),
                         language=str(self.cfg.behavior.get("asr_http_language") or "zh"),
                         prompt=ctx,
                         send_prompt=bool(self.cfg.behavior.get("asr_http_prompt", False)))
            log(f"   ASR(http) 端点 {_url}  热词 {len(ctx or '')} 字"
                + ("（已发 prompt）" if _a.send_prompt else "（不发 prompt：实测会退化）"),
                "ASR")
            try:
                _a.load()
                log(f"   ASR(http) 健康检查 OK（{_a.load_ms:.0f}ms）", "ASR")
                self.asr = _a
                return _a
            except Exception as _e:
                log(f"   ⚠️ GPU ASR 不可达（{type(_e).__name__}: {_e}）"
                    f"→ 回退进程内 CPU GGUF", "WARN")
                self.asr_backend = "gguf"
                return self._make_asr()
        from asr_own import OwnASR
        return OwnASR(language=self.cfg.behavior.get("asr_language", "Chinese"),
                      share_encoder=getattr(self.model, "A", None))

    def load_model(self):
        if self.model is not None:
            return
        log("加载意图模型 ...")
        _t = time.time()
        import torch
        log(f"   · import torch {time.time()-_t:.2f}s", "SYS", sub=True)
        # ⚠️ 实测：不限线程时 torch/OpenMP 在推理后自旋空转 —— 常开监听空转 CPU
        #    高达 220%（单核基准，20 线程池），而小模型也用不上这么多线程。
        _t = time.time()
        nth = int(self.cfg.behavior.get("torch_threads", 4) or 1)
        torch.set_num_threads(max(1, nth))
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        log(f"torch 线程数限制为 {torch.get_num_threads()}（{time.time()-_t:.2f}s）")
        _t = time.time()
        from model_runner import IntentModel
        log(f"   · import model_runner（含 transformers）{time.time()-_t:.2f}s",
            "SYS", sub=True)
        _t = time.time()
        self.model = IntentModel(
            model_dir=self.model_dir, device=self.device,
            int8=bool(self.cfg.behavior.get("intent_int8", False)),
            text_only=bool(self.cfg.behavior.get("intent_text_only", True)))
        # ⭐ 意图分类后端：embed = 句向量 kNN（零训练：加意图只加示例+重建索引）
        #   | torch = 原词袋线性头（词嵌入+8x768）。见 intent_embed.py
        self.embed_cls = None
        if str(self.cfg.behavior.get("intent_backend", "torch")).lower() == "embed":
            try:
                from intent_embed import EmbedIntent
                from pathlib import Path as _P
                # ⚠️ 血债（2026-10-16 实测，exe 上踩的）：PyInstaller onedir 下
                #    __file__ 指向 _MEIPASS（就是 _internal\），所以 __file__.parent
                #    【不是】发行目录 —— 索引会去找
                #    _internal\ops\profiles\ops-v1\index.json（不存在）→ 句向量后端
                #    整个加载失败 → 回退词袋线性头 → 再撞 head_text 缺失 → 意图全废。
                #    冻结时用 sys.executable 的目录（= dist\voice_agent\），
                #    源码运行时仍然用 __file__（两者在源码下是同一个目录）。
                _base = (_P(sys.executable).resolve().parent
                         if getattr(sys, "frozen", False)
                         else _P(__file__).resolve().parent)
                _idx = _base / str(self.cfg.behavior.get(
                    "intent_index", "intent_index.json"))
                self.embed_cls = EmbedIntent(
                    url=str(self.cfg.behavior.get("intent_embed_url")
                            or "http://172.20.71.101:30802/v1/embeddings"),
                    index=str(_idx),
                    thr=float(self.cfg.behavior.get("intent_embed_thr") or 0.30),
                    margin=float(self.cfg.behavior.get("intent_embed_margin") or 0.0))
                log(f"   意图分类：句向量 kNN（{len(self.embed_cls.labels)} 示例 / "
                    f"{len(self.embed_cls.intents)} 类 / thr={self.embed_cls.thr}）")
            except Exception as _e:
                log(f"   ⚠️ 句向量分类器加载失败（{type(_e).__name__}: {_e}）"
                    f"→ 回退词袋线性头", "WARN")
                self.embed_cls = None
        log(f"   · IntentModel 构造 {time.time()-_t:.2f}s", "SYS", sub=True)
        log(f"模型就绪：{len(self.model.INTENTS)} 个意图")

    def _dictate_async(self, text, dur, raw=None, why=""):
        """后台线程里做听写。

        ⚠️ 必须异步：听写里要读 UIA（大文档要几百 ms 甚至更久），
           同步执行会把音频主循环卡住 → 队列爆满 → 实测丢了 362 块(≈11 秒音频)。
        """
        import threading

        def _work():
            try:
                self._dictate(text, dur, raw)
            except Exception as e:
                log(f"   ⚠️ 听写线程异常: {type(e).__name__}: {e}", "WARN")

        threading.Thread(target=_work, daemon=True, name="dictate").start()

    # ── 听写：把转写打进当前焦点输入框 ──
    def _dictate(self, text, dur, raw=None):
        # ⭐ 听写模式（语音「开始输入」/「终止输入」控制）
        #   off = 明确终止 -> 不打；on = 强制打（跳过"光标在输入框"判定）；
        #   auto = 老行为（光标在输入框才打）
        try:
            import typein as _ti
            _mode = _ti.get_mode()
        except Exception:
            _mode = "auto"
        # ⚠️⚠️ 错例存档必须排在【所有 return 之前】——血债（2026-10-14 实测）：
        #   15:56:10 / 15:56:14 两段真人语音（VAD峰概率 0.876/0.905）走
        #   unknown → _dictate_async，而 _dictate 在 `if not ok_field: return False`
        #   就返回了，_save_utt 在很后面的 `if ok:` 里 —— 结果这两段
        #   **既没有 wav、index.jsonl 也没有一行**，恰好是最该复现的错例丢了。
        #   现在：进函数先存一份（accepted=False），打成功的再补一条 accepted=True。
        _archived = {"v": False}

        def _arch_fail():
            if raw is not None and not _archived["v"]:
                _archived["v"] = True
                self._save_utt(raw, text, "dictation", 1.0, "", False, dur)

        if _mode == "off":
            # [OPENCODE-H2] 需求（用户）：关闭输入/听写后不再转写输入框，只匹配意图
            log("🔇 听写已关闭（说「开始输入」恢复）→ 这句没匹配到意图，"
                "忽略、未打进输入框", "ACT")
            _arch_fail()
            return False
        self._dictation_forced = (_mode == "on")
        """光标在可编辑控件里 → 把 text 注入进去。返回 True 表示已处理。

        ⭐ 判定用 UI Automation（浏览器/Electron 没有 Win32 caret；
           实测 Chromium 窗口 caret=0x0，只有 UIA 靠得住）。
        ⭐ 注入用 SendInput + KEYEVENTF_UNICODE（中文/emoji 都能送，不动剪贴板）。
        ⭐ 开关：behavior.enable_dictation（默认 true）。
        """
        if not self.cfg.behavior.get("enable_dictation", True):
            _arch_fail()
            return False
        t = (text or "").strip()
        if len(t) < 1:
            _arch_fail()
            return False
        try:
            from typein import in_text_field, type_text
        except Exception as e:
            log(f"   ⚠️ 听写不可用: {type(e).__name__}: {e}", "WARN")
            _arch_fail()
            return False
        try:
            ok_field, why = in_text_field()
        except Exception as e:
            log(f"   ⚠️ 光标判定失败: {type(e).__name__}: {e}", "WARN")
            _arch_fail()
            return False
        if getattr(self, "_dictation_forced", False):
            why = "强制听写模式(开始输入) -> " + str(why)
        elif not ok_field:
            # [OPENCODE-E4] 血债（2026-10-19 实测）：auto 模式下焦点不是输入框时
            #   原来【静默 return】——日志里只有「转写」，用户完全不知道为什么没打进去。
            #   这里必须留一行原因（终端/非输入框/类名），否则看起来就是"转写坏了"。
            log(f"   ⌨️ 跳过听写（auto 模式）：{why}", "ACT")
            _arch_fail()                     # 不在输入框 → 交回原来的拒识逻辑
            return False
        # [OPENCODE-H] 标点→空格（参数 behavior.dictation_punct_to_space；默认关，不改原行为）
        try:
            if self.cfg.behavior.get("dictation_punct_to_space", False):
                from opencode_fastpath.dictate_norm import punct_to_space
                t = punct_to_space(t)
        except Exception:
            pass
        if self.cfg.behavior.get("dictation_add_space", True):
            t = t + " "
        try:
            ok, msg = type_text(t, dry=bool(self.dry))
        except Exception as e:
            log(f"   ⚠️ 听写注入失败: {type(e).__name__}: {e}", "WARN")
            _arch_fail()
            return False
        # ⏱ 延迟归因（从【松手】那一刻起逐段落账）
        #   t_seg  = 松手（段与松手在同一块里出来，误差 ≤ 一个 30ms 块）
        #   t_asr1 = ASR 返回；t_int1 = 意图判定返回；LAST['t0'] = 真正调用打字函数
        #   打字函数内：焦点判定(UIA) → 读一次输入框(UIA) → 逐字注入 → 验证
        #   为什么要拆这么细：UIA 那两次读可能上百 ms，而 ASR 才一百多 ms ——
        #   只看"ASR 120ms"会把真实感知延迟低估好几倍（我自己就先信口说过 0.2s）。
        self._log_timeline("听写")
        if ok:
            log(f"🎤 「{text}」  ({dur:.2f}s)  → ⌨️ 听写进输入框（{why}）")
            # [OPENCODE-I] 显示实际注入的文本（标点→空格之后）
            try:
                if self.cfg.behavior.get("dictation_punct_to_space", False):
                    log(f"   [OPENCODE] 实际注入: 「{(t or chr(32)).strip()}」", "ASR")
            except Exception:
                pass
            log(f"   ✅ {msg}")
            self.stats["executed"] += 1
            # 记下"已打进输入框的半句"，若下一段在 gap 内到达且这句没说完 → 撤回重打
            # [OPENCODE-L] 拼接状态存【原始转写】(带标点): 标点判据才准; 注入的仍是归一化文本
            self._stitch = {"text": (text or "").strip(), "t": time.time(),
                            "dictated": True, "n_chars": len(t or ""),
                            "dur": float(dur or 0)}
            if raw is not None:
                self._save_utt(raw, text, "dictation", 1.0, "", True, dur)
            return True
        log(f"   ⚠️ {msg}", "WARN")
        _arch_fail()
        return False

    # ── 处理一句音频 ──
    def _predict_intent(self, *args, **kwargs):
        """意图判定统一入口：embed(句向量 kNN) 优先，异常/关闭则回退 torch 词袋。"""
        text = kwargs.get("text_override")
        if text is None:
            for a in args:
                if isinstance(a, str):
                    text = a
                    break
        # ★ 规则直通短路：闭集（x农）用正则就是微秒级，没必要再去打 embedding 服务
        #   （血债：正则命中了还调句向量 → 白付 47ms HTTP 往返）
        if text:
            try:
                _ri0, _ra0 = parse(text, self.cfg)
            except Exception:
                _ri0, _ra0 = "unknown", None
            if _ri0 != "unknown":
                return {"text": text, "intent": _ri0, "conf": 1.0,
                        "latency_ms": 0.0, "margin": 1.0, "src": "rule"}
        if getattr(self, "embed_cls", None) is not None and text:
            try:
                r = self.embed_cls.predict(text)
                return {"text": text, "intent": r["intent"], "conf": float(r["conf"]),
                        "latency_ms": r["ms"], "margin": r["margin"], "src": "embed"}
            except Exception as _e:
                log(f"   ⚠️ 句向量分类失败（{type(_e).__name__}: {_e}）→ 回退词袋", "WARN")
        # ⭐ 回退：走 torch 词袋线性头（model_runner.IntentModel）。
        #    这里【绝不能】调 self._predict_intent —— 那就是调用自己，会无限递归
        #    （实测：句向量一失败就 RecursionError，整句意图推理崩掉）。
        if kwargs.get("text_override") is not None or (args and not isinstance(args[0], str)):
            out = self.model.predict(*args, **kwargs)
        else:
            out = self.model.predict_text(text or "")
        if isinstance(out, dict):
            out.setdefault("src", "torch")
        return out

    def handle(self, wav):
        # ⏱ 延迟归因起点：段是【松手那一块】里同一次 push 出来的，
        #   所以这一刻 ≈ 松手时刻（误差 = 一个 30ms 块）。感知延迟从这里算。
        # [OPENCODE-W] 统一返回：一句开始（重置本句账本；收账由空闲定时器做）
        try:
            from opencode_fastpath import reply as _oc_rp2
            _oc_rp2.begin(self)
        except Exception:
            pass
        self._t_seg = time.time()
        self._asr_ms = 0.0          # 先清零：否则归因会拿上一句的 ASR 耗时
        self._asr_tok = 0
        # ⭐ 段级可观测：每个进入处理的段都留痕（时长/VAD峰概率/RMS）
        #   目的：把"没生成段"（VAD 太严 ✗）和"段被丢"（门限 ✗）与"转写错"分开。
        try:
            import numpy as _np
            _dur = len(wav) / 16000.0
            _rms = float(_np.sqrt((_np.asarray(wav, dtype=_np.float32) ** 2).mean()))
            _pk = float(getattr(self.seg, "peak_prob", 0.0) or 0.0)
            _lp = float(getattr(self.seg, "last_prob", 0.0) or 0.0)
            # PTT 没有概率可报：打「VAD峰0.000」会让人以为"VAD 判它不是语音"（误导）
            _tag = ("PTT 段" if getattr(self.seg, "is_ptt", False)
                    else (f"VAD峰{max(_pk, _lp):.3f}" if self.vad_backend in ("silero", "ptt")
                          else f"电平{self.seg.level:.4f}"))
            log(f"   📥 段进入 handle: {_dur:.2f}s  {_tag}  RMS {_rms:.5f}",
                "MIC")
        except Exception:
            pass

        # 空段防御：0 长度会让 rms0 = sqrt(nan) → NaN 传进模型/规则。
        # 正常不该出现（push/flush 都有 0.2s 下限），但不能靠"不该出现"活着。
        if len(wav) == 0:
            log("⚠️ 收到空音频段，跳过（防 NaN 传播）", "WARN")
            return
        # ⭐ 存档用的原始音频（AGC 之前）。为什么要存：现场「启动DeepSeek」被转成
        #    「启动DVC Harness的。」，但那段真实音频没留下来 → 只能拿 TTS 语料猜，
        #    修不动。存下 raw + 转写结果，才是能反复复现的错例集。
        raw = np.asarray(wav, dtype=np.float32)
        if self.lazy:
            self.ensure_loaded()        # 幂等；顺带刷新 _last_use
        self.stats["heard"] += 1
        dur = len(wav) / SAMPLE_RATE

        # 错例存档（每条出口都记一次；开关 save_utterances）
        # ⚠️ 必须定义在所有 return 之前 —— 之前放在 text 赋值之后，
        #    现在"ASR 无转写"这条出口也在前面了，会 NameError。
        _save_on = bool(self.cfg.behavior.get("save_utterances", False))

        def _arch(_i, _c, _a, _ok, _t=""):
            if _save_on:
                self._save_utt(raw, _t, _i, _c, _a, _ok, dur)

        # ⭐⭐⭐ SNR 门限（整段平均 RMS）—— 已被 vad.py 的「连续发声」判据取代。
        #      默认 snr_floor_mult=0 关闭：实测段平均 RMS 对好坏没有可分性
        #      （两句正确的 0.00055/0.00039 反而低于噪声乱码段的 0.0014~0.0058）。
        #      下面这段保留，供高信噪比环境用。
        #
        # 实测现场：本机麦克风电平极低（噪声底 RMS 0.002~0.0037，
        # 语音峰值才 0.02~0.055）。VAD 门槛一旦设到噪声底附近，
        # 它就把噪声全判成语音 → 每段都撑满 max_speech(6s) → 模型输出乱码
        # → 而且乱码里偶尔带个"搜"字 → **真的去开浏览器搜索**（已发生）。
        #
        # 所以：这一段的电平如果没比噪声底高出足够倍数，直接当噪声丢掉，
        # 不进模型、不进规则、不执行任何动作。
        # ⭐⭐ 低电平护栏（血债）：实测用户对着麦克风说话，语音峰值只有
        #   0.0039 (-48 dBFS)、RMS 0.00047 -- 比设备自噪声 0.00058 还低，
        #   信噪比 ~0 dB -> Silero 判"非人声"是【正确】的，任何算法都救不回。
        #   根因在系统侧（录音音量/麦克风加强 ~0、或设备选错/静音）。
        #   这里主动告警，免得再花几轮去怀疑代码。
        try:
            _pk = float(np.abs(wav).max())
            if _pk < 0.01:
                # 提示（不是故障）：Silero 的自适应预增益已能处理轻麦克风，
                # 实测峰值 0.006 也能准确转写 → 降为 SYS，不再刷 WARN
                log(f"   麦克风偏轻：本段峰值 {_pk:.5f}（{20*np.log10(max(_pk,1e-9)):.0f} dBFS）"
                    f" —— 自适应预增益会自动补，无需处理", "SYS", sub=True)
                self.stats["low_level"] = self.stats.get("low_level", 0) + 1
        except Exception:
            pass
        floor = getattr(self, "_noise_floor", 0.0)
        if floor > 0:
            lv = float(np.sqrt((wav ** 2).mean()))
            mult = self.cfg.behavior.get("snr_floor_mult", 4.0)
            if lv < floor * mult:
                self.stats["noise"] = self.stats.get("noise", 0) + 1
                log(f"   🔇 当噪声丢掉：RMS {lv:.5f} < 噪声底 {floor:.5f}×{mult:g}")
                return

        # ⭐⭐ 增益归一化（AGC）
        #    麦克风电平远低于模型训练时的电平（TTS 正常电平），
        #    这里把输入拉到目标 RMS，让模型看到它见过的量级。
        #    ⚠️ 实测模型本身对电平【不敏感】（衰减 40 dB 仍能转对），
        #       所以 AGC 主要是为了 VAD 和日志可读性，不是精度关键。
        tgt = self.cfg.behavior.get("agc_target_rms", 0.06)
        maxg = self.cfg.behavior.get("agc_max_gain", 60.0)
        rms0 = float(np.sqrt((wav ** 2).mean()))
        # 峰值帧 RMS：比整段平均更能代表"这一段里到底有没有人声"
        _nf = int(getattr(self.seg, "frame_len", 480) or 480)
        _fr = wav[:len(wav) // _nf * _nf].reshape(-1, _nf)
        pk = float(np.max(np.sqrt((_fr ** 2).mean(axis=1)))) if len(_fr) else rms0
        if rms0 > 1e-7:
            gain = min(maxg, tgt / rms0)
            if gain <= 1.5:
                gain = 1.0
            else:
                wav = np.clip(wav * gain, -1.0, 1.0).astype(np.float32)
            # ⚠️⚠️ 这行诊断必须与分句后端解耦（血债）：
            #   旧版直接取 self.seg.frame_len / last_run_ms / threshold，
            #   而 Silero 引擎根本没有这些"能量门槛"字段 →
            #   实测 12:29:50，第一段真人语音进 handle() 就
            #   AttributeError 穿出 run_mic，**整个进程自动退出**。
            #   现在：有能量门槛字段（老 Segmenter）才显示门槛，
            #   否则显示 Silero 的 VAD 概率 + AGC 增益。
            if hasattr(self.seg, "last_run_ms"):
                # 人话版：能量 VAD 的判据是"连续超阈值的时长"，不是概率
                _b = self.cfg.behavior
                _extra = (f"  VAD: 连续超阈值 {self.seg.last_run_ms:.0f}ms"
                          f" / 需 ≥{int(_b.get('vad_min_run_ms', 240))}ms"
                          f"（阈值 {float(_b.get('vad_energy_threshold', 0)):.4f}）")
            elif getattr(self.seg, "is_ptt", False):
                # PTT：没有概率可报，报"按住多久 + 两头裁了多少 + 增益"才有诊断价值
                _t = getattr(self.seg, "last_trim", {}) or {}
                _extra = (f"  PTT 按住 {_t.get('hold_ms', 0):.0f}ms"
                          f"（语音 {_t.get('first_ms', -1)}~{_t.get('last_ms', -1)}ms"
                          f"  裁头 {_t.get('cut_head_ms', 0):.0f}ms"
                          f"  裁尾 {_t.get('cut_tail_ms', 0):.0f}ms"
                          f"  阈值 {_t.get('thr', 0):.5f}）"
                          + (f"  ⚠️{_t['dropped']}" if _t.get("dropped") else ""))
            else:
                _sp = getattr(self.seg, "last_seg_prob", None)
                if _sp is None:
                    _sp = getattr(self.seg, "last_prob", 0.0)
                _extra = (f"  VAD峰概率 {_sp:.3f}"
                          f"  AGC×{getattr(self.seg, 'gain_now', 1.0):.1f}")
            log(f"   🎚 ×{gain:.1f}  RMS {rms0:.5f}→"
                f"{float(np.sqrt((wav ** 2).mean())):.5f}"
                f"  峰值帧 {pk:.5f}" + _extra)

        # ⭐⭐ 意图来源：text（默认）| audio（旧路）
        #    text  = ASR 文本 → dmeta embedding → head_text   ← 音频塔已砍，只留 mmproj 那一个
        #    audio = 旧的 concat(音频 c1, 文本 c2) → head     ← 需要完整产物（含音频塔）
        txt_own = ""
        # ★★ 复用 A 路：A 路已把整段音频切成小块**全部解码过** → B 路不必再整段解码
        #    （同一个模型，重复解码纯属浪费：实测整段 ASR 要 150~330ms）
        #    条件：A 路覆盖 ≥95% 的段长，且有文本。A 路没跑/没覆盖 → 照常走整段 ASR。
        try:
            _seg_n = int(np.asarray(wav, dtype=np.float32).size)
            _cov = getattr(self, "_a_samples", 0)
            if (getattr(self, "_a_on", False)
                    and bool(self.cfg.behavior.get("stream_reuse_a", True))
                    and (getattr(self, "_a_text", "") or "").strip()):
                txt_own = self._a_text.strip()
                self._asr_ms = 0.0        # 本句没有额外的 ASR 开销（已由 A 路覆盖）
                # [OPENCODE-Q] 该日志每段都刷屏 → 默认静音；要看就 behavior.log_reuse_a=true
                if self.cfg.behavior.get("log_reuse_a", False):
                    log(f"   ♻️ 复用段文本（A {_cov/16000:.1f}s / 段 {_seg_n/16000:.1f}s）", "ASR")
        except Exception:
            txt_own = ""
        # ★★ B 路 ASR 已【屏蔽】（只保留 A 路转写）。仅当 A 路没产出文本时才兜底一次，
        #    且会明确打印「B 路兜底」，不再静默重复解码。
        if not txt_own and self.asr_backend in ("own", "gguf", "http", "vllm"):
            log("   ⚠️ A 路没产出文本 → B 路兜底整段 ASR（仅此一次）", "ASR")
            try:
                if self.asr is None:
                    self.asr = self._make_asr()
                    self.asr.load()
                # ⭐ 送给 ASR 的音频：默认【AGC 之前的原始音频】—— 对标 app
                #   （app 就是把浏览器采到的原始 PCM 直接发给服务；我的 AGC
                #    把信号放大 10~30 倍，是"转写不如 app"的头号嫌疑）
                #   AGC 只留给 VAD 用。
                # ⚠️ 实测血债：文件模式下 raw 可能为 None / 与 mic 模式不同 →
                #    送给 HTTP ASR 的音频坏了 -> 服务端返回空 -> 日志只显示「(无转写)」，
                #    看不出原因（HTTPASR 单测同一文件却正常 293ms）。
                _asr_in = raw if (raw is not None and
                                  self.cfg.behavior.get("asr_use_raw", True)) else wav
                if _asr_in is None:
                    _asr_in = wav
                try:
                    _arr = np.asarray(_asr_in, dtype=np.float32).ravel()
                except Exception:
                    _arr = np.asarray(wav, dtype=np.float32).ravel()
                if _arr.size == 0:
                    _arr = np.asarray(wav, dtype=np.float32).ravel()
                    pass
                log(f"   ASR 输入 {_arr.size} 点 ({type(self.asr).__name__})", "ASR", sub=True)
                # ★★ 段级峰值归一化（= DSH/Chromium 的 autoGainControl 等价物）
                #    血债：DSH 的转写用 getUserMedia({echoCancellation,noiseSuppression})
                #    → Chromium **默认开 AGC** → 静音麦克风也能用；我们用 PortAudio 裸采
                #    没有 AGC → 麦克风 -45~-57 dBFS → VAD 判不出、ASR 出「嗯。」「我也是。」
                #    这里把每段峰值拉到 ~0.6（增益上限 60×），与 DSH 的输入条件对齐。
                try:
                    _pk0 = float(np.abs(_arr).max())
                    if 0.0 < _pk0 < 0.30:
                        # [OPENCODE-T] ASR 前增益上限可配：小声时 60× 不够（实测 0.05 音量 → 12% 空转）
                        _g = min(0.60 / _pk0, float(self.cfg.behavior.get("asr_gain_cap", 200)))
                        _arr = np.clip(_arr * _g, -1.0, 1.0).astype(np.float32)
                        log(f"   AGC 段级归一化 ×{_g:.1f}  峰值 {_pk0:.5f} → "
                            f"{float(np.abs(_arr).max()):.3f}", "ASR", sub=True)
                except Exception:
                    pass
                txt_own = self.asr.transcribe(_arr)
                self._t_asr1 = time.time()      # ⏱ ASR 返回那一刻
            except Exception as e:
                log(f"❌ ASR 解码器失败: {type(e).__name__}: {e}")
            if txt_own:
                self._asr_ms = float(self.asr.last_ms)
                self._asr_tok = int(getattr(self.asr, "last_tokens", 0) or 0)

        if self.intent_from == "text":
            if not (txt_own or "").strip():
                log(f"🎤 (无转写)  ({dur:.2f}s)")
                log("   ⛔ ASR 没有输出文本，无法判意图 → 拒识")
                self.stats["rejected"] += 1
                _arch("unknown", 0.0, "", False, "")
                return
            try:
                r = self._predict_intent(txt_own)
            except Exception as e:
                log(f"❌ 文本路推理失败: {type(e).__name__}: {e}")
                traceback.print_exc()
                self.stats["failed"] += 1
                return
            self._t_int1 = time.time()          # ⏱ 意图判定返回那一刻
            log(f"转写「{txt_own}」  {self._asr_ms:.0f}ms "
                f"{self._asr_tok}tok", "ASR", sub=True)
        else:
            try:
                r_ctc = self._predict_intent(wav)
            except Exception as e:
                log(f"❌ 推理失败: {type(e).__name__}: {e}")
                traceback.print_exc()
                self.stats["failed"] += 1
                return
            # ⭐ 换回 ASR 自己的解码器：文本由它出，意图头完全不动。
            r = r_ctc
            if txt_own:
                r = self._predict_intent(wav, text_override=txt_own)
                log(f"CTC「{r_ctc['text']}」 ｜ "
                    f"转写「{txt_own}」  {self._asr_ms:.0f}ms "
                    f"{self._asr_tok}tok", "ASR", sub=True)

        intent, conf, text = r["intent"], r["conf"], r["text"]
        # 规则解析拿参数（比让模型输出参数更稳）
        rule_intent, _rule_arg = parse(text, self.cfg)

        # ⭐ 空转写 / 太短 → 直接拒识（实测背景噪声会切出「」「n」这种）
        if len(text.strip()) < 2:
            log(f"🎤 「{text}」  ({dur:.2f}s)")
            log(f"   ⛔ 转写太短（{len(text.strip())} 字），拒识")
            self.stats["rejected"] += 1
            _arch(intent, conf, "", False, text)
            return

        # ⭐⭐⭐ 听写【优先】（用户明确要求）：
        #   "我希望在输入框里输入的优先级更高，在光标能输入的情况下"
        #   场景：光标在输入框里说「在GitHub里搜X」—— 期望是【把这句话打进输入框】，
        #   而不是弹到浏览器去搜索。所以听写放在规则直通/命令执行【之前】。
        #   逃生口：句首先说「命令」（可配 dictation_bypass_prefix），则跳过听写、
        #   按命令执行，例如「命令 打开知乎」。
        # ⭐⭐ 停顿切段拼接（stitch）：实测用户一句话中间换气 ~0.5-1s 就被切成多段，
        #   后半截变成"新句子"。这里：若上一段在 stitch_gap_s 内且没说完 →
        #   ① 上一段是【听写已打进输入框】的 → 先用退格撤回，再打合并后的整句；
        #   ② 上一段没执行任何动作的 → 直接把两段文本拼起来再判。
        _gap = float(self.cfg.behavior.get("stitch_gap_s", 1.5) or 1.5)
        # [OPENCODE-J] 听写段的拼接窗单独可调（behavior.dictation_stitch_gap_s）
        try:
            _st0 = getattr(self, "_stitch", None)
            if _st0 and _st0.get("dictated"):
                _gap = float(self.cfg.behavior.get("dictation_stitch_gap_s", _gap) or _gap)
        except Exception:
            pass
        _st = getattr(self, "_stitch", None)
        if _st and (time.time() - _st.get("t", 0)) <= _gap:
            _prev = _st.get("text", "")
            # ⚠️ 标点判据不可靠：模型给【每一段】都自动补句末标点
            #   （实测「它这个过程应该是。」明明没说完也有句号）→
            #   听写类只按【时间间隔】判定是否拼接；命令类仍要求"没说完"，
            #   而且【已执行动作的命令永远不会进到这里】（不记录 -> 不拼接）。
            # [OPENCODE-K] 听写段是否允许按"时间间隔"拼接（behavior.dictation_stitch，默认 true=原行为）
            _unfinished = bool(_prev) and (
                (bool(_st.get("dictated")) and bool(self.cfg.behavior.get("dictation_stitch", True))) or
                not _prev.rstrip().endswith(("。", "！", "？", ".", "!", "?", "…")))
            if _unfinished:
                if _st.get("dictated"):
                    try:
                        from typein import send_backspaces
                        _n = int(_st.get("n_chars", 0)) + 1
                        send_backspaces(_n)
                        log(f"   ↩️ 撤回上一段 {_n} 个字符（准备打整句）", "ACT", sub=True)
                    except Exception as _e:
                        log(f"   ⚠️ 撤回失败: {type(_e).__name__}: {_e}", "WARN")
                text = _prev + text
                dur = float(dur) + float(_st.get("dur", 0))
                log(f"   ⤵️ 与上一段拼接（停顿被误切）→ 「{text[:50]}」", "ASR")
        # [OPENCODE-N] 自适应信号: 拼接过一次=切碎(窗口放宽); 否则记一次干净段(攒够再收窄)
        try:
            if getattr(self, "_oc_vad", None):
                (self._oc_vad.on_stitch() if _unfinished else self._oc_vad.on_clean())
        except Exception:
            pass
        self._stitch = None

        # ⚠️ 听写优先级【已回滚】（用户：什么都优先判定进输入框，太烂了）。
        #   回到原来的设计：命令优先；听写只在【判成非命令】的出口兜底
        #   （见下面「不像人话」和 unknown 两处 _dictate 调用）。
        #   句首「命令」前缀的逃生口也一并撤掉（不再需要）。

        # ⭐⭐ 「像不像人话」校验 —— 挡噪声乱码
        #    实测噪声会输出「核核核核」这种，而且 conf 高达 0.78~0.99，
        #    光靠置信度挡不住，还会触发浏览器 → 麦克风听到 → 正反馈。
        _b = self.cfg.behavior
        ok_plausible, why = plausible(text, self.cfg,
                                      _b.get("plausible_max_len", 30),
                                      _b.get("plausible_max_repeat", 0.45),
                                      _b.get("plausible_max_flip", 0.40))
        if not ok_plausible:
            # ⭐ 听写：不是命令，但光标在输入框里 → 把转写打进去（用户要的功能）
            #   "如果当前光标在某个输入框里面，我直接说，把这个转写打在这里面"
            #   判定用 UI Automation —— Win32 的 caret 在浏览器/Electron 上不可靠
            # ⭐⭐ 命令永远优先于听写（血债）：实测「清空输入。」被判"不像人话"
            #   （plausible 不认识触发词类命令）-> 直接进了听写 -> 命令本身被
            #   打进输入框 ✗。这里先 parse 一次：认得出命令就【放行】继续往下走
            #   （规则直通会执行它），认不出才听写。
            try:
                _ri_c, _ = parse(text, self.cfg)
            except Exception:
                _ri_c = "unknown"
            if _ri_c != "unknown":
                log(f"   门控放行：plausible 未判为人话，但规则命中 {_ri_c} → 继续走意图层", "INT")
                # ⭐⭐ 这里【绝对不能 return】！要继续往下走，规则直通/意图层才会
                #   真正调度动作。血债：原来这里无条件 return，命令被"放行"后又被
                #   丢弃 —— 用户实测「三农」只有转写、没有任何动作，日志停在
                #   "放行给规则直通" 就没了（没有任何 ACT 行）。
            else:
                self._dictate_async(text, dur, raw, "不像人话")
                return

        final_intent = intent
        # ⭐⭐ 强制听写模式（语音「开始输入」）：不是命令的话全部打进输入框，
        #   而且必须在【置信度门之前】—— 血债：句向量分类器的 conf 是【余弦相似度】
        #   (0.3~1.0)，而 conf_threshold=0.55 是给旧模型 softmax 概率定的 ✗，
        #   自由说的话常低于 0.55 -> 在到达听写兜底之前就被拒识 -> 用户实测感受：
        #   「开始输入之后就没法输入到输入框」✗
        try:
            import typein as _ti_force
            if _ti_force.get_mode() == "on":
                # ⭐⭐ 命令永远优先（血债：强制听写原样吞掉了命令，用户实测
                #   「清空输入」被当普通话打进输入框 ✗）。先 parse：认得命令就放行
                #   交给规则直通/意图处理；只有【不是命令】才强制听写。
                try:
                    _ri_f, _ = parse(text, self.cfg)
                except Exception:
                    _ri_f = "unknown"
                if _ri_f != "unknown":
                    log(f"   ⚙️ 强制听写模式下识别到命令 {_ri_f} -> 正常执行", "INT")
                else:
                    log("   ⌨️ 强制听写模式 -> 直接打进输入框（跳过意图门控）", "INT")
                    self._dictate_async(text, dur, raw, "forced")
                    return
        except Exception:
            pass

        thr = self.cfg.behavior.get("conf_threshold", 0.55)

        # ⭐⭐⭐ 多指令拆分（一句话多个命令）：必须在单条规则直通之前
        #    「电厂、精炼厂」「造两个矿车，再造一个电厂」——原来只执行第一句。
        #    A 路（流式）已经抢跑执行过的动作，这里不再重复执行。
        try:
            _multi = parse_multi(text, self.cfg)
        except Exception as _e:
            _multi = None
            log(f"   ⚠️ 多指令拆分异常（忽略）: {type(_e).__name__}: {_e}", "WARN")
        if _multi and len(_multi) >= 2:
            log(f"🎤 「{text}」  ({dur:.2f}s)")
            log(f"   ⭐ 多指令拆分：{len(_multi)} 条 → "
                f"{'；'.join(f'{i}({a})' for i, a in _multi)}", "INT")
            _done = getattr(self, "_a_executed", set())
            _oks = []
            for _it, _ar in _multi:
                _k = _akey(_it, _ar)
                if _k in _done:
                    log(f"      ⚖ {_it}({_ar}) A 路已执行 → 跳过", "MIC")
                    _oks.append(True)
                    continue
                ok, msg, _info = dispatch(_it, _ar, self.cfg, dry=self.dry)
                _oks.append(ok)
                log(f"      {'✅' if ok else '❌'} {_it}({_ar}) → {msg}")
            if all(_oks):
                self.stats["executed"] += 1
            else:
                self.stats["failed"] += 1
            try:
                self._b_action = (_multi[0][0], _multi[0][1], None)
            except Exception:
                pass
            _arch(_multi[0][0], 1.0, _multi[0][1], all(_oks), text)
            return

        # ⭐⭐⭐ 规则直通（必须在置信度门【之前】）
        #   血债：原来置信度门排在规则优先前面，纯文本头置信度偏低（均值 0.794，
        #   而 concat 头 0.983），于是「打开矩阵乘优化文件」这种【规则能精确命中
        #   闭集】的指令被模型低置信先枪毙 → 端到端 79.15%，127 个错例里 113 个
        #   是「→ unknown」。规则命中是确定性的（要求触发词 + 闭集条目都在），
        #   比模型置信度可靠，所以放它先走。
        #   ⚠️ 仍然后置在 plausible() 之后：那一步要求句子里有已知词，噪声过不来。
        #   ⚠️ 血债两条：
        #     ① dsh_run/dsh_new 原来不在这个名单里 → _rule_arg_r 恒为空 →
        #        DSH 语音命令永远只走"没给任务"的提示分支，动作从没被触发过。
        #     ② 就算放进名单，resolve() 在参数为空时会回退成【整句】，
        #        「新建DSH会话」会被当成任务文本。所以 dsh 直接用 parse 切出的
        #        _rule_arg（空就是空）——这才符合"没说做什么就别动"。
        # ⚠️⚠️ 血债（一定要看清结构，别再插成两份链）：
        #   之前我的字符串替换把这里插成了【两份】_rule_hit 链 —— 第一份里
        #   误写 `_rule_hit = True`，紧接着第二份又 `_rule_hit = False` 覆盖，
        #   而第二份链里没有 clear_input/close_app/site_search 分支 →
        #   这三个功能【全部是死的】（实测：「清空输入。」永不直通，最后被
        #   闭集门控拒识）。
        #   ⇒ 现在只保留一条链，分成"① 取参数 / ② 判可执行"两段。
        # ① 取规则参数
        if rule_intent in ("dictation_on", "dictation_off"):
            _rule_arg_r = ""            # 无参数命令
        elif rule_intent == "clear_input":
            _rule_arg_r = ""                 # 无参数命令
        elif rule_intent == "close_app":
            _rule_arg_r = resolve(rule_intent, text, self.cfg)
        elif rule_intent in ("dsh_run", "dsh_new"):
            _rule_arg_r = _rule_arg
        elif str(rule_intent).startswith("game_"):
            # ⭐ OpenRA 语音控制：参数已经在 openra profile 的规则直通里切好了
            #    （坐标 / 单位 / 姿态 / 数量），不要再走桌面闭集的 resolve()
            _rule_arg_r = _rule_arg
        elif rule_intent in ("launch_app", "open_url", "open_file",
                             "web_search", "file_search", "exact_search",
                             "site_search"):
            _rule_arg_r = resolve(rule_intent, text, self.cfg)
        else:
            _rule_arg_r = ""
        # ② 判这条规则是否可执行
        _rule_hit = False
        if rule_intent in ("launch_app", "open_url", "open_file"):
            _rule_hit = bool(resolve_closed(rule_intent, _rule_arg_r, self.cfg))
        elif rule_intent == "clear_input":
            _rule_hit = True                 # 命中触发词即可（无参数）
        elif rule_intent == "close_app":
            _rule_hit = bool(_rule_arg_r)    # 必须带应用名
        elif rule_intent == "site_search":
            _rule_hit = "\t" in str(_rule_arg_r)   # 参数形如 站点\t查询词
        elif rule_intent == "exact_search":
            _rule_hit = "精确" in text
        elif rule_intent in ("web_search", "file_search"):
            # 参数化意图：必须真的切出了触发词 + 非空参数
            _rule_hit = bool(_rule_arg_r) and len(str(_rule_arg_r)) >= 2
        elif str(rule_intent).startswith("game_"):
            # ⭐ OpenRA 语音控制：profile 的规则直通命中即可执行（参数已切好）
            _rule_hit = True
        elif rule_intent in ("dsh_run", "dsh_new"):
            # DSH 会话：必须带任务内容（「新建会话」光一句不算，得说清做什么）
            _rule_hit = bool(_rule_arg_r) and len(str(_rule_arg_r)) >= 2
            if not _rule_hit:
                # 触发了但没给任务 → 明确告诉用户怎么办，别静默
                # （实测：用户反复说「新建会话」，之前既没动作也没提示）
                log(f"🎤 「{text}」  ({dur:.2f}s)")
                log(f"   ⚠️  识别到{'新建会话' if rule_intent == 'dsh_new' else 'DSH 任务'}"
                    f"（转写「{text}」），但没说【要做什么】→ 不执行")
                log("      说法示例：「新建DSH会话，跑一下测试」"
                    "/「让DSH 看看这个bug」", "INT", sub=True)
                self.stats["rejected"] += 1
                _arch(rule_intent, conf, "", False, text)
                return
        if _rule_hit and self.cfg.behavior.get("rule_direct", True):
            log(f"🎤 「{text}」  ({dur:.2f}s)")
            log(f"   ⭐ 规则直通：{rule_intent} 参数={_rule_arg_r!r}"
                f"（模型判 {intent} conf={conf:.3f}，不参与否决）", "INT")
            self._t_act0 = time.time()          # ⏱ 动作开始
            ok, msg, info = dispatch(rule_intent, _rule_arg_r, self.cfg,
                                     dry=self.dry)
            self._t_act1 = time.time()          # ⏱ 动作返回（规则直通路）
            if ok:
                self.stats["executed"] += 1
                log(f"   ✅ {msg}")
                _arch(rule_intent, conf, _rule_arg_r, True, text)
            else:
                self.stats["failed"] += 1
                log(f"   ❌ {msg}")
                _arch(rule_intent, conf, _rule_arg_r, False, text)
            self._log_timeline("命令·规则直通", getattr(self, "_t_act0", None),
                               self._t_act1)
            return

        # ⭐ 置信度门槛：低于阈值一律拒识
        #    （实测 0.30 太低 —— 背景噪声「n」以 conf=0.577 触发了浏览器搜索）
        if conf < thr:
            log(f"🎤 「{text}」  ({dur:.2f}s)")
            log(f"   ⛔ 置信度 {conf:.3f} < {thr} → 拒识（模型原本判 {intent}）")
            self.stats["rejected"] += 1
            _arch(intent, conf, "", False, text)
            return

        # ⭐ open_web 屏蔽（用户指令：七类命令里屏蔽第七类）
        #    模型可能判 open_web（它是 8 个意图类之一），这里拦掉：
        #    不在闭集就不执行任何动作 —— 避免把任意文本送进浏览器/搜索。
        if final_intent == "open_web" and not self.cfg.behavior.get(
                "enable_open_web", False):
            log(f"🎤 「{text}」  ({dur:.2f}s)")
            log("   ⛔ 不在闭集，且 open_web 已屏蔽 → 不执行"
                "（要恢复：intent_config.json 的 enable_open_web=true）")
            self.stats["rejected"] += 1
            _arch("open_web", conf, "", False, text)
            return

        # 精确查找必须含"精确"关键词（门控）
        kw = None
        for it in self.cfg.raw["intents"]:
            if it["id"] == final_intent:
                kw = it.get("gate_keyword")
        if kw and kw not in text:
            log(f"⚠️  {final_intent} 需要关键词「{kw}」，转写里没有 → 降级 unknown")
            final_intent = "unknown"

        # ⭐ 参数用 resolve：规则拿不到就按【模型给的意图】去对应闭集 snap
        #    （CTC 会吃掉英文首字母：Ollama→llama、PowerShell→owerhell）
        arg = resolve(final_intent, text, self.cfg)

        # ⭐⭐⭐ 规则优先（关键设计）
        #   规则解析器【直接查闭集】，比模型的泛化可靠得多。
        #   实测：新加「微信」到配置里（模型从没见过），
        #     模型把「帮我打开微信」判成 open_url → snap 成了「微博」（开错站！）
        #     但规则直接查 apps 闭集 → launch_app/微信 ✓
        #   ⇒ 规则能命中闭集时就用规则，这样【加新命令不用重训】。
        if rule_intent in ("launch_app", "open_url", "open_file"):
            r_arg = resolve(rule_intent, text, self.cfg)
            if resolve_closed(rule_intent, r_arg, self.cfg):
                if rule_intent != final_intent:
                    log(f"   ⭐ 规则命中闭集 → 用规则（模型判的是 {final_intent}，"
                        f"规则判 {rule_intent}）")
                final_intent, arg = rule_intent, r_arg

        # ⭐⭐ 闭集意图：参数解析不出来 → 降级 unknown
        #    实测踩过的坑：噪声被判成 open_url 但参数是「打开」，
        #    然后报「网址闭集里没有「打开」」并计入 failed —— 应该算拒识。
        if final_intent in ("launch_app", "open_url", "open_file"):
            if not resolve_closed(final_intent, arg, self.cfg):
                # ⭐「保证不漏」接线路（2026-10-17 方案定案 #3）：本意图闭集不认
                #   → 不立刻拒识，先对【三表并集 83 条】做音素级 top-K 兜一次
                #   （模型判错意图表时跨表救回）。top1 分够高且唯一才改判执行；
                #   碰不到就维持拒识 —— 行为只放宽、不猜，反向假命中守住。
                _g_it, _g_arg, _g_sc, _g_sc2 = resolve_by_rank(text, self.cfg)
                if _g_it:
                    log(f"🎤 「{text}」  ({dur:.2f}s)")
                    log(f"   ⭐ 音素级 top-K 改判：{final_intent}({arg!r} 不在闭集)"
                        f" → {_g_it}(「{_g_arg}」)"
                        f"  分数 {_g_sc:.3f} / 次高 {_g_sc2:.3f}")
                    final_intent, arg = _g_it, _g_arg
                else:
                    log(f"🎤 「{text}」  ({dur:.2f}s)")
                    log(f"   ⛔ 闭集意图 {final_intent} 的参数 {arg!r} 在闭集里找不到"
                        f" → 拒识（top-K 兜底最高 {_g_sc:.3f}，未达门槛）")
                    self.stats["rejected"] += 1
                    _arch(final_intent, conf, arg, False, text)
                    return

        _ms = f"  {self._asr_ms:.0f}ms {self._asr_tok}tok" if self._asr_ms else ""
        log(f"「{text}」  {dur:.2f}s{_ms}", "ASR")
        # ★ 闭集对齐后的结果（用户要求：日志要能直接看到"改成了什么"）
        #   原始转写可能带谐音（如「三龙」），对齐层会归一成闭集成员（「三农」）。
        #   ⚠️ 文案【跟 profile 走】—— 原来写死 aoe4 的「补农/训练X个农民」，
        #      在 ops 域会打出完全无关的话（见 align_note 的血债注释）。
        if r.get("src") == "rule" and arg:
            _an = align_note(self.cfg, text, arg, final_intent)
            if _an:
                log(_an, "INT", sub=True)
        # ⚠️ 用户实测问过「到底走的哪一层」—— 原来只打意图名，分不清是
        #    短语表 / 闭集 / items别名 / AI触发。profile.parse() 会把它记在 last_layer 上。
        _layer = ""
        try:
            from actions import _game_profile_obj
            _layer = getattr(_game_profile_obj(self.cfg), "last_layer", "") or ""
        except Exception:
            pass
        log(f"意图={final_intent}  conf={conf:.3f}  参数={arg!r}  "
            f"{r['latency_ms']:.0f}ms"
            + (f"  ← 命中层：{_layer}" if _layer else ""), "INT")
        if final_intent != intent:
            log(f"（模型原本判 {intent}，被规则/门控改成 {final_intent}）", "INT", sub=True)

        # ⭐ 听写优先于"unknown 不执行"：光标在输入框里就把转写打进去
        if final_intent == "unknown":
            # [OPENCODE-E2] B 路兜底：整句查唯一意图清单（全子串：前后缀不影响）→ 命中就直接
            #   执行，不再掉成听写（血债：「调整静音。」被当听写打进终端）
            try:
                from opencode_fastpath import runtime as _oc_rt3
                if _oc_rt3.b_try(self, text, dur, raw):
                    return
            except Exception as _e6:
                log(f"   ⚠️ B 路兜底异常（忽略）: {type(_e6).__name__}: {_e6}", "WARN")
            # ⭐「保证不漏」接线路（方案 #3）：规则判出了【闭集意图】但模型 unknown
            #   时（正是「打开直呼」这类语音错例的经典形态），先用音素级 top-K
            #   对三表并集兜一次 —— top1 分够高且唯一才改判执行；否则维持听写
            #   （改判错比听写错更糟，门槛从严）。
            if rule_intent in ("launch_app", "open_url", "open_file"):
                _u_it, _u_arg, _u_sc, _u_sc2 = resolve_by_rank(text, self.cfg)
                if _u_it:
                    log(f"🎤 「{text}」  ({dur:.2f}s)")
                    log(f"   ⭐ 音素级 top-K 改判：模型 unknown + 规则 {rule_intent}"
                        f" → {_u_it}(「{_u_arg}」)"
                        f"  分数 {_u_sc:.3f} / 次高 {_u_sc2:.3f}")
                    final_intent, arg = _u_it, _u_arg
                else:
                    log(f"   ⛔ top-K 兜底最高 {_u_sc:.3f}（次 {_u_sc2:.3f}）"
                        f"未达门槛 → 维持听写", "ACT", sub=True)
                    self._dictate_async(text, dur, raw, "unknown")
                    return
            else:
                self._dictate_async(text, dur, raw, "unknown")
                return

        # 供流式对账用：记下 B 路（整段权威）最终决定执行什么
        try:
            self._b_action = (final_intent, arg, None)
        except Exception:
            pass

        # ⭐ A 路（流式抢跑）已经真的执行过同一个动作 → B 路不重复执行
        #    （没有这一步，开了 stream_exec 就会「边说边做 + 松手再做一遍」）
        if _akey(final_intent, arg) in getattr(self, "_a_executed", set()):
            log(f"   ⚖ A 路已抢跑执行 {final_intent}({arg}) → B 路不重复", "MIC")
            self.stats["executed"] += 1
            _arch(final_intent, conf, arg, True, text)
            return

        self._t_act0 = time.time()              # ⏱ 动作开始（正常路）
        ok, msg, info = dispatch(final_intent, arg, self.cfg, dry=self.dry)
        self._t_act1 = time.time()              # ⏱ 动作返回
        if final_intent == "unknown":
            self.stats["rejected"] += 1
            log(f"   ⛔ {msg}")
            _arch(final_intent, conf, arg, False, text)
        elif ok:
            self.stats["executed"] += 1
            log(f"   ✅ {msg}")
            _arch(final_intent, conf, arg, True, text)
        else:
            self.stats["failed"] += 1
            log(f"   ❌ {msg}")
            _arch(final_intent, conf, arg, False, text)
        self._log_timeline("命令", self._t_act0, self._t_act1)

    # ── A 路（抢跑）= 把音频切成更细的块，直接提交给【同一个 GPU ASR】 ──
    #    ⭐ 架构澄清（用户）：全产品只有一个模型（GPU 上的 ASR），一切请求都打到它；
    #      "流式抢跑"与"非抢跑"的唯一区别 = 是否把音频切得更细、更早提交。
    #      所以这里没有 WS、没有适配器、没有重复解码循环 —— 只是"切细 + 单飞提交"。
    def _a_queue_loop(self):
        """常驻线程：从队列串行取段 → 整句 ASR → handle()。单飞不并发，且不丢段。"""
        while True:
            try:
                seg = self._a_queue.get()
            except Exception:
                return
            if seg is None:
                return
            try:
                self._a_worker(seg)
            except Exception as e:
                log(f"      A 路队列处理异常（忽略）: {type(e).__name__}: {e}", "SYS", sub=True)
            finally:
                try:
                    self._a_queue.task_done()
                except Exception:
                    pass

    def _a_worker(self, chunk):
        """后台线程：整句一次提交给同一个 ASR → 文本交给 handle（B 路复用，0ms）"""
        # [OPENCODE-E3] 每段开始清空"本句已执行"记录
        #   血债（2026-10-19 实测）：A 路开启时 run_mic 先把 utters 清空，
        #   后面 `if utters: _stream_release()` 永远不执行 → _a_executed 从第一条
        #   命中起就再也不清 → 同一意图+参数的第二句起被【静默去重】
        #   （日志里连"拼音识别 MISS"都没有），用户看到就是
        #   「打开炉石传说没命中」。按段清记录：本段 A 路执行后紧接着的
        #   handle() 仍看得到本句的 key（B 路不会重复执行）。
        self._a_executed = set()
        self._a_actions = []
        try:
            if self.asr is None:
                return
            # ★ 首尾静音裁剪 + 【动态预滚上限】（用户指定：总长 1/5，最高 640ms）
            #   段的结构 = [引擎 640ms 预滚] + [语音] + [尾部端点静音]
            #   → 头部保留量 = min(640ms, 总长/5)（= 剪掉 640 − 总长/5 ✓）
            #   ★★ 但纯算术剪法在【段 < 0.8s】时会切到起音 ✗（预滚最后 ~150ms 是语音
            #      起音 = VAD 检测延迟 hits2×30+平滑3×30），所以【两者取小】：
            #        头部保留 = min( 用户规则(总长/5, ≤640ms), 语音起点 − 100ms )
            #   血债：阈值原本是【绝对 0.02】✗，而 AGC 会把底噪也放大（增益顶格 ×60 时
            #   环境噪声 0.0005 → 0.03 > 0.02）→ 起点永远算成 0 → 安全上限失效，
            #   实测「补农」段 1.09s 里 0.76s 空白被当语音送进 ASR ✗
            _arr = np.asarray(chunk, dtype=np.float32).ravel()
            self._oc_preroll_ms = 0.0          # 本段实际保留的预滚（起点 → 语音起点）
            _pk_seg = float(np.abs(_arr).max()) if _arr.size else 0.0
            # [OPENCODE-I3] 起音别切（用户实测：吞的是【前面】不是后面）
            #   旧：max(0.02, 0.10×段峰值) —— 轻声起音/擦音（sh/s、气声）达不到峰值的
            #   10% → 被当静音切掉 = 前半句被吞。
            #   新：对齐 DSH-Voice-Input-Plugin 的【相对噪声底】判据
            #   thr = max(噪声底×3, 0.004)，再夹一个 ≤0.05×峰值的上限（防底噪漂高）。
            _floor = float(np.percentile(np.abs(_arr), 20)) if _arr.size else 0.0
            _thr = min(max(_floor * 3.0, 0.004), max(0.02, 0.05 * _pk_seg))
            _idx = np.where(np.abs(_arr) >= _thr)[0]
            if _idx.size > 0:
                # [OPENCODE-R] 预滚"砍多少" = 可校准+可自适应参数；基准 = VAD 送来的 640ms
                #   keep = min(640 − ahead_trim_ms, 总长 × ahead_frac)；起点前再留 onset_margin_ms
                _b2 = self.cfg.behavior
                _m = int(SAMPLE_RATE * float(_b2.get("onset_margin_ms", 100)) / 1000.0)
                # ★ 上限基准 = 实际预滚（vad_preroll_ms），不再写死 640：
                #   预滚改成 320 后，若还按 640 算，日志/上限会和实际段结构对不上。
                _pre_base = float(getattr(self.seg, "preroll_ms", 640.0) or 640.0)
                _trim = int(_b2.get("ahead_trim_ms",
                            max(0, int(_pre_base) - int(_b2.get("ahead_cap_ms", 640)))))
                _cap = int(SAMPLE_RATE * max(0, _pre_base - _trim) / 1000.0)
                _lim_head = min(_cap, int(len(_arr) * float(_b2.get("ahead_frac", 0.2))))
                _seg_len = len(_arr)
                _a0 = min(_lim_head, max(0, int(_idx[0]) - _m))             # ★★ 两者取小
                _a1 = min(len(_arr), int(_idx[-1]) + _m)
                if _a1 - _a0 >= 3200:
                    _cut = (len(_arr) - (_a1 - _a0)) / SAMPLE_RATE
                    _arr = _arr[_a0:_a1]
                    _pre_keep_ms = max(0, int(_idx[0]) - _a0) / SAMPLE_RATE * 1000.0
                    self._oc_preroll_ms = _pre_keep_ms
                    if _cut > 0.05:
                        # [OPENCODE-V] 表述修正：原段 → 送 ASR 的长度 + 三个约束
                        log(f"   裁剪静音 {_cut:.2f}s：{len(_arr)/SAMPLE_RATE + _cut:.2f}s"
                            f" → 送 ASR {len(_arr)/SAMPLE_RATE:.2f}s"
                            f"（头部保留 {_a0/SAMPLE_RATE*1000:.0f}ms"
                            f" = min[{_pre_base:.0f}−砍{_trim}ms={max(0, _pre_base-_trim):.0f}ms, "
                            f"总长×{float(_b2.get('ahead_frac', 0.2)):.2f}={_seg_len*float(_b2.get('ahead_frac', 0.2))/SAMPLE_RATE*1000:.0f}ms, "
                            f"起点前{_m/SAMPLE_RATE*1000:.0f}ms={max(0, int(_idx[0]) - _m)/SAMPLE_RATE*1000:.0f}ms]"
                            f"，预滚 {_pre_keep_ms:.0f}ms，语音起点 {_idx[0]/SAMPLE_RATE:.2f}s）",
                            "ASR", level="info")
                    else:
                        log(f"   预滚 {_pre_keep_ms:.0f}ms（段首保留 {_a0/SAMPLE_RATE*1000:.0f}ms，"
                            f"语音起点 {_idx[0]/SAMPLE_RATE:.2f}s，未裁剪）", "ASR", level="info")
            # [OPENCODE-S] 预滚裁剪的【在线信号】: 送 ASR 音频的段首能量比
            #   >0.35 = 起音被切(砍太多) / <0.06 连续多段 = 段首全静音(可多砍)
            try:
                if getattr(self, "_oc_vad", None) and getattr(chunk, "size", 0):
                    _h = chunk[:min(len(chunk), int(SAMPLE_RATE * 0.10))]
                    _ra = float(np.sqrt(np.mean(chunk ** 2)))
                    _rh = float(np.sqrt(np.mean(_h ** 2)))
                    self._oc_vad.on_segment_head(_rh / _ra if _ra > 1e-9 else 0.0)
            except Exception:
                pass
            # ★★ 血债（2026-10-19 用户实测）：裁剪结果赋值那行曾被上面这段
            #   补丁替换掉 → 裁剪只写了日志、ASR 拿到的还是没裁的原音频
            #   （用户原话："裁剪了为什么还要报原长度"）。
            self._oc_seg_s = len(chunk) / SAMPLE_RATE          # 原段（含预滚/尾静音）
            chunk = _arr
            self._oc_sent_s = len(chunk) / SAMPLE_RATE         # 真正送 ASR 的长度
            txt = self.asr.transcribe(np.asarray(chunk, dtype=np.float32))
            self._a_samples = len(chunk)
            self._a_text = (txt or "").strip()
            # [OPENCODE-A] A 路: 先打拼音标注, 再走快路径(影子只记录; live 命中则提前执行并跳过本段 handle)
            try:
                from opencode_fastpath.pinyin import annotate as _oc_anno
                log(f"   [OPENCODE] 拼音标注: {_oc_anno(self._a_text)}", "ASR")
            except Exception:
                pass
            try:
                from opencode_fastpath.runtime import a_path_preempt
                if a_path_preempt(self, self._a_text):
                    return
            except Exception:
                pass
            _ms = float(getattr(self.asr, "last_ms", 0.0) or 0.0)
            # [OPENCODE-G2] 转写行报【真正送 ASR 的长度】，并注明 原段 − 裁掉（
            #   血债：原来恒报原段长度，看着"裁了却没生效"）
            _oc_sent = float(getattr(self, "_oc_sent_s", 0.0) or len(chunk) / 16000.0)
            _oc_seg = float(getattr(self, "_oc_seg_s", 0.0) or _oc_sent)
            _oc_cut = _oc_seg - _oc_sent
            _oc_pre = float(getattr(self, "_oc_preroll_ms", 0.0) or 0.0)
            log(f"   VAD 段转写 {_oc_sent:.2f}s（送 ASR）"
                + (f"；原段 {_oc_seg:.2f}s − 裁 {_oc_cut:.2f}s" if _oc_cut > 0.05 else "")
                + f"；预滚 {_oc_pre:.0f}ms"
                + f" → 「{self._a_text or '(空)'}」  {_ms:.0f}ms", "ASR")
            if not self._a_text:
                log("   段无文本 → 丢弃（不触发整段兜底）", "ASR")
                return
            # self._a_try_preempt(self._a_text)   # ★ 已停用：整段方案下没有部分文本，抢跑是死的 ✗
            import time as _t2
            _t0 = _t2.time()
            self.handle(np.asarray(chunk, dtype=np.float32))   # 意图+动作（复用文本）
            _t1 = _t2.time()
            try:
                _close = getattr(self, "_a_t_close", None)
                _vend = getattr(self, "_a_t_voice_end", None)
                if _close and _vend:
                    _det = max(0.0, (_close - _vend)) * 1000      # 检测延迟（说完→判定关闭）
                    _tot = (_t1 - _vend) * 1000                   # 说完→动作完成
                    log(f"   ⇒ 说完→游戏生效 {_tot:.0f}ms（检测 {_det:.0f} + ASR {self._asr_ms:.0f}"
                        f" + 意图与动作 {(_t1-_t0)*1000:.0f}；游戏侧轮询 ≤100ms）", "MIC")
            except Exception:
                pass
        except Exception as e:
            log(f"      A 路整句处理失败（忽略）: {type(e).__name__}: {e}", "SYS", sub=True)
        finally:
            self._a_busy = False

    def _a_try_preempt(self, partial):
        """闭集部分文本一命中规则就执行（可撤销动作 + _akey 去重，B 路不重复）。"""
        # [OPENCODE-B] 部分文本前缀探测: live 命中即提前执行(影子/关闭不改变流程)
        try:
            from opencode_fastpath.runtime import partial_preempt
            if partial_preempt(self, partial):
                return
        except Exception:
            pass
        try:
            ri, ra = parse(partial, self.cfg)
        except Exception:
            return
        if ri == "unknown" or not getattr(self, "_stream_exec", False):
            return
        k = _akey(ri, ra)
        if k in self._a_executed:
            return
        try:
            ok, msg, _i = dispatch(ri, ra, self.cfg, dry=self.dry)
            # ★★ 血债：原来【无条件】标记已执行 + 打成功日志 ✗ → dispatch 失败时
            #    命令被吞掉（handle() 看到"已执行"就跳过）→ 实测 18:43 四次
            #    「补农」全部没写命令文件 ✗。只有 ok 才算数 ✓
            if ok:
                self._a_executed.add(k)
                self._a_actions.append({"text": partial.strip(), "intent": ri, "arg": ra})
                log(f"   ⚡提前执行 {ri}({ra}) ← 「{partial.strip()}」", "MIC")
            else:
                log(f"   提前执行未成功（{msg}）→ 交回主路径执行", "MIC", sub=True)
        except Exception as e:
            log(f"      提前执行异常（交回主路径）: {type(e).__name__}: {e}", "SYS", sub=True)

    def _a_flush(self, seg=None):
        """VAD 给出完整段（含它自带的 640ms 预滚）→ 裁掉首尾静音 → 入队串行处理"""
        if not getattr(self, "_a_on", False):
            return
        try:
            if seg is None:
                return
            seg = np.asarray(seg, dtype=np.float32).ravel()
            _min = int(SAMPLE_RATE * float(self.cfg.behavior.get("stream_min_s", 0.5) or 0.5))
            if len(seg) < _min:
                # [OPENCODE-I2] 别丢！对齐 DSH-Voice-Input-Plugin：它 minSegmentMs=1000
                #   （不够就继续攒），我们原来 <stream_min_s 直接扔 → 短口令被吞。
                #   现在：只有"极短"（<0.15s，基本是噪声）才丢（打 WARN）；其余照常送 ASR。
                if len(seg) < int(SAMPLE_RATE * 0.15):
                    log(f"   ⚠️ VAD 段极短（{len(seg)/16000:.2f}s）→ 丢弃（阈值 0.15s）",
                        "WARN")
                    return
                log(f"   ⚠️ VAD 段偏短（{len(seg)/16000:.2f}s）但仍送 ASR（不吞输入）", "WARN")
            self._a_queue.put(seg)          # ★ 入队：串行处理，绝不丢段
            _q = self._a_queue.qsize()
            log(f"   VAD 段 {len(seg)/16000:.2f}s → 入队（队列 {_q}）"
                + ("  ⚠️ 上一句还在处理，稍后自动跟上" if _q > 1 else ""), "MIC")
        except Exception as e:
            log(f"   A 路收尾异常（忽略）: {type(e).__name__}: {e}", "SYS", sub=True)

    def _stream_release(self):
        """段结束：等 A 路的尾巴块落地，然后与 B 路权威对账"""
        if not getattr(self, "_a_on", False):
            return
        try:
            # 等尾巴块提交完（最多 1.5s，别卡住主循环）
            _t0 = time.time()
            while getattr(self, "_a_busy", False) and (time.time() - _t0) < 1.5:
                time.sleep(0.02)
            a = {_akey(x["intent"], x["arg"]) for x in self._a_actions}
            b = getattr(self, "_b_action", None)
            if b is None:
                if a:
                    log(f"   ⚖ A/B 对账：A 抢跑 {len(a)} 条；B 路未产出动作", "MIC")
            else:
                kb = _akey(b[0], b[1], b[2] if len(b) > 2 else None)
                if not a:
                    log(f"   ⚖ A/B 对账：A 没抢跑，B={b[0]}({b[1]})", "MIC")
                elif kb in a and len(a) == 1:
                    log(f"   ⚖ A/B 对账：一致 ✅（{b[0]}({b[1]})）", "MIC")
                else:
                    log(f"   ⚖ A/B 对账：A={sorted(a)} vs B={b[0]}({b[1]}) → 以 B 为准",
                        "MIC")
        except Exception as e:
            log(f"   A 路收尾异常（忽略）: {type(e).__name__}: {e}", "SYS", sub=True)
        finally:
            # ⚠️ 只清"已执行记录/对账"，【不能清 _a_text】—— handle() 紧接着要用它
            #    复用（血债：清了就复用不到，B 路会再整段解码一次 335ms ✗）。
            #    _a_text/_a_samples 在【下一段开口时】才清零（见 _stream_tick）。
            self._a_actions, self._b_action = [], None
            self._a_executed = set()
            self._a_buf = []

    def _log_timeline(self, tag, t_act0=None, t_act1=None):
        """⏱ 从【松手】那一刻把每个环节落账 —— 听写路与命令路共用。


        t_seg  = 松手（段和松手在同一块里 push 出来，误差 ≤ 一个 30ms 块）
        t_asr1 = ASR 返回          t_int1 = 意图判定返回
        LAST['t0'] = 真正调用打字函数（其后是 焦点判定→读框→逐字注入→验证）
        t_act0/t_act1 = 动作开始 / 动作结束

        为什么值得拆这么细：只看"ASR 120ms"会把真实等待低估几倍 ——
        UIA 那两次读（焦点判定/读整个输入框）实测各上百 ms，比 ASR 还贵。
        """
        try:
            _t0 = getattr(self, "_t_seg", None)
            if not _t0:
                return
            import typein as _ti
            _tm = dict(getattr(_ti, "LAST", {}) or {})
            f = lambda t: (t - _t0) * 1000 if t else None          # noqa: E731
            asr = f(getattr(self, "_t_asr1", None))
            it1 = f(getattr(self, "_t_int1", None))
            a0, a1 = f(t_act0), f(t_act1)
            ty0 = f(_tm.get("t0"))
            now = (time.time() - _t0) * 1000
            # ★ 常开监听（vad）没有"松手"，措辞跟着触发方式走（埋点别误导）
            t0name = "松手" if getattr(self, "trigger_mode", "vad") == "ptt" else "段结束"
            seg = [f"{t0name}→ASR返回 {asr:.0f}ms" if asr else f"{t0name}→?"]
            if asr and it1:
                seg.append(f"→意图返回 {it1:.0f}ms(+{it1-asr:.0f})")
            if a0:
                seg.append(f"→动作开始 {a0:.0f}ms(+{a0-(it1 or a0):.0f})")
                if a1:
                    seg.append(f"→动作结束 {a1:.0f}ms(+{a1-a0:.0f})")
            if ty0:
                seg.append(f"→进入打字函数 {ty0:.0f}ms(+{ty0-(it1 or a0 or ty0):.0f})")
                fm, rd, sd = _tm.get("field_ms"), _tm.get("read_ms"), _tm.get("send_ms")
                if fm is not None:
                    seg.append(f"→焦点判定完 {ty0+fm:.0f}ms(+{fm})")
                if rd is not None and fm is not None:
                    seg.append(f"→读框完 {ty0+fm+rd:.0f}ms(+{rd})")
                if sd is not None:
                    seg.append(f"→字符全进框 {ty0+sd:.0f}ms(+{sd-(fm or 0)-(rd or 0)})")
            log(f"   ⏱ [{tag}] " + "  ".join(seg), "MIC")
            if a1:
                log(f"      ⇒ {t0name}→执行完成 {a1:.0f}ms（ASR {asr or 0:.0f}"
                    f" + 意图 {(it1 or 0)-(asr or 0):.0f}"
                    f" + 调度 {(a0 or 0)-(it1 or 0):.0f}"
                    f" + 动作 {a1-(a0 or 0):.0f}）", "MIC", sub=True)
            elif ty0 and _tm.get("send_ms") is not None:
                # [OPENCODE-Y] 表述修正：验证已由固定 sleep 改为轮询确认（typein 改动）
                _ver = float(_tm.get("verify_ms", 0) or 0)
                log(f"      ⇒ 文字进框 {ty0+_tm['send_ms']:.0f}ms"
                    f"  ｜ 日志写出 {now:.0f}ms（其中验证确认 {_ver:.0f}ms，字已进框）",
                    "MIC", sub=True)
        except Exception as _e:
            log(f"   ⚠️ 时间线失败: {type(_e).__name__}: {_e}", "WARN")

    # ── 错例存档（raw 音频 + 转写/判定结果）──
    def _save_utt(self, raw, text, intent, conf, arg, accepted, dur):
        """把这一句的【原始音频】和判定结果落盘，供离线反复复现。

        只在 save_utterances=true 时开。文件名带时间和意图，索引写 jsonl：
            utt/143012_launch_app_启动DeepSeek.wav
            utt/index.jsonl  {"ts":..., "wav":..., "text":..., "intent":...}

        ⚠️ 分来源（src）：mic = 真实麦克风，file = --wav 跑语料。
           实测踩中：跑一次 609 条 val 复测就往 utt/ 灌 609 个 TTS 文件，
           把真实麦克风的错例全淹了（一次实验 1833 条 / 153 MB）。
           所以 file 模式默认不存档（save_utterances_file=false）。
        """
        try:
            import json as _json
            import re as _re
            import wave as _wave
            src = getattr(self, "_src", "mic")
            if src == "file" and not self.cfg.behavior.get(
                    "save_utterances_file", False):
                return
            d = Path(self.cfg.behavior.get("utt_dir", "utt"))
            d.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%H%M%S")
            slug = _re.sub(r"[^\w\u4e00-\u9fff]+", "", (text or ""))[:16] or "empty"
            name = f"{ts}_{src}_{intent}_{slug}.wav"
            p = d / name
            x = np.clip(np.asarray(raw, dtype=np.float32), -1.0, 1.0)
            with _wave.open(str(p), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes((x * 32767.0).astype("<i2").tobytes())
            rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "src": src,
                   "wav": name,
                   "dur": round(float(dur), 2), "rms": round(
                       float(np.sqrt((x ** 2).mean())), 6),
                   "text": text, "intent": intent, "conf": round(float(conf), 3),
                   "arg": arg, "accepted": bool(accepted)}
            with open(d / "index.jsonl", "a", encoding="utf-8") as f:
                f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            log(f"⚠️ 错例存档失败: {type(e).__name__}: {e}", "WARN")

    # ── 常开循环 ──
    def run_mic(self, device=None):
        self._src = "mic"          # 错例来源标记（file 模式默认不入档）
        try:
            import sounddevice as sd
        except Exception as e:
            log(f"❌ 缺 sounddevice: {e}")
            log("   装：pip install sounddevice")
            return 2
        if self.lazy:
            _m = self.idle_unload_s / 60.0
            log(f"🕒 动态加载：启动不装模型（空闲 "
                f"{_m:.0f}min 自动卸载；开口即后台预加载）")
            # ⚠️ 提醒：lazy 模式下【第一句话必丢开头】—— IntentModel 构造实测 17s
            #    （全程 19.6s），这期间音频队列会满（实测丢 244 块 = 7.3s）。
            #    常开控制台建议 lazy_load=false（启动就装好）。ops 域已经关掉了。
            log("   ⚠️ lazy 模式：第一句话会撞上模型加载，开头可能丢；"
                "常开建议 lazy_load=false", "WARN", sub=True)
        else:
            self.load_model()
            if self.asr_backend in ("own", "gguf", "http", "vllm"):
                try:
                    self.asr = self._make_asr()
                    self.asr.load()
                except Exception as e:
                    log(f"❌ ASR 加载失败（{self.asr_backend}）: {type(e).__name__}: {e}")
                    self.asr = None
            if self.quantize_int8 and self.asr is not None and self.asr_backend == "own":
                self._apply_int8()
            if self.cfg.behavior.get("trim_after_load", True):
                self._trim_working_set()

        if device is None:
            try:
                devs = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
                log("可用输入设备：")
                for i, d in enumerate(devs):
                    log(f"   [{i}] {d['name']}  ({d['max_input_channels']}ch)")
            except Exception:
                pass
            device = sd.default.device[0]

        log(f"🎧 开始常开监听（设备 {device}，{SAMPLE_RATE}Hz）... 说话就行")

        # ⭐⭐ 启动时标定环境噪声 —— 只给噪声跟踪器一个初值
        #    实测教训（两个方向都踩过）：
        #      · 门槛写死 0.0025：比本机噪声底还低 → 噪声全判成语音 → 段段撑满 6s
        #      · 门槛写死 max(0.008, floor×3)：0.008 比本机语音帧 0.0015~0.004 还高
        #        5~10 倍 → 能量判据永不触发 → 只剩 webrtcvad 在数字静音上乱切
        #    ⟹ 门槛既不能写死、也不能只按噪声底缩放，必须由 vad.py 在
        #      [rms_floor, rms_ceiling] 区间内随噪声漂移（结构照抄 DSH 插件）。
        # ⭐⭐ 启动标定 —— 只有能量后端需要（它要估噪声底来定门槛）
        #    Silero 后端不需要：判据是神经网络概率，电平由前置 AGC 归一化，
        #    所以跳过这 1.5 秒的等待，也不再打那个没意义的"门槛区间"。
        _need_cal = self.vad_backend not in ("silero", "ptt")
        if self.vad_backend == "ptt":
            log(f"触发方式 = 按住说话（{getattr(self.seg, 'key_name', '?')}）："
                f"不按就不听；段起点 = 按下 − {self.seg.preroll_ms:.0f}ms 预滚动，"
                f"松手即送 ASR（不走 704ms 静音等待）", "MIC")
        elif not _need_cal:
            log("分句后端 = Silero：跳过噪声标定（AGC 前置、判据是概率）", "MIC")
        try:
            if _need_cal:
                log("🎚 标定环境噪声（1.5 秒，请先别说话）...")
                cal = sd.rec(int(SAMPLE_RATE * 1.5), samplerate=SAMPLE_RATE, channels=1,
                             dtype="float32", device=device)
                sd.wait()
                # ⚠️ 用【帧 RMS 的中位数】而不是整段均值：均值会被这 1.5 秒里的说话/
                #    敲键盘/系统音拉高。实测踩中：均值算出 0.00697 → 门槛区间被抬到
                #    [0.0105, 0.0558]，而平时说话的帧只有 0.0023~0.0089 → 直接漏掉。
                _fr = cal[:len(cal) // 480 * 480, 0].reshape(-1, 480)
                _frs = np.sqrt((_fr ** 2).mean(axis=1))
                floor = float(np.median(_frs))
                self.seg.set_noise_floor(floor)
                self._noise_floor = floor
                log(f"🎚 标定帧RMS 中位 {floor:.5f} / p90 "
                    f"{float(np.percentile(_frs, 90)):.5f} / 均值 {float(_frs.mean()):.5f}"
                    f" → 门槛区间 [{self.seg.rms_floor:.5f}, {self.seg.rms_ceiling:.5f}]"
                    f"（跟踪 × {self.seg.voice_margin:g}）")
        except Exception as e:
            log(f"🎚 标定失败（{type(e).__name__}），用配置里的门槛 "
                f"{self.seg.energy_threshold:.4f}")

        # ⭐⭐ 流式抢跑（A 路）—— 默认关；开了才在【按住期间】边转写边判边（可选）执行
        #    A 路只做两件确定性的事：规则层直通 + 只留可撤销动作（见 stream_consume.py）
        #    B 路（松手后整段 ASR → handle()）永远是权威，A 与它不一致时以 B 为准
        #    设计依据见 STREAMING_INTENT_PLAN.md §〇之二 / STREAMING_ENGINE_NOTES.md
        self._stream = None
        self._consumer = None
        self._stream = None            # 新架构不再使用（无 WS）
        self._stream_texts = None
        self._stream_texts = None
        self._a_actions = []          # 本句 A 路（流式抢跑）已决定的动作
        self._a_executed = set()      # 本句 A 路【已经真的执行】的动作键（B 路去重用）
        self._b_action = None         # 本句 B 路（整段权威）的动作，用于对账
        _sb = self.cfg.behavior            # ⚠️ 这里没有 __init__ 的局部名 b（血债：
                                           #   早先写成 b.get(...) → NameError 把
                                           #   run_mic 整个打挂，app 启动即死）
        # ── A 路（抢跑）初始化 ──────────────────────────────────────────
        #   ⭐ 架构（用户澄清）：全产品只有一个模型（GPU 上的 ASR），一切请求打到它；
        #     "抢跑"只是把音频切得更细、更早提交。没有 WS / 没有适配器 / 没有重复解码。
        self._a_on = False
        if _sb.get("stream_enable", False):
            self._a_on = True
            self._stream_exec = bool(_sb.get("stream_exec", False))
            _chunk_s = float(_sb.get("stream_chunk_s", 1.0) or 1.0)
            self._a_chunk = max(3200, int(SAMPLE_RATE * _chunk_s))   # ≥0.2s
            self._a_buf = []
            self._a_text = ""
            self._a_samples = 0
            self._a_busy = False
            self._a_P = float(_sb.get("stream_P_s", 0.2) or 0.2)
            self._a_k = int(_sb.get("stream_k", 1) or 1)
            self._a_thr = float(_sb.get("stream_thr", 0.05) or 0.05)
            self._a_ring = []
            self._a_win = []
            self._a_seg = []
            self._a_in = False
            self._a_below = 0
            # ★ 预热"游戏是否在跑"缓存：否则【第一条】命令的动作段要白付 ~100ms
            #   （tasklist 子进程 ✗）。预热后每条命令的动作都是 ~5ms ✓
            try:
                from aoe4 import bridge as _br
                import threading as _thw
                _thw.Thread(target=_br.is_game_running, daemon=True,
                            name="vb-warm-gamecheck").start()
            except Exception:
                pass
            import queue as _qq, threading as _th2
            self._a_queue = _qq.Queue()          # ★ 段队列（不丢段）
            _th2.Thread(target=self._a_queue_loop, daemon=True,
                        name="vb-a-queue").start()
            # [OPENCODE-C] 逐字早火: 接 realtime 流(部分文本 → 快路径多子串 → 提前执行)
            self._oc_wire = None
            if _sb.get("stream_url"):
                try:
                    from opencode_fastpath.stream_wire import StreamWire
                    self._oc_wire = StreamWire(self, _sb.get("stream_url"),
                                               _sb.get("stream_model", "qwen3-asr"),
                                               int(_sb.get("stream_start_ms", 300) or 300),
                                               log=lambda m: log(m, "ASR"))
                    self._oc_wire.start()
                except Exception as _e:
                    log(f"   [OPENCODE] 流式早火未启用: {type(_e).__name__}: {_e}", "WARN")
            _ex = "开" if self._stream_exec else "关"
            # [OPENCODE-F] PTT 模式下 self.seg 是 PttGate(无 .sm) → 原日志行必然 AttributeError 崩溃
            _sm = getattr(self.seg, "sm", None)
            _segdesc = (f"Silero VAD prob {_sm.prob_threshold} / hits {_sm.required_hits} / "
                        f"misses {_sm.required_misses}(≈{int(_sm.required_misses)*32}ms静音窗) / 平滑 {_sm.smoothing_window}") \
                if _sm else "PTT 触发（按住说话）"
            _ftrim = max(0, 640 - int(_sb.get("ahead_cap_ms", 640)))
            log(f"⚡分段已启用：{_segdesc}；头部保留 = min[VAD 640ms−砍{_ftrim}ms={640-_ftrim}ms, "
                f"总长×{_sb.get('ahead_frac', 0.2)}, 起点前{_sb.get('onset_margin_ms', 100)}ms]"
                f"（语音校准/在线自适应）；A路提前执行={_ex}", "MIC")
            # [OPENCODE-M] 静音窗: 载入本说话人档案 + 启动在线自适应(拼接=切碎信号)
            try:
                from opencode_fastpath.vad_adapt import apply_at_startup
                self._oc_vad = apply_at_startup(self)
            except Exception as _e3:
                self._oc_vad = None
                log(f"   [OPENCODE] VAD 自适应未启用: {type(_e3).__name__}: {_e3}", "WARN")
            # [OPENCODE-O] 注册「调整静音判定」意图动作 + 记录当前 agent（动作回调需要它来改 VAD）
            try:
                from opencode_fastpath import vad_action as _oc_va
                import actions as _oc_actions
                _oc_actions.ACTIONS["vad_calibrate"] = _oc_va.act_vad_calibrate
                _oc_va.set_agent(self)
            except Exception as _e4:
                log(f"   [OPENCODE] 静音判定动作未注册: {type(_e4).__name__}: {_e4}", "WARN")
            # [OPENCODE-X] 日志级别：behavior.log_level(默认 info) / env VOICE_AGENT_LOG 覆盖
            try:
                import os as _ocos2
                from dlog import set_level as _oc_set_level, get_level as _oc_get_level
                _oc_set_level(_ocos2.environ.get("VOICE_AGENT_LOG")
                             or self.cfg.behavior.get("log_level", "info"))
                log(f"日志级别={_oc_get_level()}"
                    f"（debug 用 VOICE_AGENT_LOG=debug 或 behavior.log_level=debug）", "SYS")
            except Exception:
                pass
            # [OPENCODE-G] 唯一意图开关（intent_config.json 的 intent_registry）+ 统一返回 + 命令列表
            try:
                import actions as _oc_actions2
                from opencode_fastpath import cmd_list as _oc_cl
                from opencode_fastpath import intents as _oc_intents
                from opencode_fastpath import reply as _oc_rp
                from opencode_fastpath import sens as _oc_sens
                from opencode_fastpath import micwatch as _oc_mw2
                from opencode_fastpath import domains as _oc_dom
                from opencode_fastpath import favs as _oc_fav
                from opencode_fastpath import hotwords as _oc_hw
                _oc_actions2.ACTIONS["show_commands"] = _oc_cl.act_show_commands
                _oc_actions2.ACTIONS["sens_up"] = _oc_sens.act_sens_up
                _oc_actions2.ACTIONS["sens_down"] = _oc_sens.act_sens_down
                _oc_actions2.ACTIONS["sens_show"] = _oc_sens.act_sens_show
                _oc_actions2.ACTIONS["asr_diag"] = _oc_sens.act_asr_diag
                _oc_actions2.ACTIONS["reload_intents"] = _oc_sens.act_reload_intents
                _oc_sens.set_agent(self)
                _oc_actions2.ACTIONS["domain_daily"] = _oc_dom.act_domain_daily
                _oc_actions2.ACTIONS["domain_game"] = _oc_dom.act_domain_game
                _oc_actions2.ACTIONS["domain_ops"] = _oc_dom.act_domain_ops
                _oc_actions2.ACTIONS["fav_open"] = _oc_fav.act_fav_open
                _oc_actions2.ACTIONS["fav_show"] = _oc_fav.act_fav_show
                _oc_actions2.ACTIONS["fav_sync"] = _oc_fav.act_fav_sync
                _oc_hw.set_agent(self)          # ASR 热词：从唯一清单自动生成
                _oc_hw.apply(self)
                _oc_mw2.install(self.cfg, self)     # 麦克风监控看门狗
                _oc_intents.install(self.cfg)        # 加载唯一开关（并打一行口径）
                _oc_rp.set_agent(self)
                _oc_rp.install()
            except Exception as _e5:
                log(f"   [OPENCODE] 意图表/统一返回未注册: {type(_e5).__name__}: {_e5}", "WARN")

        running = {"v": True}

        def _stop(signum, frame):
            running["v"] = False
        try:
            signal.signal(signal.SIGINT, _stop)
            signal.signal(signal.SIGTERM, _stop)
        except Exception:
            pass

        t0 = time.time()
        last_meter = 0.0
        last_warn = 0.0
        # ⭐⭐ 采集与处理解耦：独立线程持续抽流 → 队列 → 主线程消费
        #   实测（probe_overflow.py）：停读 3 秒，10 秒窗口里只读到 241/334 块
        #   = **静默丢掉 ~2.8 秒音频，而 PortAudio 的 overflowed 标记报 0 次**。
        #   所以主循环里任何阻塞（加载模型 2~4.4s、ASR 0.3~1.6s、GC）都会吃音。
        #   读取必须独立成线程、永不间断；处理慢只是让队列堆积。
        import threading
        import queue as _queue
        # ⚠️ 200×30ms = 6 秒缓冲实测【不够】：首次开口触发的后台预加载花了 21.6s
        #    （因为 intent_text.pt 缺失、回退到了 292M 参数的完整产物），
        #    队列满丢 527 块 = 15.8 秒音频。文本路产物到位后加载会快一个量级，
        #    但加载/GC 都可能偶发变慢，所以放到 400 块 = 12 秒留余量。
        #    内存代价：400×480×4B ≈ 768 KB，可以忽略。
        q = _queue.Queue(maxsize=400)          # 400×30ms = 12 秒缓冲
        stop = {"v": False}
        stat = {"blocks": 0, "ovf": 0, "qdrop": 0, "rd_died": 0, "err": 0}
        last_drop = {"q": 0, "o": 0, "e": 0}   # 上次上报时的计数（用来算增量）

        def _reader():
            try:
                with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                    dtype="float32", blocksize=BLOCK,
                                    device=device) as st:
                    while not stop["v"]:
                        try:
                            data, ovf = st.read(BLOCK)
                        except Exception as e:
                            log(f"读取失败: {e}")
                            break
                        stat["blocks"] += 1
                        # [OPENCODE-F2] 麦克风监控：块到达间隔 / 队列深度 / 驱动溢出
                        try:
                            from opencode_fastpath import micwatch as _oc_mw
                            _oc_mw.tick(qsize=q.qsize(), ovf=bool(ovf))
                        except Exception:
                            pass
                        stat["ovf"] += bool(ovf)
                        try:
                            q.put_nowait(data[:, 0].copy())
                        except _queue.Full:
                            stat["qdrop"] += 1
                        # [OPENCODE-P] 静音判定校准：同一块音频 tee 给 vad_action（非阻塞；未校准时是空转）
                        try:
                            from opencode_fastpath import vad_action as _oc_va2
                            _oc_va2.feed(data[:, 0])
                        except Exception:
                            pass
            except Exception as e:
                log(f"❌ 采集线程异常: {type(e).__name__}: {e}")

        _rd = threading.Thread(target=_reader, daemon=True, name="mic-reader")
        _rd.start()
        log("🎧 采集线程已启动（读取/处理解耦：加载与推理期间不再丢音）")
        last_rd_check = 0.0
        try:
            while running["v"]:
                try:
                    blk = q.get(timeout=0.5)
                except _queue.Empty:
                    blk = None

                # ⭐⭐ 采集线程看门狗：采集线程若死掉，进程会"还在跑但彻底聋了"，
                #    比崩掉更隐蔽。这里检测并重启（3 秒退避，最多 5 次）。
                now = time.time()
                if now - last_rd_check > 3.0:
                    last_rd_check = now
                    if not _rd.is_alive() and not stop["v"]:
                        stat["rd_died"] += 1
                        log(f"❌ 采集线程已死，重启（第 {stat['rd_died']} 次）", "ERR")
                        if stat["rd_died"] > 5:
                            log("❌ 采集线程反复死亡，停止（避免聋着假运行）", "ERR")
                            break
                        _rd = threading.Thread(target=_reader, daemon=True,
                                               name="mic-reader")
                        _rd.start()

                if blk is None:
                    continue
                # [OPENCODE-D] 逐字早火: 每个音频块推给 realtime 流
                if getattr(self, "_oc_wire", None):
                    self._oc_wire.push(blk)

                # ⭐⭐ 单段隔离：一段音频出任何异常都不该杀死整个常开进程。
                #    血债记录：handle() 里一句 self.seg.frame_len 让 App 自动退出，
                #    用户只看到"怎么自动关闭了？？"。异常必须被吞在这里并计数。
                try:
                    utters = self.seg.push(blk)
                    # [OPENCODE-U] 起判（何时触发输入）自适应信号：持续有声但没切出段=被漏掉
                    try:
                        if getattr(self, "_oc_vad", None):
                            self._oc_vad.on_block(blk, bool(utters))
                    except Exception:
                        pass
                except Exception as e:
                    stat["err"] += 1
                    log(f"❌ 分句失败: {type(e).__name__}: {e}", "ERR")
                    traceback.print_exc()
                    utters = []
                # ⚡ 流式抢跑（A 路）：按住期间边喂音频边消费增量文本
                if getattr(self, "_a_on", False):
                    for _u in (utters or []):
                        # [OPENCODE-E] 段边界: 清空流式累积文本(下一段重新开始)
                        if getattr(self, "_oc_wire", None):
                            self._oc_wire.boundary()
                        self._a_flush(_u)       # ★ VAD 的完整段（含自带预滚）→ 裁剪 → 入队
                    utters = []                 # 旧 B 路不再直接 handle
                if utters:
                    self._stream_release()
                for utter in utters:
                    try:
                        self.handle(utter)
                    except Exception as e:
                        stat["err"] += 1
                        log(f"❌ 处理该段失败（已跳过，进程继续）: "
                            f"{type(e).__name__}: {e}", "ERR")
                        traceback.print_exc()
                # ⭐ 动态加载：VAD 刚判到"在说话"就后台装模型，
                #    等你把话说完（1~3 秒）模型已经就绪，感知不到等待
                if (self.lazy and self.preload_on_onset
                        and self.seg.state == "speech"):
                    self.preload_async("VAD 开口")
                # ⭐ 每 5 秒打一次电平 —— 一眼看出麦克风有没有收到东西
                now = time.time()
                if now - last_meter > 5.0:
                    lv = self.seg.level
                    bar = "█" * min(30, int(lv * 1500))
                    if self.vad_backend in ("silero", "ptt"):
                        if self.vad_backend == "ptt":
                            # PTT 没有概率可看：直接显示"在不在按住"（这才是唯一的状态）
                            _extra = ("🔴 按住中（说完松手就出字）" if self.seg.state == "speech"
                                      else "待命：按住右 Ctrl 说话")
                        else:
                            # ⚠️ 这个数字是【采样瞬间】的概率（多数时刻落在静音上 -> 显示 0.000）
                            #   真正判断"有没有听到人声"要看 ASR 那行的【VAD峰概率】（本段峰值）。
                            #   标签改清楚，免得再被它误导（我自己就被误导过一轮）。
                            _extra = f"VAD瞬时 {getattr(self.seg, 'last_prob', 0):.3f}"
                        log(f"电平 {lv:.4f}  AGC×{self.seg.gain_now:.1f}  {_extra}  {bar}",
                            "MIC")
                    else:
                        log(f"电平 {lv:.4f}  {bar}", "MIC")
                        # ⚠️ 旧版这里用写死的 0.005 判"太低"，每 5 秒刷一次（实测 307 行）
                        #    且误报。改成：只有低于【当前 VAD 门槛】才提醒，且每分钟最多一次。
                        if lv < self.seg.threshold and now - last_warn > 60:
                            log(f"电平低于门槛 {self.seg.threshold:.5f}（麦克风偏轻，"
                                f"AGC 会补；这是提示不是故障）", "SYS")
                            last_warn = now
                    # ⚠️ 血债（2026-10-16 实测）：stat 是【累计】计数且从不重置，
                    #    所以启动时丢过一次之后，这条日志会每 5 秒重刷同一个数字
                    #    （实测 527 从 08:46:55 一直刷到 08:53:46，看着像持续丢块，
                    #    其实是 21.6s 模型加载期间一次性丢的）。
                    #    改成报【本区间增量】+ 累计，且只有增量非零才打。
                    _dq = stat["qdrop"] - last_drop["q"]
                    _dv = stat["ovf"] - last_drop["o"]
                    _de = stat["err"] - last_drop["e"]
                    if _dq or _dv or _de:
                        log(f"采集丢块（本区间）队列满 {_dq} / 驱动溢出 {_dv} / "
                            f"段内异常 {_de}（各 30ms）  ·  累计 队列满 {stat['qdrop']}"
                            f" / 驱动溢出 {stat['ovf']}", "WARN")
                    last_drop["q"], last_drop["o"], last_drop["e"] = (
                        stat["qdrop"], stat["ovf"], stat["err"])
                    last_meter = now
                    if self.lazy:
                        self.unload_idle(self.idle_unload_s)
        except KeyboardInterrupt:
            pass
        finally:
            stop["v"] = True
            try:
                tail = self.seg.flush()
            except Exception as e:
                log(f"❌ 收尾 flush 失败: {type(e).__name__}: {e}", "ERR")
                tail = None
            if tail is not None and len(tail) > SAMPLE_RATE * 0.3:
                try:
                    self.handle(tail)
                except Exception as e:
                    log(f"❌ 收尾段处理失败: {type(e).__name__}: {e}", "ERR")
        self.report(time.time() - t0)
        return 0

    # ── 文件模式（测试） ──
    def run_files(self, paths):
        self.load_model()
        # 标记来源：file 模式的错例默认不入档（否则跑一次 val 复测就灌 600+ 个
        # TTS 文件，把真实麦克风的错例淹掉 —— 实测踩中，1833 条 / 153 MB）
        self._src = "file"
        t0 = time.time()
        for p in paths:
            p = Path(p)
            if not p.exists():
                log(f"❌ 不存在: {p}")
                continue
            from model_runner import read_wav
            wav = read_wav(str(p))
            log(f"--- 文件 {p.name} ({len(wav)/SAMPLE_RATE:.2f}s) ---")
            self.handle(wav)
        self.report(time.time() - t0)
        return 0

    def report(self, secs):
        s = self.stats
        log("=" * 62)
        log(f"统计：听到 {s['heard']} 句 ｜ 执行 {s['executed']} ｜ "
            f"拒识 {s['rejected']} ｜ 失败 {s['failed']} ｜ 用时 {secs:.0f}s")
        log("=" * 62)


# ══════════════════════════════════════════════════════════════
#  工具接口 —— 给强 AI / LLM 用（不需要麦克风、不需要模型）
#
#  设计目标（用户要求：「作为工具持续的由强AI添加命令」「让 llm 自行使用查询」）：
#    · --json           所有输出都是 JSON，LLM 好解析
#    · --query TEXT     纯文本 → (意图, 参数, 动作)，不用音频
#    · --add-app/-url/-file   往配置里加一条（LLM 直接调用）
#    · --remove ID      删一条
#    · --list --json    列出全部命令（LLM 查有什么可用）
#
#  ⭐ 关键：加新程序/网址/文件【不需要重训模型】——
#     因为 launch_app/open_url/open_file 已经是训练好的类，
#     新条目只是往闭集里加一行，靠规则解析 + snap 就能命中。
#     （实测：加「微信」「天猫」后 4/4 全对，模型一个参数没动）
# ══════════════════════════════════════════════════════════════
def tool_commands_json(cfg: Config):
    return {
        "config": str(cfg.path),
        "apps": [{"id": a["id"], "name": a["name"], "aliases": a["aliases"],
                  "path": a["path"], "exists": Path(a["path"]).exists()}
                 for a in cfg.apps],
        "urls": [{"id": u["id"], "name": u["name"], "aliases": u["aliases"], "url": u["url"]}
                 for u in cfg.urls],
        "files": [{"id": f["id"], "name": f["name"], "aliases": f["aliases"],
                   "path": f["path"], "exists": Path(f["path"]).exists()}
                  for f in cfg.files],
        "intents": [i["id"] for i in cfg.raw["intents"]],
        "triggers": {k: v for k, v in cfg.triggers.items()},
        "behavior": cfg.behavior,
    }


def align_note(cfg, text: str, arg, intent: str) -> str:
    """闭集对齐的日志文案 —— 从【当前 profile】取，取不到就用通用写法。

    ⚠️⚠️ 血债（2026-10-17 实测，用户当场骂的）：
       这里原来写死了 aoe4 的东西 ——
           _canon = "补农"
           log(f"闭集对齐：「{text}」→「{_canon}」→ 训练 {arg} 个农民")
       而且判据只是「规则层有参数」，【不管当前是哪个域】。
       于是在 ops 域说「主机资源面板」会打出：
           闭集对齐：「首席资源面板。」→「补农」→ 训练 首席资源面板。 个农民
       —— 完全是另一个域的话，看着像路由错了，其实是这行日志写死的。

       **对齐文案必须跟 profile 走（换域 = 换数据）**，所以现在读
       profile.json 的 `display.align`：
           "display": { "align": { "canon": "补农",
                                   "note": "训练 {arg} 个农民",
                                   "intents": ["aoe4_train_villager"] } }
       没配的域（ops / openra / 桌面）自动走通用写法「原文 → 参数」。
    """
    prof = None
    try:
        from actions import _game_profile_obj
        prof = _game_profile_obj(cfg)
    except Exception:
        prof = None
    al = {}
    try:
        al = ((prof.raw or {}).get("display") or {}).get("align") or {}
    except Exception:
        al = {}
    if al:
        ids = al.get("intents") or []
        if not ids or intent in ids:
            canon = al.get("canon") or ""
            # 原文里已经含规范写法时用 note_same（复现原来那个分支：
            # 「补农」→ 只打「闭集对齐「补农」→ 训练 3 个农民」，不画多余的箭头）
            tpl = al.get("note") or ""
            if canon and canon in text and al.get("note_same"):
                tpl = al["note_same"]
            if tpl:
                try:
                    return tpl.format(arg=arg, text=text, canon=canon, intent=intent)
                except Exception:
                    return tpl
            if canon and canon != text:
                return f"闭集对齐：「{text}」→「{canon}」"
    # 通用：只说「原文 → 参数」（参数和原文一样就没什么可说的）
    if arg and arg != text:
        return f"闭集对齐：「{text}」→ 参数「{arg}」"
    return ""


def tool_doctor(cfg) -> int:
    """环境自检：把运维域依赖的每条链路实际跑一遍。

    ⚠️ 为什么需要：ops 的动作全在【外部进程 / 网络服务】上，出问题时日志往往
       只有一句「失败」，很难分清是 WSL 没通、kubectl 不在、Prometheus 挂了、
       还是模型文件缺了。这里一次全查并指出下一步。
       也是排查「frozen exe 里 wsl.exe 拒绝访问」时加的。
    """
    import os as _os
    import urllib.request
    from pathlib import Path as _P

    def line(ok, name, detail=""):
        print(f"  {'✅' if ok else '❌'} {name:<24} {detail}")

    print("\n" + "=" * 76)
    print("  环境自检（--doctor）")
    print("=" * 76)
    print(f"  frozen         = {getattr(sys, 'frozen', False)}")
    print(f"  sys.executable = {sys.executable}")
    print(f"  __file__       = {__file__}")
    print(f"  cwd            = {_os.getcwd()}")
    print(f"  SystemRoot     = {_os.environ.get('SystemRoot')}")
    print(f"  PATH           = {_os.environ.get('PATH', '')[:120]}")

    base = (_P(sys.executable).resolve().parent if getattr(sys, "frozen", False)
            else _P(__file__).resolve().parent)

    print("\n  ① 模型产物（纯文本路要的）")
    for f in ("model/intent_text.pt", "model/_vocab.txt", "model/meta.json",
              "model/intent.pt", "model/bundle.safetensors"):
        p = base / f
        line(p.exists(), f, f"{p.stat().st_size:,} B" if p.exists() else "缺失")

    print("\n  ② 句向量索引")
    rel = str(cfg.behavior.get("intent_index", "intent_index.json"))
    idx = base / rel
    line(idx.exists(), rel,
         f"{idx.stat().st_size:,} B" if idx.exists() else f"缺失 —— 找的是 {idx}")

    print("\n  ③ WSL（ops 的 kubectl / ps / systemctl 全靠它）")
    try:
        from ops.game_actions import _run_wsl, _wsl_exe
        print(f"     wsl.exe 路径 = {_wsl_exe()}")
        ok, out = _run_wsl("echo doctor-ok", timeout=20)
        line(ok, "wsl.exe echo", out.replace("\n", " ")[:100])
        if ok:
            ok2, out2 = _run_wsl("kubectl get nodes --no-headers | head -3", timeout=30)
            line(ok2, "kubectl", out2.replace("\n", " ")[:100])
    except Exception as e:
        line(False, "wsl 检查", f"{type(e).__name__}: {e}")

    print("\n  ④ 子进程（桌面域的关闭程序 / ops 的 kubectl 都靠它）")
    import subprocess as _sp
    try:
        from ops.game_actions import WSL_DISTRO as _WD
    except Exception:
        _WD = "Ubuntu-24.04"
    for name, cmd in (("cmd.exe echo", ["cmd.exe", "/c", "echo child-ok"]),
                      ("where wsl.exe", ["cmd.exe", "/c", "where wsl.exe"]),
                      ("wsl.exe 直接起", [r"C:\Windows\System32\wsl.exe",
                                          "-d", _WD, "--", "echo", "wsl-direct"]),
                      ("tasklist", ["tasklist", "/FI", "IMAGENAME eq explorer.exe"]),
                      ("powershell", ["powershell", "-NoProfile", "-Command", "echo ps-ok"])):
        try:
            r = _sp.run(cmd, capture_output=True, timeout=20)
            o = (r.stdout or b"")[:50]
            e = (r.stderr or b"")[:120]
            if e and e.count(b"\x00") > len(e) // 4:
                e = e.decode("utf-16-le", errors="replace").encode("utf-8")[:120]
            line(r.returncode == 0, name,
                 f"rc={r.returncode} out={o!r} err={e!r}")
        except Exception as e:
            line(False, name, f"{type(e).__name__}: {e}")

    print("\n  ⑤ 网络服务")
    # ⚠️ Prometheus / Grafana 是 NodePort（iptables DNAT，没有监听套接字），
    #    从 Windows 侧连不上 —— 所以 ops 的 PromQL 一直走 WSL curl（见 game_actions
    #    里的「NodePort 血债」注释）。doctor 也照这个方式来，否则会误报。
    for name, url in (("ASR 30805", "http://127.0.0.1:30805/v1/models"),
                      ("embed 30802", "http://172.20.71.101:30802/v1/models")):
        try:
            with urllib.request.urlopen(url, timeout=6) as r:
                line(True, name, f"HTTP {r.status}（Windows 直连）")
        except Exception as e:
            line(False, name, type(e).__name__)
    try:
        from ops.game_actions import _run_wsl as _rw
        for name, url in (("Prometheus 30090", "http://127.0.0.1:30090/-/healthy"),
                          ("Grafana 30030", "http://127.0.0.1:30030/api/health")):
            ok, out = _rw(f"curl -s -o /dev/null -w '%{{http_code}}' -m 6 {url}", timeout=15)
            line(ok and out.strip().startswith("2"), name, f"HTTP {out.strip()}（WSL curl）")
    except Exception as e:
        line(False, "Prometheus/Grafana", f"{type(e).__name__}: {e}")

    print("\n" + "=" * 76)
    return 0


def tool_query(cfg: Config, text):
    """纯文本 → 意图 + 参数 + 将要执行的动作（不执行）"""
    it, arg = parse(text, cfg)
    arg = resolve(it, text, cfg)
    hit = resolve_closed(it, arg, cfg)
    if it in ("launch_app", "open_url", "open_file"):
        if not hit:
            return {"input": text, "intent": "unknown", "arg": arg,
                    "reason": f"闭集里没有「{arg}」", "would_do": None}
    action = {"launch_app": "启动程序", "open_url": "浏览器打开", "open_file": "打开文件",
              "open_web": "浏览器打开/搜索", "web_search": "浏览器搜索",
              "file_search": "文件管理器搜索", "exact_search": "精确查找(转写原文)",
              "unknown": "不执行"}.get(it, "?")
    target = None
    if hit:
        target = hit.get("path") or hit.get("url")
    elif it == "web_search":
        target = cfg.search["web_engine"].replace("{q}", arg)
    elif it == "file_search":
        target = f"search-ms:query={arg}  (范围 {cfg.search.get('file_manager_root')})"
    elif it == "exact_search":
        target = cfg.search["web_engine"].replace("{q}", arg)
    # ⚠️ 血债（2026-10-14 测出）：原来写 hit["id"]，而 resolve_closed() 在
    #    web_search / exact_search 这类"没有闭集条目"的路径上返回的对象没有 id 键
    #    → KeyError → `--query "打开B站吧！打开B站。"` 直接崩。
    #    语音主链路 handle() 不走 tool_query，所以线上没受影响。
    return {"input": text, "intent": it, "arg": arg, "action": action,
            "target": target,
            "resolved": (hit.get("id") if isinstance(hit, dict) else None)}


def tool_add(cfg: Config, kind, name, target, aliases=None, dry=False):
    """往配置里加一条。kind = app | url | file"""
    if kind not in ("app", "url", "file"):
        return {"ok": False, "error": f"kind 必须是 app/url/file，收到 {kind}"}
    key = {"app": "apps", "url": "urls", "file": "files"}[kind]
    lst = cfg.raw[key]
    cid = name.lower().replace(" ", "_")
    if any(x["id"] == cid for x in lst):
        return {"ok": False, "error": f"id 「{cid}」 已存在"}
    al = aliases if aliases else [name]
    entry = {"id": cid, "name": name, "aliases": al}
    entry["url" if kind == "url" else "path"] = target
    lst.append(entry)
    if not dry:
        cfg.path.write_text(json.dumps(cfg.raw, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    return {"ok": True, "added": entry, "kind": kind, "written": not dry,
            "note": "不用重训模型 —— 规则解析器会直接查闭集"}


def tool_remove(cfg: Config, cid, dry=False):
    for key in ("apps", "urls", "files"):
        lst = cfg.raw[key]
        for i, x in enumerate(lst):
            if x["id"] == cid or x["name"] == cid:
                rm = lst.pop(i)
                if not dry:
                    cfg.path.write_text(json.dumps(cfg.raw, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
                return {"ok": True, "removed": rm, "from": key, "written": not dry}
    return {"ok": False, "error": f"找不到 id/name = {cid}"}


# ══════════════════════════════════════════════════════════════
_CODE_FP = None


def _code_fp():
    """关键源文件指纹（前 8 位十六进制）。

    为什么需要：这个项目里"跑的是哪一版"被反复搞混 —— 日志显示崩溃、但代码其实
    已经修好（旧日志），或者以为重启了、其实窗口里还是旧进程。启动横幅里打一个
    指纹，配合 git log 就能一眼定位。
    """
    global _CODE_FP
    if _CODE_FP is None:
        import hashlib
        h = hashlib.md5()
        here = Path(__file__).parent
        for f in ("voice_agent.py", "vad_silero.py", "vad.py",
                  "model_runner.py", "actions.py", "asr_gguf.py", "dlog.py"):
            try:
                h.update(f.encode() + b"\0")
                h.update((here / f).read_bytes())
            except Exception:
                pass
        _CODE_FP = h.hexdigest()[:8]
    return _CODE_FP


def _split_c_stderr():
    r"""把 C 层（llama.cpp / mtmd / torch / PortAudio）的 stderr 分流到独立文件。

    这些库大多绕过 Python 的 sys.stderr、直接写 fd 2（实测占日志 78~92% 行数，
    而且没有时间戳、和我们的日志交错在一起，看起来像报错）。
    `llama_log_set` 只能拦下走 ggml/llama 回调的那部分，clip loader 与 mtmd 的
    `add_text:` / `audio_tokens->` / `clip_encode:` 是直接写的 —— 所以这里在
    进程最早期把 fd 2 指向独立文件：主日志只剩应用层，C 层仍留档可查。

    ⚠️ 两个踩过的坑（都让人误判成"又崩了"）：
      1) 用 O_APPEND 追加 → 上一次崩溃的 traceback 留在文件里，
         下一次看日志时看到旧堆栈，以为新进程又崩在同一处（实测踩中）；
      2) 文件里没有任何版本/时间标识 → 无法判断看到的是哪一版、哪一次运行。
    所以：每次启动把旧文件滚成 .prev，新文件头写明启动时间/pid/代码指纹。
    """
    path = os.environ.get("VOICE_AGENT_C_STDERR")
    if not path:
        return
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            prev = path + ".prev"
            try:
                if os.path.exists(prev):
                    os.remove(prev)
                os.replace(path, prev)
            except Exception:
                pass
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        hdr = (f"=== C 层 stderr（llama.cpp / mtmd / torch / PortAudio）\n"
               f"=== 启动 {time.strftime('%Y-%m-%d %H:%M:%S')}  pid={os.getpid()}"
               f"  代码指纹 {_code_fp()}\n"
               f"=== 本文件只含【本次运行】的 C 层输出；上次运行已滚到 .prev\n")
        os.write(fd, hdr.encode("utf-8", "replace"))
        os.dup2(fd, 2)
        os.close(fd)
    except Exception:
        pass


def main():
    _split_c_stderr()
    ap = argparse.ArgumentParser(add_help=True,
                                 description="语音意图识别（常开流式；开关见 intent_config.json → intent_registry）")
    ap.add_argument("--config", default=None, help="意图配置文件路径")
    ap.add_argument("--model-dir", default=None, help="模型目录（含 intent.pt / bundle.safetensors）")
    ap.add_argument("--device", default=None, help="音频输入设备号")
    ap.add_argument("--device-type", default="cpu", choices=["cpu", "cuda"], help="推理设备")
    ap.add_argument("--list", action="store_true", help="只列出所有可执行命令")
    ap.add_argument("--dry-run", action="store_true", help="不真的执行，只打印")
    ap.add_argument("--wav", nargs="*", default=None, help="跑音频文件（不用麦克风）")
    ap.add_argument("--selftest", action="store_true", help="跑规则分句自测")
    ap.add_argument("--doctor", action="store_true",
                    help="环境自检：wsl.exe / kubectl / Prometheus / ASR / 句向量 / 模型文件")
    # ── 工具接口（给 LLM / 强 AI 用，不需要麦克风）──
    ap.add_argument("--json", action="store_true", help="输出 JSON（配合 --list/--query 用）")
    ap.add_argument("--query", default=None, metavar="TEXT",
                    help="纯文本查询：这条话会判成什么意图、执行什么（不执行）")
    ap.add_argument("--add-app", nargs=2, metavar=("NAME", "PATH"),
                    help="加一个程序（不用重训）")
    ap.add_argument("--add-url", nargs=2, metavar=("NAME", "URL"),
                    help="加一个网址（不用重训）")
    ap.add_argument("--add-file", nargs=2, metavar=("NAME", "PATH"),
                    help="加一个文件（不用重训）")
    ap.add_argument("--remove", default=None, metavar="ID_OR_NAME", help="删一条")
    args = ap.parse_args()

    cfg = Config(args.config)

    # ── 工具接口（先处理，不加载模型）──
    if args.add_app or args.add_url or args.add_file:
        kind, pair = (("app", args.add_app) if args.add_app else
                      ("url", args.add_url) if args.add_url else ("file", args.add_file))
        r = tool_add(cfg, kind, pair[0], pair[1])
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return 0 if r["ok"] else 1
    if args.remove:
        r = tool_remove(cfg, args.remove)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return 0 if r["ok"] else 1
    if args.query:
        r = tool_query(cfg, args.query)
        if args.json:
            print(json.dumps(r, ensure_ascii=False, indent=1))
        else:
            print(f"  输入  {r['input']}")
            print(f"  意图  {r['intent']}")
            print(f"  参数  {r['arg']!r}")
            print(f"  动作  {r.get('action')}")
            print(f"  目标  {r.get('target')}")
            if r.get("reason"):
                print(f"  原因  {r['reason']}")
        return 0
    if args.doctor:
        return tool_doctor(cfg)
    if args.list:
        if args.json:
            print(json.dumps(tool_commands_json(cfg), ensure_ascii=False, indent=1))
        else:
            print(list_commands(cfg))
        return 0
    if args.selftest:
        print("\n" + "=" * 78)
        print("  分句自测（纯规则）")
        print("=" * 78)
        cases = ["启动记事本", "打开B站", "打开B站网页", "打开CUDA编程指南文件",
                 "打开github.com", "打开一个不存在的东西", "搜索python教程",
                 "搜索文件季度报告", "精确查找resnet50", "今天天气怎么样"]
        for c in cases:
            it, arg = parse(c, cfg)
            ok, msg, _ = dispatch(it, arg, cfg, dry=True)
            print(f"  {c:26s} → {it:14s} {msg}")
        return 0

    print(banner_for(cfg))
    log(f"代码指纹 {_code_fp()}  （voice_agent/vad_silero/vad/model_runner/"
        f"actions/asr_gguf/dlog 的 MD5 前 8 位，用来确认跑的是哪一版）", "SYS")
    dry = True if args.dry_run else None
    agent = Agent(cfg, model_dir=args.model_dir, device=args.device_type, dry=dry)
    log(f"VAD 参数（实际生效）：{agent.vad_summary()}", "SYS")
    log(f"预滚/裁剪参数（实际生效）：{agent.trim_summary()}", "SYS", level="info")
    if dry:
        log("⚠️  DRY-RUN 模式：只打印，不真的启动程序/开浏览器")

    if args.wav:
        return agent.run_files(args.wav)
    dev = int(args.device) if args.device is not None else None
    return agent.run_mic(dev)


if __name__ == "__main__":
    sys.exit(main())
