# -*- coding: utf-8 -*-
r"""用它自己的解码器转写 —— Qwen3-ASR-0.6B 全量（含 28 层 Qwen3 解码器）

对比对象：现在 built 走的是「A(audio_tower) → 自训 ctc_head 贪心解码」，
本模块走「A → 它自己的 28 层解码器自回归生成」（即设计里被拆掉的那 751.63M 参数）。

架构（E:\models\qwen3-asr-0.6b-official\config.json 实测）：
  Qwen3ASRForConditionalGeneration
  ├─ audio_tower : qwen3_asr_audio_encoder，18 层，128 mel，output_dim 1024（= 设计里的 A）
  └─ text_config : qwen3 解码器，28 层 / hidden 1024 / 16 heads / 8 KV heads(GQA) /
                   head_dim 128 / intermediate 3072 / vocab 151936 / RoPE θ=1e6 /
                   RMSNorm / SiLU / tie_word_embeddings=True
  参数：thinker.model 596.05M + audio_tower 186.38M + lm_head 155.58M = 938M（bf16 1876 MB）

用法（独立自测）:
  E:\voice_agent_build\Scripts\python.exe asr_own.py wavs\train_launch_app_0000.wav
"""
import importlib
import os
import time

import numpy as np

try:
    from dlog import log as _log
except Exception:
    def _log(msg="", tag=None, sub=None):
        print(f"{('[' + tag + '] ') if tag else ''}{msg}", flush=True)

DEFAULT_DIR = r"E:\models\qwen3-asr-0.6b-official"


def _find_class(names):
    for mod_path in ("transformers", "transformers.models.qwen3_asr",
                     "transformers.models.qwen3_asr.modeling_qwen3_asr",
                     "transformers.models.qwen3_asr.processing_qwen3_asr"):
        try:
            mod = importlib.import_module(mod_path)
        except Exception:
            continue
        for n in names:
            cls = getattr(mod, n, None)
            if cls is not None:
                return cls
    return None


class OwnASR:
    """用模型自己的解码器做转写（自回归 + 语言模型，会纠错、会断句）"""

    def __init__(self, model_dir=DEFAULT_DIR, device="cpu", dtype="float32",
                 language="Chinese", prompt=None, max_new_tokens=128,
                 share_encoder=None):
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.language = language
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        # 复用外部已有的 audio_tower（意图模型里的 A），避免装第二份 186.38M
        self.share_encoder = share_encoder
        self.model = None
        self.proc = None
        self.load_ms = 0.0
        self.last_ms = 0.0
        self.last_tokens = 0

    def load(self):
        if self.model is not None:
            return
        import json as _json
        import torch
        from safetensors.torch import load_file
        from transformers import AutoProcessor
        from transformers.models.qwen3_asr.configuration_qwen3_asr import (
            Qwen3ASRConfig, Qwen3ASREncoderConfig)

        t0 = time.perf_counter()

        # ⚠️⚠️ 这份 checkpoint 的嵌套配置挂在 `thinker_config` 下（Qwen3-Omni 系布局），
        #      而 transformers 的 Qwen3ASRConfig 只认 `audio_config`/`text_config`
        #      → from_pretrained() 读不到，静默用默认值（24 层音频塔 / output_dim 3584），
        #      结果【全部权重 UNEXPECTED/MISSING，加载出来是个随机模型】——
        #      而且不报错！所以必须显式构造 config + 按名映射装权重。
        raw = _json.loads(open(os.path.join(self.model_dir, "config.json"),
                               encoding="utf-8").read())
        tc = raw.get("thinker_config", raw)
        acfg_ok = set(Qwen3ASREncoderConfig().to_dict().keys())
        acfg = {k: v for k, v in tc["audio_config"].items() if k in acfg_ok}
        # 老 checkpoint 写的是 qwen3_asr_audio_encoder，transformers 注册名是 qwen3_asr_encoder
        acfg["model_type"] = "qwen3_asr_encoder"
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        tcfg_ok = set(Qwen3Config().to_dict().keys())
        tcfg = {k: v for k, v in tc["text_config"].items() if k in tcfg_ok}
        tcfg.setdefault("tie_word_embeddings", True)
        cfg = Qwen3ASRConfig(
            audio_config=acfg, text_config=tcfg,
            audio_token_id=tc.get("audio_token_id"),
            timestamp_token_id=tc.get("timestamp_token_id", 151705),
        )
        ModelCls = _find_class(["Qwen3ASRForConditionalGeneration"])
        self.model = ModelCls(cfg)

        sd = load_file(os.path.join(self.model_dir, "model.safetensors"))
        shared = self.share_encoder is not None
        mapped, skipped = {}, 0
        for k, v in sd.items():
            # ⚠️ 投影层改名：checkpoint 是 audio_tower.proj1/proj2，
            #    类里叫 multi_modal_projector.linear_1/linear_2（形状一致 896→896、896→1024）。
            #    漏了这条 → 投影层随机初始化 → 音频特征无意义 → 解码器输出通顺但内容随机
            #    （实测就是这样：真值「现在打开远程桌面」被解码成「我们有新的方法。」）
            if k.startswith("thinker.audio_tower.proj"):
                rest = k[len("thinker.audio_tower.proj"):]      # "1.weight" / "2.bias"
                mapped["model.multi_modal_projector.linear_" + rest] = v
            elif k.startswith("thinker.audio_tower."):
                if shared:
                    skipped += 1        # 复用意图模型的 A，不再装第二份 186.38M
                    continue
                mapped["model.audio_tower." + k[len("thinker.audio_tower."):]] = v
            elif k.startswith("thinker.model."):
                mapped["model.language_model." + k[len("thinker.model."):]] = v
            elif k == "thinker.lm_head.weight":
                mapped["lm_head.weight"] = v
            else:
                skipped += 1
        missing, unexpected = self.model.load_state_dict(mapped, strict=False)
        if shared:
            old = self.model.model.audio_tower
            self.model.model.audio_tower = self.share_encoder
            del old
            import gc
            gc.collect()
            missing = [m for m in missing if "audio_tower" not in m]
        miss_layers = [m for m in missing if ".layers." in m or "embed" in m]
        _log(f"映射 {len(mapped)} 张量（跳过 {skipped}，共享 audio_tower={shared}）"
             f"  missing {len(missing)} / unexpected {len(unexpected)}", "ASR")
        if miss_layers[:3]:
            _log(f"关键缺失: {miss_layers[:3]}", "WARN", sub=True)
        self._install_af_cache()
        self.model.eval()
        self.model.to(self.device)

        ProcCls = _find_class(["Qwen3ASRProcessor"])
        self.proc = (AutoProcessor.from_pretrained(self.model_dir)
                     if ProcCls is None else ProcCls.from_pretrained(self.model_dir))
        self.load_ms = (time.perf_counter() - t0) * 1000
        n = sum(p.numel() for p in self.model.parameters())
        _log(f"torch 载入 {n/1e6:.1f} M 参数 (fp32)  {self.load_ms/1000:.1f}s", "ASR")

    def _install_af_cache(self):
        """给 get_audio_features 加记忆化 + 计数器。

        目的有二：
          1) 量化"编码到底算了几次"（不猜）。若一次转写里 >1 次，说明生成过程中
             audio_tower 被反复重算，这里就顺手把它压成 1 次；
          2) 提供 af_calls 供基准/日志核对。
        """
        model = self.model
        if getattr(model, "_af_installed", False):
            return
        orig = model.get_audio_features

        def wrapper(*a, **kw):
            model._af_calls = getattr(model, "_af_calls", 0) + 1
            hit = getattr(model, "_af_cache", None)
            if hit is None:
                hit = orig(*a, **kw)
                model._af_cache = hit
            return hit

        model.get_audio_features = wrapper
        model._af_installed = True

    # ── 卸载（动态加载用）──
    def unload(self):
        self.model = None
        self.proc = None
        import gc
        gc.collect()

    def transcribe(self, wav):
        """wav: float32 单声道 16 kHz numpy → 文本"""
        self.load()
        import torch
        wav = np.asarray(wav, dtype=np.float32).ravel()
        # ⚠️ 这个编码器要求 mel 帧数是 n_window*2=100 的倍数（= 整秒）。
        #    voice_agent 里是 pad 特征绕过的；走官方处理器只能 pad 音频样本，
        #    否则报 "expects padded_feature_length to be a multiple of 100, but got 280"。
        pad = (-len(wav)) % 16000
        if pad:
            wav = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])
        if getattr(self.model, "_af_installed", False):
            self.model._af_cache = None      # 每次转写重置缓存
            self.model._af_calls = 0
        t0 = time.perf_counter()

        # ⚠️ 本仓库这份 chat_template.json 只渲染 system+user+生成提示，
        #    不渲染 assistant 段 —— 所以处理器自带的 apply_transcription_request()
        #    （它靠 continue_final_message 预填 "language X<asr_text>"）会直接报错。
        #    改走标准路径：add_generation_prompt=True，然后自己把语言提示接到 input_ids 后面。
        msgs = []
        if self.prompt:
            msgs.append({"role": "system", "content": self.prompt})
        msgs.append({"role": "user",
                     "content": [{"type": "audio", "audio": wav}]})
        inputs = self.proc.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}

        n_prompt = int(inputs["input_ids"].shape[1])
        if self.language:
            pre = self.proc.tokenizer.encode(
                f"language {self.language}<asr_text>", add_special_tokens=False)
            if pre:
                pre_t = torch.tensor([pre], dtype=inputs["input_ids"].dtype,
                                     device=inputs["input_ids"].device)
                inputs["input_ids"] = torch.cat([inputs["input_ids"], pre_t], dim=1)
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.cat(
                        [inputs["attention_mask"],
                         torch.ones_like(pre_t)], dim=1)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        gen = out[0, int(inputs["input_ids"].shape[1]):]
        self.last_tokens = int(gen.shape[0])
        txt = self.proc.decode(gen, return_format="transcription_only")
        if isinstance(txt, list):
            txt = txt[0] if txt else ""
        self.last_ms = (time.perf_counter() - t0) * 1000
        return str(txt).strip()


def _load_wav(path):
    import wave
    with wave.open(str(path), "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


if __name__ == "__main__":
    import sys
    asr = OwnASR()
    paths = sys.argv[1:] or [str(os.path.join(os.path.dirname(__file__), "wavs",
                                              "train_launch_app_0000.wav"))]
    for p in paths:
        w, sr = _load_wav(p)
        assert sr == 16000, f"{p} sr={sr}"
        t = asr.transcribe(w)
        print(f"  {os.path.basename(p):44s} {len(w)/16000:5.2f}s → "
              f"「{t}」  ({asr.last_ms:.0f} ms)")
