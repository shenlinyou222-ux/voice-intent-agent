# -*- coding: utf-8 -*-
"""意图模型推理（V5 集成架构）—— 自包含，供 voice_agent.py 和 exe 使用

    wav → A(18层) → ctc_head → logits
        ├─ V1: p=softmax(logits/τ) → z=p@E_pad → 压缩r=3 → [CLS,z] → B → CLS1
        └─ V3: CTC解码 → 字符串 → dmeta分词 → 精确行 → B → CLS2
        → concat(CLS1,CLS2) → intent_head → N 类

模型权重全部放在一个 safetensors 里（约 283 MB），跟 exe 分开，方便换模型。
"""
import io
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dlog import log as _log
except Exception:                                     # 独立跑时兜底
    def _log(msg="", tag=None, sub=None):
        print(f"{('[' + tag + '] ') if tag else ''}{msg}", flush=True)

# ⭐ 冻结（PyInstaller）后 __file__ 指向 _internal\，不是 exe 所在目录，
#   所以配置/模型必须按 sys.executable 找。不处理的话会报
#   FileNotFoundError: ...\dist\voice_agent\_internal\intent_config.json
if getattr(sys, "frozen", False):
    HERE = Path(sys.executable).resolve().parent
else:
    HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = HERE / "model"
DEFAULT_ASR_DIR = Path(r"E:\models\qwen3-asr-0.6b-official")


def read_wav_bytes(raw):
    """从 bytes 读 wav → float32 mono 16k"""
    with wave.open(io.BytesIO(raw), "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        data = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        x = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    if sr != 16000:
        t = np.linspace(0, len(x) / sr, int(len(x) * 16000 / sr), endpoint=False)
        x = np.interp(t, np.arange(len(x)) / sr, x).astype(np.float32)
    return np.ascontiguousarray(x)


def read_wav(path):
    with open(path, "rb") as f:
        return read_wav_bytes(f.read())


class IntentModel:
    def __init__(self, model_dir=None, asr_dir=None, device="cpu", verbose=True,
                 int8=False, text_only=False):
        """int8=True 时直接加载离线打好的 int8 产物（进程内不出现 fp32 权重）。
        text_only=True 时加载【纯文本路】产物：只有 B(int8)+word_emb+head_text，
        不含音频塔 —— 音频塔在 ASR 侧 mmproj 里已经有一个。
        """
        from transformers import AutoTokenizer, BertConfig, BertModel  # noqa: F401
        from transformers.models.qwen3_asr.feature_extraction_qwen3_asr import Qwen3ASRFeatureExtractor
        from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASREncoder
        from transformers.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASREncoderConfig

        self.device = device
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        asr_dir = Path(asr_dir) if asr_dir else DEFAULT_ASR_DIR
        self.asr_dir = asr_dir
        t0 = time.time()
        self.text_only = bool(text_only)
        if text_only:
            tp = self.model_dir / "intent_text.pt"
            if tp.exists():
                self._init_text(tp, verbose, t0)
                return
            if verbose:
                print(f"[model] 没找到 {tp.name}，回退完整 int8 产物", flush=True)
            self.text_only = False
        i8 = self.model_dir / "intent_int8b.pt"
        if not i8.exists():
            i8 = self.model_dir / "intent_int8.pt"
        if int8 and i8.exists():
            self._init_int8(i8, asr_dir, verbose, t0)
            return
        if int8 and verbose:
            print(f"[model] 没找到 {i8.name}，回退 fp32 bundle", flush=True)

        from safetensors.torch import load_file
        meta = json.loads((self.model_dir / "meta.json").read_text(encoding="utf-8"))
        self.INTENTS = meta["intents"]
        self.TAU = float(meta["tau"])
        self.R = int(meta["R"])
        self.K = int(meta["K"])

        ck = torch.load(str(self.model_dir / "intent.pt"), map_location="cpu", weights_only=False)
        self._ck = ck
        # shared_chars 存在 intent.pt 里（meta.json 只放标量）
        self.SHARED = ck["shared_chars"] if "shared_chars" in ck else meta["shared_chars"]
        self.K = len(self.SHARED)
        self.char2i = {c: i + 1 for i, c in enumerate(self.SHARED)}
        bundle = self.model_dir / "bundle.safetensors"
        self.bundle_path = bundle
        if not bundle.exists():
            raise FileNotFoundError(f"缺模型包 {bundle}，先跑 package_model.py")

        # ── 从 bundle 载入 ──
        SD = load_file(str(bundle))
        self._sd = SD
        bert_cfg = BertConfig(**json.loads(
            SD["bert_config_json"].numpy().tobytes().decode("utf-8")))
        vocab_txt = SD["vocab_txt"].numpy().tobytes().decode("utf-8")
        _vt = self.model_dir / "_vocab.txt"
        if not _vt.exists():
            _vt.write_text(vocab_txt, encoding="utf-8")

        # ── A ──
        raw = json.loads((asr_dir / "config.json").read_text(encoding="utf-8"))
        acfg = dict(raw["thinker_config"]["audio_config"])
        allowed = set(Qwen3ASREncoderConfig().to_dict().keys())
        self.A = Qwen3ASREncoder(Qwen3ASREncoderConfig(**{k: v for k, v in acfg.items() if k in allowed}))
        self.A.load_state_dict({k[len("audio_tower."):]: v for k, v in SD.items()
                                if k.startswith("audio_tower.")}, strict=False)
        # ── B ──
        self.bert = BertModel(bert_cfg, add_pooling_layer=False)
        self.bert.encoder.load_state_dict(
            {k[len("text_encoder.encoder."):]: v for k, v in SD.items()
             if k.startswith("text_encoder.encoder.")}, strict=False)
        self.bert.embeddings.load_state_dict(
            {k: SD["bert_emb." + k] for k in
             ("position_embeddings.weight", "token_type_embeddings.weight",
              "LayerNorm.weight", "LayerNorm.bias")}, strict=False)
        with torch.no_grad():
            self.bert.embeddings.word_embeddings.weight.data = SD["bert_word_emb"].clone()
        self.E_ALL = self.bert.embeddings.word_embeddings.weight.data
        # E_pad = 交集字符的精确词嵌入行（第 0 行留给 blank）
        # ⭐ 整词 token 的行 = 该词在 dmeta 分词后所有 token 嵌入的【均值】
        #    （见 train_wordtoken.py，这样 V1 的软查表路径也能用）
        _bl = vocab_txt.splitlines()
        _b2i = {}
        for _i, _t in enumerate(_bl):
            _t = _t.replace("##", "")
            if len(_t) == 1:
                _b2i.setdefault(_t, _i)
        from transformers import BertTokenizer
        self.tok = BertTokenizer(str(_vt), do_lower_case=True)
        _rows = [torch.zeros(1, 768)]
        for _c in self.SHARED:
            if _c.startswith("<w:"):
                _ids = self.tok(_c[3:-1], add_special_tokens=False)["input_ids"]
                _rows.append(self.E_ALL[_ids].mean(0, keepdim=True) if _ids
                             else torch.zeros(1, 768))
            else:
                _rows.append(self.E_ALL[_b2i[_c]].unsqueeze(0))
        self.E_PAD = torch.cat(_rows, 0)
        self.CLS_ROW = self.E_ALL[_bl.index("[CLS]")]

        # ── 头（ctc_head + intent_head）──
        self.ctc = nn.Linear(896, self.K + 1)
        self.ctc.weight.data = ck["ctc_head"]; self.ctc.bias.data = ck["ctc_bias"]
        self.head = nn.Linear(1536, len(self.INTENTS))
        self.head.weight.data = ck["head_w"]; self.head.bias.data = ck["head_b"]

        for m in (self.A, self.bert, self.ctc, self.head):
            m.to(device).eval()
            for p in m.parameters():
                p.requires_grad = False
        self.E_PAD = self.E_PAD.to(device)
        self.CLS_ROW = self.CLS_ROW.to(device)
        self.E_ALL = self.E_ALL.to(device)
        self.fe = Qwen3ASRFeatureExtractor.from_pretrained(str(asr_dir))
        # ⚠️ 载完就把 safetensors 原始字典丢掉：load_state_dict / 逐张量赋值都是
        #    【拷贝】语义，字典里那份副本之后没人再用。
        #    实测：_sd 白占 781 MB（占进程 RSS 的 17%），del 后 RSS 立刻 -742 MB。
        #    全仓库只有本文件 78/90 两行赋值、之后零引用。
        self._sd = None
        self._ck = None
        if verbose:
            n = (sum(p.numel() for p in self.A.parameters())
                 + sum(p.numel() for p in self.bert.parameters())
                 + sum(p.numel() for p in self.ctc.parameters())
                 + sum(p.numel() for p in self.head.parameters()))
            _log(f"{len(self.INTENTS)} 类  {n/1e6:.2f} M 参数(fp32)  "
                 f"{time.time()-t0:.1f}s", "MODEL")

    # ── int8 产物加载：进程内只出现 int8 权重（A / B encoder），词表保持 fp32 ──
    def _init_int8(self, path, asr_dir, verbose, t0):
        import torch.nn as nn
        from transformers import BertConfig, BertModel, BertTokenizer
        from transformers.models.qwen3_asr.feature_extraction_qwen3_asr import Qwen3ASRFeatureExtractor
        from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASREncoder
        from transformers.models.qwen3_asr.configuration_qwen3_asr import Qwen3ASREncoderConfig
        from torch.ao.quantization import quantize_dynamic, default_dynamic_qconfig
        QLIN = {nn.Linear: default_dynamic_qconfig}
        QCONV = {nn.Linear: default_dynamic_qconfig, nn.Conv1d: default_dynamic_qconfig}

        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        self.INTENTS = ck["intents"]
        self.TAU = float(ck["tau"])
        self.R = int(ck["R"])
        self.SHARED = ck["shared_chars"]
        self.K = len(self.SHARED)
        self.char2i = {c: i + 1 for i, c in enumerate(self.SHARED)}

        # A：优先用打包好的【量化模块对象】（加载时完全不物化 fp32）；
        #    没有就退回"建结构→量化→灌 state_dict"（会短暂出现 fp32）
        if "A_module" in ck:
            self.A = ck["A_module"]
        else:
            allowed = set(Qwen3ASREncoderConfig().to_dict().keys())
            acfg = {k: v for k, v in ck["asr_audio_config"].items() if k in allowed}
            self.A = Qwen3ASREncoder(Qwen3ASREncoderConfig(**acfg))
            self.A = quantize_dynamic(self.A, QCONV, dtype=torch.qint8)
            self.A.load_state_dict(ck["A_q"])

        # B：encoder 走 int8，词嵌入表保持 fp32（E_ALL / E_PAD 要查表）
        # ⚠️ 关键：先把壳的层数设成 0 —— 反正马上要用 int8 encoder 整个替换掉，
        #    不设的话会先分配 340 MB 的 fp32 encoder，替换后分配器只回收 ~30 MB
        #    （实测该步 +389 MB，纯白占）。
        bcfg = BertConfig(**ck["bert_config"])
        n_layers_real = bcfg.num_hidden_layers
        if "Benc_module" in ck:
            bcfg.num_hidden_layers = 0
        self.bert = BertModel(bcfg, add_pooling_layer=False)
        emb = dict(ck["bert_emb"])
        emb["word_embeddings.weight"] = ck["word_emb"]
        self.bert.embeddings.load_state_dict(emb, strict=False)
        if "Benc_module" in ck:
            self.bert.encoder = ck["Benc_module"]
            self.bert.config.num_hidden_layers = n_layers_real   # 还原配置
        else:
            self.bert.encoder = quantize_dynamic(self.bert.encoder, QLIN,
                                                 dtype=torch.qint8)
            self.bert.encoder.load_state_dict(ck["Benc_q"])
        self.E_ALL = self.bert.embeddings.word_embeddings.weight.data

        _vt = self.model_dir / "_vocab.txt"
        _bl = _vt.read_text(encoding="utf-8").splitlines()
        _b2i = {}
        for _i, _t in enumerate(_bl):
            _t = _t.replace("##", "")
            if len(_t) == 1:
                _b2i.setdefault(_t, _i)
        self.tok = BertTokenizer(str(_vt), do_lower_case=True)
        _rows = [torch.zeros(1, 768)]
        for _c in self.SHARED:
            if _c.startswith("<w:"):
                _ids = self.tok(_c[3:-1], add_special_tokens=False)["input_ids"]
                _rows.append(self.E_ALL[_ids].mean(0, keepdim=True) if _ids
                             else torch.zeros(1, 768))
            else:
                _rows.append(self.E_ALL[_b2i[_c]].unsqueeze(0))
        self.E_PAD = torch.cat(_rows, 0)
        self.CLS_ROW = self.E_ALL[_bl.index("[CLS]")]

        self.ctc = nn.Linear(896, self.K + 1)
        self.ctc.weight.data = ck["ctc_head"]
        self.ctc.bias.data = ck["ctc_bias"]
        self.head = nn.Linear(1536, len(self.INTENTS))
        self.head.weight.data = ck["head_w"]
        self.head.bias.data = ck["head_b"]

        for _m in (self.A, self.bert, self.ctc, self.head):
            _m.to(self.device).eval()
            for _p in _m.parameters():
                _p.requires_grad = False
        self.E_PAD = self.E_PAD.to(self.device)
        self.CLS_ROW = self.CLS_ROW.to(self.device)
        self.E_ALL = self.E_ALL.to(self.device)
        self.fe = Qwen3ASRFeatureExtractor.from_pretrained(str(asr_dir))
        if verbose:
            va = ck.get("val_acc_int8")
            _extra = f"  val(重训头) {va*100:.1f}%" if va else ""
            _log(f"int8 产物 {path.name}  {len(self.INTENTS)} 类{_extra}  "
                 f"{time.time()-t0:.1f}s", "MODEL")

    # ── 纯文本路产物加载：只有 B(int8) + word_emb + head_text ──
    #    砍掉了 A 音频塔(198MB)、ctc_head(24MB)、E_PAD(20MB)：
    #    音频塔在 ASR 侧的 mmproj 里已经有一个，意图这条路不需要第二份。
    def _init_text(self, path, verbose, t0):
        import torch.nn as nn
        from transformers import BertConfig, BertModel, BertTokenizer

        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        self.INTENTS = ck["intents"]
        self.TAU = float(ck.get("tau", 0.1))
        self.R = int(ck.get("R", 3))
        self.SHARED = ck.get("shared_chars", [])
        self.K = len(self.SHARED)

        # 与 int8 路同样的 0 层壳技巧：不先分配 340MB 的 fp32 encoder
        bcfg = BertConfig(**ck["bert_config"])
        n_layers_real = bcfg.num_hidden_layers
        bcfg.num_hidden_layers = 0
        self.bert = BertModel(bcfg, add_pooling_layer=False)
        emb = dict(ck["bert_emb"])
        emb["word_embeddings.weight"] = ck["word_emb"]
        self.bert.embeddings.load_state_dict(emb, strict=False)
        self.bert.encoder = ck["Benc_module"]
        self.bert.config.num_hidden_layers = n_layers_real
        self.E_ALL = self.bert.embeddings.word_embeddings.weight.data

        self.tok = BertTokenizer(str(self.model_dir / "_vocab.txt"),
                                 do_lower_case=True)
        self.head_text = nn.Linear(768, len(self.INTENTS))
        self.head_text.weight.data = ck["head_text_w"]
        self.head_text.bias.data = ck["head_text_b"]

        # 音频路的东西一律不建（这就是省下来的 222MB）
        self.A = None
        self.ctc = None
        self.fe = None
        self.E_PAD = None
        self.CLS_ROW = None

        for _m in (self.bert, self.head_text):
            _m.to(self.device).eval()
            for _p in _m.parameters():
                _p.requires_grad = False
        self.E_ALL = self.E_ALL.to(self.device)
        if verbose:
            _log(f"纯文本产物 {path.name}  {len(self.INTENTS)} 类  "
                 f"val {ck.get('val_acc_text', 0)*100:.1f}%  "
                 f"（无音频塔/无 CTC 头）  {time.time()-t0:.1f}s", "MODEL")

    @torch.no_grad()
    def predict_text(self, text):
        """纯文本路：文本 → dmeta tokenizer → E_ALL 行 → B → head_text → 意图

        ⭐ 这是砍掉音频塔之后的主路：意图只由【ASR 文本】决定，
           embedding 模型负责把文本映射到意图空间（用户原话：
           "embedding 是从输入文本进行 embedding 匹配意图的，现在只走这条道"）。
        """
        t0 = time.time()
        s = text or ""
        # ⚠️ 血债（2026-10-16 实测）：head_text 只在 _init_text()（纯文本路产物）里创建。
        #    若 model/intent_text.pt 缺失 → __init__ 回退完整产物 → head_text 不存在
        #    → 这里抛 AttributeError，而调用方把它当"回退失败"吞掉，只留一行 ERR，
        #    现象是「句向量后端没加载 + 文本路也崩」= 意图完全不可用。
        #    改成明确报错 + 怎么修（裸 AttributeError 看不出是文件缺失）。
        if not hasattr(self, "head_text"):
            raise RuntimeError(
                "纯文本路不可用：没有 head_text —— 说明加载时没走 _init_text()，"
                "通常是 model/intent_text.pt 缺失、回退到了完整产物。"
                "修法：把 intent_text.pt 放进模型目录，或改用 intent_backend=embed"
                "（句向量 kNN，不需要 head_text）")
        ids = self.tok(s, truncation=True, max_length=64)["input_ids"]
        c2 = self._b_cls(self.E_ALL[ids])
        p = torch.softmax(self.head_text(c2.unsqueeze(0)), -1)[0]
        top = torch.topk(p, min(5, len(self.INTENTS)))
        return {
            "intent": self.INTENTS[int(top.indices[0])],
            "conf": float(top.values[0]),
            "topk": [(self.INTENTS[int(i)], float(v))
                     for v, i in zip(top.values, top.indices)],
            "text": s,
            "frames": 0,
            "latency_ms": (time.time() - t0) * 1000,
        }

    # ────────────────────────────────────────────────
    def hotwords_from_config(self, cfg):
        """从意图配置里抽出【闭集英文专有名词】当热词。

        ⭐ 用户的想法：CTC 对英文单词弱（Blender → lede了r），
          但我们只需要覆盖【闭集里那几十个英文名字】，表小而精确。
          （上次拿通用 n-gram 当热词必然失败，这次条件完全不同。）
        """
        out = set()
        for group in (cfg.apps, cfg.urls, cfg.files):
            for e in group:
                for k in ("name", "aliases"):
                    v = e.get(k)
                    vals = v if isinstance(v, list) else [v]
                    for s in vals:
                        if not s:
                            continue
                        # 只取纯 ASCII 的词（含数字/连字符/点）
                        t = "".join(ch for ch in str(s).lower()
                                    if ch.isascii()).strip()
                        if len(t) >= 2 and any(c.isalpha() for c in t):
                            out.add(t)
        return sorted(out)

    def ctc_decode_hot(self, logits, hotwords=None, alpha=0.0, first_alpha=0.0):
        """带【热词引导】的 CTC 贪心解码

        两个偏置点（实测总结）：
          ① 【词中】：末尾 ASCII 串是某个热词的前缀时 → 给下一个字母加 α
             「blen」→ 强推「d」
          ② 【词首】：末尾是中文（或还没出字）时 → 给所有热词的首字母加 first_alpha
             ⭐ 这才是关键：模型丢的是【首字母】
                Blender → lender / Ollama → llama / Steam → team / Python → ython
                末尾 ASCII 串永远不是热词的前缀，所以①根本触发不了。
                必须在「中文→英文交界处」把首字母顶上去。

        ⚠️ 两个 α 都不能太大：热词引导是加先验，过大会把对的改成错的。
        """
        import re as _re
        idx = logits.argmax(-1).tolist()
        out, prev = [], -1
        for t in range(logits.shape[0]):
            lg = logits[t]
            if hotwords and (alpha > 0 or first_alpha > 0) and out:
                s = "".join(out)
                m = _re.search(r"[a-z0-9\-\.]*$", s)
                tail = m.group() if m else ""
                if tail:
                    if alpha > 0:
                        lg = lg.clone()
                        for w in hotwords:
                            if w.startswith(tail) and len(w) > len(tail):
                                ci = self.char2i.get(w[len(tail)])
                                if ci is not None:
                                    lg[ci] += alpha
                elif first_alpha > 0:
                    # 末尾是中文 → 准备起一个英文词，把热词首字母顶上去
                    lg = lg.clone()
                    for w in hotwords:
                        ci = self.char2i.get(w[0])
                        if ci is not None:
                            lg[ci] += first_alpha
            k = int(lg.argmax())
            if k != prev and k != 0:
                out.append(self.SHARED[k - 1])
            prev = k
        return "".join(out)

    def ctc_decode(self, logits):
        """CTC 贪心解码

        ⭐ 整词 token 要映射回词：<w:blender> → "blender"
           （输出表 = 6628 字符 + N 个英文整词 token，见 train_wordtoken.py）
        """
        idx = logits.argmax(-1).tolist()
        out, prev = [], -1
        for i in idx:
            if i != prev and i != 0:
                t = self.SHARED[i - 1]
                out.append(t[3:-1] if t.startswith("<w:") else t)
            prev = i
        return "".join(out)

    @torch.no_grad()
    def _b_cls(self, rows):
        o = self.bert(inputs_embeds=rows.unsqueeze(0),
                      attention_mask=torch.ones(1, rows.shape[0], device=self.device),
                      return_dict=True)
        return o.last_hidden_state[0, 0]

    @torch.no_grad()
    def predict(self, wav, text_override=None):
        """返回意图/置信度/文本；中间特征存在 self.last_feat（1536 维）
        wav: float32 mono 16k numpy → dict

        ⚠️ 纯文本路产物（text_only）里没有音频塔：这时只接受 text_override，
           纯音频调用会直接报错 —— 不要静默给个错结果。
        """
        if getattr(self, "A", None) is None:
            if text_override is None:
                raise RuntimeError(
                    "text_only 产物不含音频塔：请用 predict_text(text)，"
                    "或改用完整 int8 产物")
            return self.predict_text(text_override)
        t0 = time.time()
        f = self.fe(wav, sampling_rate=16000, return_tensors="pt", padding=False,
                    return_attention_mask=True)
        inp = f["input_features"].to(self.device)
        msk = f["attention_mask"].to(self.device).long()
        # 补到 chunk(100 mel 帧) 整数倍 —— encoder 要求
        nmel = inp.shape[-1]
        pad = (-nmel) % 100
        if pad:
            inp = F.pad(inp, (0, pad)); msk = F.pad(msk, (0, pad))
        h = self.A(inp, msk).last_hidden_state
        T = h.shape[0]
        logits = self.ctc(h)
        # V1
        z = torch.softmax(logits / self.TAU, -1) @ self.E_PAD
        Kk = max(1, int(np.ceil(T / self.R)))
        zc = F.adaptive_avg_pool1d(z.T.unsqueeze(0), Kk).squeeze(0).T
        c1 = self._b_cls(torch.cat([self.CLS_ROW.view(1, 768), zc], 0))
        # V3
        #   文本来源可外挂：ASR 自己的解码器出的文本直接进来，
        #   就不再依赖自训 ctc_head 的贪心解码（CTC 域内只有 80.84%）
        s = text_override if text_override is not None else self.ctc_decode(logits)
        ids = self.tok(s, truncation=True, max_length=64)["input_ids"]
        c2 = self._b_cls(self.E_ALL[ids])
        feat = torch.cat([c1, c2])
        self.last_feat = feat.detach()
        p = torch.softmax(self.head(feat.unsqueeze(0)), -1)[0]
        top = torch.topk(p, min(5, len(self.INTENTS)))
        return {
            "intent": self.INTENTS[int(top.indices[0])],
            "conf": float(top.values[0]),
            "topk": [(self.INTENTS[int(i)], float(v)) for v, i in zip(top.values, top.indices)],
            "text": s,
            "frames": int(T),
            "latency_ms": (time.time() - t0) * 1000,
        }

    def features(self, wav, text_override=None):
        """只取 V5 的 1536 维特征（用于重训头 / 缓存）"""
        self.predict(wav, text_override=text_override)
        return self.last_feat


if __name__ == "__main__":
    import sys
    m = IntentModel(device="cpu")
    if len(sys.argv) > 1:
        wav = read_wav(sys.argv[1])
        r = m.predict(wav)
        print(json.dumps({k: v for k, v in r.items() if k != "topk"}, ensure_ascii=False, indent=1))
        print("top5:", [(a, round(b, 4)) for a, b in r["topk"]])
