# -*- coding: utf-8 -*-
"""逐层拼音标注探针（零训练，真机音频）——回答「浅层读出的拼音与最后一层差多少」。

原理（无训练、无新参数）：
  1. 真机 wav → Qwen3-ASR 贪心解码（do_sample=False = 生产路径）；
     加载走 asr_own.OwnASR（checkpoint 嵌套在 thinker_config 下，必须显式映射，
     否则 transformers 会静默加载随机权重 —— 见 asr_own.py 注释）；
  2. generate(return_dict_in_generate=True, output_hidden_states=True) 抓
     每个输出步、每个 decoder 层（28 层 + embedding 输出 = 29 个状态）的隐状态，
     KV 快速前向，与全量前向逐位一致；
  3. 「logit lens」读出 = argmax(lm_head(final_norm(h)))：tie_word_embeddings=True，
     lm_head 就是那张 151936×1024 词表矩阵（asr_vocab_trim 实证），每层都能用
     【同一张表】把该层隐状态最近邻地读成一个 token —— 零训练。
     29 层的 (·,1024) 状态堆成一批，一次 matmul 全读出；
  4. 该 token 直接 pypinyin 标注，逐层与「最终输出文本的拼音标注」对拍 ——
     最后一层(=模型实际采样那层)按构造 ≈100% 一致，作锚点。

指标（按层聚合，分母 = max(该层串长, 参考串长) 之和）：
  token一致%  = 该层 argmax token id == 最终采样 token id 的步数占比
  汉字一致%   = 该层读出串 与 最终文本的字符序列 difflib 匹配数 / 参考长度
  拼音一致%   = 声母韵母(降调)序列 difflib 匹配数 / 参考音节数 ——「西」vs「希」算对

用法：
  E:\\voice_agent_build\\Scripts\\python.exe -X utf8 _probe_pinyin_layers.py
"""
import json
import sys
import time
import wave
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
sys.path.insert(0, str(HERE))

import numpy as np
import torch

from asr_own import OwnASR

A_DIR = r"E:\models\qwen3-asr-0.6b-official"
WAVS = sorted((HERE / "_hot_wavs").glob("h*.wav"), key=lambda p: int(p.stem[1:]))
MAX_NEW = 48

# 真机录音标签（来自 diag_asr.py 的录音清单；仅供打印，不参与指标）
LABELS = {0: "打开Blender", 4: "启动Ollama", 6: "启动PowerShell", 8: "启动Python",
          9: "打开Steam", 11: "打开Trae", 12: "启动Clash", 13: "打开Node",
          16: "打开GitHub", 17: "打开HuggingFace", 18: "打开ModelScope",
          19: "上CSDN"}


def read_wav(path: Path):
    """stdlib wave 读 PCM(≤32bit) → float32 单声道。"""
    with wave.open(str(path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sw == 1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"{path.name}: 采样宽 {sw} 不支持")
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def to_syll(s):
    """CJK 字 → [(声母,韵母)]；非汉字丢弃（与 phon_match.syllables 同口径）。"""
    from phon_match import syllables
    out = []
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff":
            out.extend(tuple(x) for x in syllables(ch))
    return out


def to_cjk(s):
    return "".join(c for c in s if "\u4e00" <= c <= "\u9fff")


def _smatch(a, b):
    """difflib 匹配的元素总数（两序列间最长公共块的合并大小）。"""
    return sum(bl.size for bl in SequenceMatcher(None, a, b).get_matching_blocks())


def main():
    asr = OwnASR(model_dir=A_DIR, device="cpu", dtype="float32",
                 language="Chinese", max_new_tokens=MAX_NEW)
    asr.load()
    model = asr.model
    proc = asr.proc
    # 定位 decoder 主体与 lm_head
    dec, dname = None, None
    for name, mod in model.named_modules():
        if (hasattr(mod, "layers") and hasattr(mod, "norm")
                and hasattr(mod, "embed_tokens")):
            dec, dname = mod, name
            break
    assert dec is not None, "定位 decoder 失败"
    v, hid = dec.embed_tokens.weight.shape
    lmh, lname = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.bias is None \
                and mod.weight.shape[0] == v and mod.weight.shape[1] == hid:
            lmh, lname = mod, name
            break
    assert lmh is not None, "定位 lm_head 失败"
    nl = len(dec.layers)
    print(f"  decoder={dname!r}  layers={nl}  hidden={hid}", flush=True)
    print(f"  lm_head={lname!r}  vocab={v}  "
          f"tied={lmh.weight.data_ptr() == dec.embed_tokens.weight.data_ptr()}",
          flush=True)
    norm = dec.norm
    tokz = proc.tokenizer
    special_ids = {getattr(tokz, a, None) for a in
                   ("eos_token_id", "bos_token_id", "pad_token_id")
                   if getattr(tokz, a, None) is not None}

    results = []
    with torch.no_grad():
        for p in WAVS:
            idx = int(p.stem[1:])
            wav, sr = read_wav(p)
            t0 = time.perf_counter()
            # 复刻 OwnASR.transcribe 的输入构造（pad 整秒 + chat 模板 + 语言提示），
            # 但带 output_hidden_states 抓全部层
            pad = (-len(wav)) % 16000
            if pad:
                wav = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])
            msgs = [{"role": "user",
                     "content": [{"type": "audio", "audio": wav}]}]
            inputs = proc.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
            inputs = {k: (vv.to("cpu") if hasattr(vv, "to") else vv)
                      for k, vv in inputs.items()}
            n_prompt = int(inputs["input_ids"].shape[1])
            pre = tokz.encode("language Chinese<asr_text>",
                              add_special_tokens=False)
            inputs["input_ids"] = torch.cat(
                [inputs["input_ids"],
                 torch.tensor([pre], dtype=inputs["input_ids"].dtype)], dim=1)
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"],
                 torch.ones((1, len(pre)), dtype=inputs["attention_mask"].dtype)],
                dim=1)

            gen = model.generate(**inputs, max_new_tokens=MAX_NEW,
                                 do_sample=False,
                                 return_dict_in_generate=True,
                                 output_hidden_states=True)
            gen_ids = gen.sequences[0].tolist()
            ref_text = str(proc.decode(torch.tensor(gen_ids[n_prompt:]),
                                       return_format="transcription_only")).strip()
            if isinstance(ref_text, list):
                ref_text = ref_text[0] if ref_text else ""
            ms = (time.perf_counter() - t0) * 1000
            if not to_cjk(ref_text):
                print(f"  [!] h{idx} 空转写，跳过", flush=True)
                continue
            steps = getattr(gen, "hidden_states", None) or getattr(
                gen, "decoder_hidden_states", None)
            assert steps is not None, f"h{idx} 拿不到隐状态"
            n_steps = len(steps)
            ref_ids = gen_ids[len(gen_ids) - n_steps:]   # 每步采样出的 id
            ref_han, ref_py = to_cjk(ref_text), to_syll(ref_text)
            per_layer = {l: {"han": [], "py": [], "tok_agree": 0}
                         for l in range(nl + 1)}
            for t in range(n_steps):
                # 29 个状态堆一批，一次 matmul 全读出
                hs = []
                for l in range(nl + 1):
                    h = steps[t][l]
                    if h.dim() >= 3:                     # (B, T, H) → 最后位置
                        h = h[:, -1, :]
                    else:                                # (1, H) 直接用
                        h = h.reshape(1, -1)
                    hs.append(h)
                H = torch.cat(hs, dim=0)              # (nl+1, hid)
                Hn = norm(H)                          # lens 的 final_norm
                pks = (Hn @ lmh.weight.T).argmax(dim=-1).tolist()  # (nl+1,)
                for l, pk in enumerate(pks):
                    if l >= 1 and t < len(ref_ids) and pk == ref_ids[t]:
                        per_layer[l]["tok_agree"] += 1
                    if pk in special_ids:
                        continue
                    s = tokz.decode([pk]).strip()
                    if not s or (s.startswith("<") and s.endswith(">")):
                        continue                      # 特殊/标签形 token 不计
                    per_layer[l]["han"].append(s)
                    per_layer[l]["py"].extend(to_syll(s))
            rec = {"idx": idx, "label": LABELS.get(idx, ""), "ref": ref_text,
                   "n_han": len(ref_han), "n_py": len(ref_py),
                   "n_steps": n_steps, "ms": round(ms), "layers": {},
                   "prev": {}}
            for l in range(nl + 1):
                pl = per_layer[l]
                han_chars = [c for c in "".join(pl["han"])
                             if "\u4e00" <= c <= "\u9fff"]  # 双侧都只留汉字
                rec["layers"][l] = {
                    "tok_agree": pl["tok_agree"],
                    "m_han": _smatch(han_chars, list(ref_han)),
                    "m_py": _smatch(pl["py"], [tuple(x) for x in ref_py]),
                    "n_han_l": len(han_chars), "n_py_l": len(pl["py"]),
                    "s": "".join(pl["han"])[:60]}
            results.append(rec)
            print(f"  h{idx:<2d} {LABELS.get(idx, ''):<14s} → {ref_text!r} "
                  f"({ms:.0f} ms, {n_steps} 步)", flush=True)

    # ── 汇总：按层聚合 ─────────────────────────────────────────────
    print("\n" + "=" * 76, flush=True)
    print(f"  逐层「拼音标注」一致率 vs 最后一层(=最终输出) · 真机 "
          f"{len(results)} 条 · 零训练", flush=True)
    print("=" * 76, flush=True)
    print(f"  {'层':>3s} {'token一致%':>10s} {'汉字一致%':>10s} {'拼音一致%':>10s}",
          flush=True)
    tot_steps = sum(r["n_steps"] for r in results)
    agg = {}
    for l in range(nl + 1):
        mh = sum(r["layers"][l]["m_han"] for r in results)
        mp = sum(r["layers"][l]["m_py"] for r in results)
        sh = sum(max(r["layers"][l]["n_han_l"], r["n_han"]) for r in results)
        sp = sum(max(r["layers"][l]["n_py_l"], r["n_py"]) for r in results)
        ta = (100.0 * sum(r["layers"][l]["tok_agree"] for r in results)
              / max(tot_steps, 1)) if l >= 1 else None
        agg[l] = {"han": 100.0 * mh / max(sh, 1),
                  "py": 100.0 * mp / max(sp, 1), "tok": ta}
        print(f"  {l:>3d} {('%.1f' % ta) if ta is not None else '--':>10s} "
              f"{agg[l]['han']:>10.1f} {agg[l]['py']:>10.1f}", flush=True)
    # 第 0 层（embedding 输出）读出≈上一个 token 的平凡回声，仅作参考不算结论
    # 打印一条样本的逐层读出串，肉眼看形态随深度的渐变
    if results:
        r0 = results[0]
        show = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 26,
                27, 28]
        print(f"\n  样本 h{r0['idx']} 参考 {r0['ref']!r}：", flush=True)
        for l in show:
            if l in agg and l in r0["layers"]:
                s = r0["layers"][l].get("s", "")
                print(f"    L{l:<2d} → {s if s else '（全部被跳过/无汉字）'}",
                      flush=True)
    json.dump({"meta": {"wav_dir": str(HERE / "_hot_wavs"), "n": len(results),
                        "device": "cpu", "total_steps": tot_steps},
               "agg": {str(k): vv for k, vv in agg.items()}, "detail": results},
              open(HERE / "_probe_pinyin_layers.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n-> {HERE / '_probe_pinyin_layers.json'}", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n总耗时 {time.time() - t0:.0f}s", flush=True)
