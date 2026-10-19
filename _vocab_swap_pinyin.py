# -*- coding: utf-8 -*-
r"""词表置换：ASR 输出词表 151936(汉字) → 拼音音节表(~410 类)，纯映射构造、零训练。

用户定案（2026-10-18）：「保持 v2 原方案；把词表换了，对应汉字都换成拼音标注，
省词表开销。这是一个两个矩阵的对齐、一个映射；理论上精度百分百。」

机制 v3（映射，不训练）——分组=读音组，代表行=组内词频 top1 的原矩阵行：
  exact    旧 logits 的组边缘化（= 映射理论上限；诊断用，部署用不了大表）
  rep1     每组 top1 代表行组成 K+1 行的头（可 tie：E_in 与 W 同行）
  rep3     每组 top3 代表行组成 ≤3K+1 行的头，argmax 后按行→组归并
  centroid 组内频权均值行；nsum 单位化行频权和（v2 已弱，仅对照）
  输入侧 E_in[k] = 组 top1 代表行（回喂原矩阵行，主干动态与原模型同基）；
  解码防回声：top==上一音节时 δ(默认 1.0) 抑制该组行再 argmax —— 对应
  原模型中同音字形散布的隐性行为，映射参数不动，只是解码规则。
  提示词/特殊行原 id 保留；eos 行=旧 eos 行并入输出头。

了对拍：
  DEV   teacher-forced 逐单字 token 位置：各 arm 读音集命中率 / 主读音率；
        基线 = 旧头 argmax 字主读音（旧路自身水平）。
  真机  置换模型自回归直出音节流 vs 原模型汉字流 pypinyin 标注（difflib 保持
        率）+ 路由 resolve_by_rank A(原字文本) vs B(音节流直喂 ut) 交叉。

用法：
  E:\voice_agent_build\Scripts\python.exe -X utf8 _vocab_swap_pinyin.py [--dev 300] [--utt 60] [--real-only] [--delta 1.0]
"""
import argparse
import io
import json
import re
import sys
import time
import wave
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
sys.path.insert(0, str(HERE))

import numpy as np
import torch

from asr_own import OwnASR

A_DIR = r"E:\models\qwen3-asr-0.6b-official"

P = print
_FIN_RE = re.compile(r"^[a-zü]+$")


def read_wav(path):
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
        raise ValueError(f"{path}: 采样宽 {sw} 不支持")
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def all_token_strings(tokz):
    """id→字符串（convert_ids_to_tokens + byte 还原）。"""
    try:
        n = len(tokz)
    except Exception:
        n = getattr(tokz, "vocab_size", 0) or 0
    if not n:
        return None
    try:
        raw = tokz.convert_ids_to_tokens(list(range(n)))
    except Exception:
        return None
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    k = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + k)
            k += 1
    c2b = {chr(c): b for b, c in zip(bs, cs)}
    out = []
    for t in raw:
        try:
            out.append(bytes(c2b[ch] for ch in t).decode("utf-8", "ignore"))
        except Exception:
            out.append(t or "")
    return out


def load_tts():
    jobs = []
    for line in io.open(HERE / "intent_tts.tsv", encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            jobs.append((parts[0], parts[1]))
    freq = defaultdict(int)
    for _, t in jobs:
        for ch in t:
            if "\u4e00" <= ch <= "\u9fff":
                freq[ch] += 1
    return jobs, dict(freq)


def load_utt(limit=60):
    if limit <= 0:
        return []
    rows = [json.loads(l) for l in
            io.open(HERE / "utt" / "index.jsonl", encoding="utf-8") if l.strip()]
    ok = []
    for r in rows:
        p = HERE / "utt" / r.get("wav", "")
        if (r.get("src") == "mic" and float(r.get("dur") or 0) > 0.6
                and float(r.get("rms") or 0) > 0.004
                and (r.get("text") or "").strip() and p.exists()):
            ok.append((str(r["ts"]), p, r.get("text") or ""))
    ok.sort()
    if len(ok) > limit:
        step = len(ok) / limit
        ok = [ok[int(i * step)] for i in range(limit)]
    return ok


def init_py():
    """readings(ch)->(primary, 读法集)；与 phon_match.syllables 同口径；缓存。"""
    from phon_match import _ti, _tf, _strip_tone
    from pypinyin import pinyin as ppy
    cache = {}

    def readings(ch):
        v = cache.get(ch)
        if v is not None:
            return v
        got = set()
        primary = None
        rds = ppy(ch, heteronym=True)[0]
        if rds:
            primary = (_ti(rds[0], strict=False),
                       _strip_tone(_tf(rds[0], strict=False)))
        for rd in rds:
            ini = _ti(rd, strict=False)
            fin = _strip_tone(_tf(rd, strict=False))
            if (fin and _FIN_RE.match(fin)
                    and (ini == "" or _FIN_RE.match(ini))):
                got.add((ini, fin))
        cache[ch] = (primary, got)
        return primary, got
    return readings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, default=300)
    ap.add_argument("--utt", type=int, default=60)
    ap.add_argument("--real-only", action="store_true")
    ap.add_argument("--delta", type=float, default=1.0, help="即时回声抑制量")
    ap.add_argument("--force-arm", default=None,
                    help="强制运行时臂（如 full对照/wl白名单/charsub/rep3）")
    args = ap.parse_args()
    readings = init_py()

    asr = OwnASR(model_dir=A_DIR, device="cpu", dtype="float32",
                 language="Chinese", max_new_tokens=64)
    asr.load()
    model, proc = asr.model, asr.proc
    dec = model.model.language_model
    nl = len(dec.layers)
    norm = dec.norm
    E_old = dec.embed_tokens.weight.detach()
    tokz = proc.tokenizer
    hid = E_old.shape[1]
    jobs, freq = load_tts()
    P(f"TTS 语料 {len(jobs)} 条，字种 {len(freq)}")
    report = {"meta": {"built": time.strftime("%F %T"), "device": "cpu",
                       "delta": args.delta}}

    # ══ Part 1 ｜映射构造 ══
    P("\n== Part 1 映射构造 ==")
    tstrs = all_token_strings(tokz)
    char_ids = defaultdict(list)
    if tstrs:
        for i, s in enumerate(tstrs):
            if len(s) == 1 and "\u4e00" <= s <= "\u9fff":
                char_ids[s].append(i)
    else:
        for i in range(min(E_old.shape[0], 20000)):
            s = tokz.decode([i]).strip()
            if len(s) == 1 and "\u4e00" <= s <= "\u9fff":
                char_ids[s].append(i)
    n_tok = sum(len(v) for v in char_ids.values())
    P(f"  单字 token {n_tok} 个（{len(char_ids)} 个汉字）")

    # 音节表=语料字符的读法（干净口径；vocab 罕用字的怪读只并入已有组）
    sylls = set()
    junk_fin = 0
    for ch in list(char_ids) + list(freq):
        _, got = readings(ch)
        for s in got:
            if any(c.isdigit() for c in s[0] + s[1]):
                junk_fin += 1
                continue
            sylls.add(s)
    SYLLS = sorted(sylls)
    syl2i = {s: i for i, s in enumerate(SYLLS)}
    K = len(SYLLS)
    P(f"  音节类 {K}（剔垃圾读音 {junk_fin} 条）")

    members = defaultdict(list)
    for ch, ids in char_ids.items():
        for s in readings(ch)[1]:
            k = syl2i.get(s)
            if k is not None:
                members[k].append((ch, ids))
    rows1, rows3, lab3 = [], [], []
    row_cent = torch.zeros(K, hid)
    row_nsum = torch.zeros(K, hid)
    n_empty = 0
    for k in range(K):
        mem = members.get(k) or []
        if not mem:
            n_empty += 1
            continue
        order = sorted(mem, key=lambda ci: -(freq.get(ci[0], 0) or 1))
        rows = torch.stack([E_old[ids[0]] for _, ids in order])
        ws = np.array([max(freq.get(ch, 0), 1) for ch, _ in order],
                      dtype=np.float32)
        rows1.append(rows[0])
        rows3.extend(rows[:3])
        lab3.extend([k] * min(3, len(rows)))
        row_cent[k] = rows.t().mv(torch.from_numpy(ws)) / ws.sum()
        un = rows / rows.norm(dim=1, keepdim=True).clamp(min=1e-9)
        row_nsum[k] = un.t().mv(torch.from_numpy(ws))
    assert n_empty == 0, f"空组 {n_empty}"
    rep1 = torch.stack(rows1)                              # (K,hid)
    eos_id = tokz.eos_token_id
    eos_row = E_old[eos_id].unsqueeze(0)
    # 生成配置的 eos 是 [151643,151645] 两个 —— 少一个模型就到不了"停"
    im_row = E_old[151643].unsqueeze(0) if E_old.shape[0] > 151643 else None
    # ★ charsub 行：全部单字 token 行的子集（真·映射——读数=每字主读音，
    #   回喂=自己 argmax 的原汉字行，主干动态与原模型逐位同）
    CHARS = sorted(char_ids)
    W_chars = torch.stack([E_old[char_ids[c][0]] for c in CHARS])
    W_char = torch.cat([W_chars, eos_row]
                       + ([im_row] if im_row is not None else []), dim=0)
    cN = W_chars.shape[0]
    W_rep1 = torch.cat([rep1, eos_row]
                       + ([im_row] if im_row is not None else []), dim=0)
    W_rep3 = torch.cat([torch.stack(rows3), eos_row]
                       + ([im_row] if im_row is not None else []), dim=0)
    lab3_t = torch.tensor(lab3 + [K] * (2 if im_row is not None else 1),
                          dtype=torch.long)     # 行→组(末两行=eos)
    ROW2G_TAIL = torch.tensor(list(range(K)) + [K]
                              * (2 if im_row is not None else 1),
                              dtype=torch.long)
    dummy = np.zeros(16000, dtype=np.float32)
    msgs = [{"role": "user", "content": [{"type": "audio", "audio": dummy}]}]
    inp0 = proc.apply_chat_template(msgs, tokenize=True,
                                    add_generation_prompt=True,
                                    return_dict=True, return_tensors="pt")
    pre = tokz.encode("language Chinese<asr_text>", add_special_tokens=False)
    prompt_ids = set(inp0["input_ids"][0].tolist()) | set(pre)
    audio_tok = getattr(model.config, "audio_token_id", None)
    kept_ids = sorted(prompt_ids - ({audio_tok} if audio_tok is not None else set())
                      | ({eos_id} if eos_id is not None else set()))
    kept_rows = {i: E_old[i].clone() for i in kept_ids}
    # ★ v5 白名单行：单字全集 + 语料/真机日志实际用过的全部 token（多字词/
    #   拉丁/标点）。输出 = token 串的音节标注（1:N 展开，词 token 一步出多
    #   音节），回喂 = 原 token 行 → 主干条件分布与原模型逐位同。
    wl_texts = [t for _, t in jobs]
    try:
        for l in io.open(HERE / "utt" / "index.jsonl", encoding="utf-8"):
            if l.strip():
                try:
                    wl_texts.append(json.loads(l).get("text") or "")
                except Exception:
                    pass
    except Exception:
        pass
    wl_ids = set(kept_ids) | {i for v in char_ids.values() for i in v}
    for t in wl_texts:
        try:
            wl_ids |= set(tokz.encode(t, add_special_tokens=False))
        except Exception:
            pass
    WL = sorted(wl_ids)
    from phon_match import text_stream
    WL_SYL = []                       # 行→音节标注（0/1/2+ 个）
    for i in WL:
        s = (tstrs[i] if tstrs else "") or tokz.decode([i]).strip()
        if len(s) == 1 and "\u4e00" <= s <= "\u9fff":
            p = readings(s)[0]
            WL_SYL.append([p] if p else [])
        else:
            WL_SYL.append([tuple(x) for x in text_stream(s)])
    W_wl = torch.cat([torch.stack([E_old[i] for i in WL]), eos_row]
                     + ([im_row] if im_row is not None else []), dim=0)
    WL_SYL.append([])                 # eos 行
    if im_row is not None:
        WL_SYL.append([])             # im_end 行
    cN_wl = len(WL)
    P(f"  输出头行数: rep1 {W_rep1.shape[0]}  rep3 {W_rep3.shape[0]}"
      f"  charsub {cN}  wl {cN_wl}")
    P(f"  保留行(提示词/特殊/eos) {len(kept_ids)} 个")
    report["inventory"] = {
        "syllables": K, "single_char_tokens": n_tok, "chars": len(char_ids),
        "kept_ids": len(kept_ids), "old_vocab": int(E_old.shape[0]),
        "old_bytes_bf16": int(E_old.numel() * 2),
        "rep1_bytes": int(W_rep1.numel() * 2),
        "rep3_bytes": int(W_rep3.numel() * 2),
        "charsub_bytes": int(W_char.numel() * 2), "charsub_rows": cN,
        "wl_bytes": int(W_wl.numel() * 2), "wl_rows": cN_wl,
    }
    P(f"  省账: 词表块 {report['inventory']['old_bytes_bf16']/2**20:.0f} MB → "
      f"rep1 {report['inventory']['rep1_bytes']/2**20:.1f} MB / "
      f"rep3 {report['inventory']['rep3_bytes']/2**20:.1f} MB / "
      f"wl {report['inventory']['wl_bytes']/2**20:.1f} MB")

    # ══ Part 2 ｜dev 诊断 ══
    arm = None
    if not args.real_only:
        P(f"\n== Part 2 dev 诊断（teacher-forced 抽 {args.dev} 条）==")
        ids_dev = sorted(jobs)[::max(1, len(jobs) // args.dev)][:args.dev]
        dev = [(HERE / "wavs" / f"{i}.wav", t) for i, t in ids_dev]
        Hs, Gt = [], []
        t0 = time.time()
        with torch.no_grad():
            for n, (p, gt) in enumerate(dev):
                if not p.exists():
                    continue
                try:
                    wav, _ = read_wav(p)
                except Exception:
                    continue
                pad = (-len(wav)) % 16000
                if pad:
                    wav = np.concatenate([wav, np.zeros(pad, np.float32)])
                msgs = [{"role": "user",
                         "content": [{"type": "audio", "audio": wav}]}]
                inp = proc.apply_chat_template(msgs, tokenize=True,
                                               add_generation_prompt=True,
                                               return_dict=True,
                                               return_tensors="pt")
                gids = tokz.encode(gt, add_special_tokens=False)
                inp["input_ids"] = torch.cat(
                    [inp["input_ids"],
                     torch.tensor([gids], dtype=inp["input_ids"].dtype)], dim=1)
                inp["attention_mask"] = torch.cat(
                    [inp["attention_mask"],
                     torch.ones((1, len(gids)),
                                dtype=inp["attention_mask"].dtype)], dim=1)
                out = model(**inp, return_dict=True, output_hidden_states=True)
                h_l = out.hidden_states[nl][0]
                P_pos = int(inp["input_ids"].shape[1]) - len(gids)
                for i, gid in enumerate(gids):
                    j = P_pos + i - 1
                    if j < 0:
                        continue
                    s = tstrs[gid] if tstrs else tokz.decode([gid]).strip()
                    if len(s) == 1 and "\u4e00" <= s <= "\u9fff":
                        Hs.append(h_l[j].clone())
                        Gt.append(s)
                if (n + 1) % 100 == 0:
                    P(f"    {n + 1}/{len(dev)} 条 · {time.time()-t0:.0f}s")

        H = torch.stack(Hs)
        Lf = norm(H)
        l_old = Lf @ E_old.T
        top_old = l_old.argmax(-1).tolist()
        top_exact = torch.zeros(len(Gt), K + 1)
        with torch.no_grad():
            for k in range(K):
                mem = [i for _, idsl in members[k] for i in idsl]
                if not mem:
                    continue
                mi = torch.tensor(mem, dtype=torch.long)
                top_exact[:, k] = torch.logsumexp(l_old[:, mi], dim=-1)
            top_exact[:, K] = l_old[:, eos_id]
        top_exact = top_exact.argmax(-1).tolist()

        def lab2syl(g):
            return SYLLS[g] if g < K else "<eos>"

        tops = {
            "exact边缘化": [(lab2syl(g if g < K else K),
                             g) for g in top_exact],
            "rep1": [(lab2syl(int(g)), int(g))
                     for g in (Lf @ W_rep1.T).argmax(-1).tolist()],
            "rep3": [(lab2syl(int(lab3_t[int(g)])), int(lab3_t[int(g)]))
                     for g in (Lf @ W_rep3.T).argmax(-1).tolist()],
            "centroid": [(lab2syl(int(g)), int(g))
                         for g in (Lf @ torch.cat(
                             [row_cent, eos_row], dim=0).T).argmax(-1).tolist()],
            "nsum": [(lab2syl(int(g)), int(g))
                     for g in (Lf @ torch.cat(
                         [row_nsum, eos_row], dim=0).T).argmax(-1).tolist()],
        }
        res = {}
        for nm, lst in tops.items():
            hit = prim = 0
            for (tS, g), ch in zip(lst, Gt):
                if g < K:
                    gt_read = set()
                    for c in ch:                     # 多字 GT:各字读音并集
                        gt_read |= readings(c)[1]
                    hit += 1 if tS in gt_read else 0
                    prim += 1 if tS == readings(ch[0])[0] else 0
            res[nm] = {"hit": round(100.0 * hit / len(Gt), 1),
                       "primary": round(100.0 * prim / len(Gt), 1)}
            P(f"    {nm:<9s} 读音集命中 {res[nm]['hit']:.1f}%   "
              f"主读音 {res[nm]['primary']:.1f}%")
        hit_o = 0
        for t, ch in zip(top_old, Gt):
            ts = tokz.decode([t]).strip()
            if len(ts) == 1 and "\u4e00" <= ts <= "\u9fff":
                hit_o += 1 if readings(ts)[0] == readings(ch)[0] else 0
        # charsub：旧头在【单字行子集】上的 argmax → 主读音
        Wch = Lf @ W_char.T
        top_ch = Wch.argmax(-1).tolist()
        ch_hit = ch_hitset = ch_char = 0
        for r, ch in zip(top_ch, Gt):
            if r >= cN:
                continue
            ch_char += 1
            p = readings(CHARS[r])[0]
            ch_hit += 1 if p == readings(ch)[0] else 0
            ch_hitset += 1 if p in readings(ch)[1] else 0
        old_n = sum(1 for t in top_old
                    if (lambda s: len(s) == 1 and "\u4e00" <= s <= "\u9fff")(
                        tokz.decode([t]).strip()))
        P(f"    {'旧argmax':<9s} 主读音一致率 {100.0*hit_o/max(old_n,1):.1f}%"
          f"（旧路自身水平, 单字位置 n={old_n}）")
        res["charsub"] = {
            "hit": round(100.0 * ch_hitset / max(ch_char, 1), 1),
            "primary": round(100.0 * ch_hit / max(ch_char, 1), 1),
            "n_char_picks": ch_char}
        P(f"    {'charsub':<9s} 读音集命中 {res['charsub']['hit']:.1f}%   "
          f"主读音 {res['charsub']['primary']:.1f}%"
          f"（选中单字 {ch_char}/{len(Gt)}）")
        # wl白名单:旧头在白名单 token 域的 argmax → token 串的音节标注
        top_wl = (Lf @ W_wl.T).argmax(-1).tolist()
        wl_hit = wl_prim = wl_n = 0
        for r, ch in zip(top_wl, Gt):
            gt_read = set()
            for c in ch:
                gt_read |= readings(c)[1]
            syls = WL_SYL[r] if r < cN_wl else []
            if not syls:
                continue
            wl_n += 1
            if all(s in gt_read for s in syls):
                wl_hit += 1
            if syls[0] == readings(ch[0])[0]:
                wl_prim += 1
        res["wl白名单"] = {
            "hit": round(100.0 * wl_hit / max(wl_n, 1), 1),
            "primary": round(100.0 * wl_prim / max(wl_n, 1), 1),
            "n_tok_picks": wl_n}
        P(f"    {'wl白名单':<9s} 读音集命中 {res['wl白名单']['hit']:.1f}%   "
          f"首音节 {res['wl白名单']['primary']:.1f}%"
          f"（token 决策 {wl_n}/{len(Gt)}）")
        report["dev"] = {"n_pos": len(Gt), "arms": res,
                         "oldbaseline_primary": (
                             round(100.0 * hit_o / max(old_n, 1), 1)
                             if old_n else 0.0)}
        cands = ["wl白名单", "charsub", "rep3", "rep1", "centroid", "nsum"]
        arm = max(cands, key=lambda nm: (res[nm]["hit"], res[nm]["primary"]))
        report["dev"]["arm"] = arm
        P(f"  ★ 部署头 = {arm}")

    # 选择输出头/回喂行
    arm = args.force_arm or arm
    im_add = [im_row] if im_row is not None else []
    if arm == "full对照":
        W_out = E_old                       # 完整词表:解码环正确性控制臂
        tied = True
    elif arm == "wl白名单":
        W_out = W_wl
        tied = True
    elif arm == "charsub":
        W_out = W_char
        ROWLAB = [readings(c)[0] for c in CHARS] + ["<eos>", "<eos>"]
        tied = True
    elif arm == "rep3":
        W_out = W_rep3
        row2g = lab3_t
        tied = False
    elif arm == "centroid":
        W_out = torch.cat([row_cent, eos_row] + im_add, dim=0)
        row2g = ROW2G_TAIL
        tied = False
    elif arm == "nsum":
        W_out = torch.cat([row_nsum, eos_row] + im_add, dim=0)
        row2g = ROW2G_TAIL
        tied = False
    else:                            # rep1（默认）
        W_out = W_rep1
        row2g = ROW2G_TAIL
        tied = True
    report["runtime"] = {"arm": arm or "rep1(缺省)", "tied": tied}
    P(f"  运行时头 {arm or 'rep1(缺省)'}  行数 {W_out.shape[0]}  tied={tied}")

    # ══ Part 3 ｜真机对拍 ══
    P("\n== Part 3 真机对拍 ==")
    hot = []
    if args.utt >= 0:
        hot = sorted((HERE / "_hot_wavs").glob("h*.wav"),
                     key=lambda p: int(p.stem[1:]))
    utt = load_utt(args.utt)
    LABELS = {0: "打开Blender", 4: "启动Ollama", 6: "启动PowerShell",
              8: "启动Python", 9: "打开Steam", 11: "打开Trae", 12: "启动Clash",
              13: "打开Node", 16: "打开GitHub", 17: "打开HuggingFace",
              18: "打开ModelScope", 19: "上CSDN"}
    from actions import Config, resolve_by_rank

    CFG = Config(Path(r"C:\Users\user\Desktop\ds harness\voice_agent"
                      r"\intent_config.json"))

    def has_affix(stream):
        from phon_match import text_stream
        targets = []
        for su in (list(CFG.triggers.get("url_suffix") or []) + ["网页版"]
                   + list(CFG.triggers.get("file_suffix") or [])):
            tg = [tuple(x) for x in text_stream(su)]
            if len(tg) >= 2:
                targets.append(tg)
        sn = [tuple(x) for x in stream]
        for tg in targets:
            sm = SequenceMatcher(None, sn, tg)
            if any(b.size >= 2 for b in sm.get_matching_blocks()):
                return True
        return False

    # 上一组各行的行号（回声抑制用）
    if arm == "rep3":
        rows_of_group = defaultdict(list)
        for i, g in enumerate(lab3):
            rows_of_group[g].append(i)
    else:
        rows_of_group = {k: [k] for k in range(K)}

    def swapped_stream(wav):
        """置换模型自回归：新头+代表行回喂+即时回声抑制。返回 (流, 耗时ms, 抑制次数)"""
        pad = (-len(wav)) % 16000
        if pad:
            wav = np.concatenate([wav, np.zeros(pad, np.float32)])
        msgs = [{"role": "user",
                 "content": [{"type": "audio", "audio": wav}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True,
                                       add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt")
        inp = {k: (v.to("cpu") if hasattr(v, "to") else v)
               for k, v in inp.items()}
        inp["input_ids"] = torch.cat(
            [inp["input_ids"],
             torch.tensor([pre], dtype=inp["input_ids"].dtype)], dim=1)
        inp["attention_mask"] = torch.cat(
            [inp["attention_mask"],
             torch.ones((1, len(pre)), dtype=inp["attention_mask"].dtype)],
            dim=1)
        t0 = time.perf_counter()
        sup = 0
        with torch.no_grad():
            out = model(**inp, use_cache=True, return_dict=True,
                        output_hidden_states=True)
            past = out.past_key_values
            h = out.hidden_states[nl][:, -1:, :]
            cur = int(inp["input_ids"].shape[1])
            stream = []
            am = inp["attention_mask"].clone()
            hist = []                                   # 已发组的组号
            max_steps = 96 if arm == "full对照" else 64
            for _ in range(max_steps):
                v = (norm(h) @ W_out.T).reshape(-1)
                pk = int(v.argmax())
                if arm == "full对照":
                    # 控制臂:流存原 token 解字符串,回喂原行,动态应=generate
                    if pk == eos_id or (im_row is not None and pk == 151643):
                        break
                    s = tokz.decode([pk]).strip()
                    if s:
                        stream.append(s)
                    e = E_old[pk].unsqueeze(0).unsqueeze(0)
                elif arm == "wl白名单":
                    if pk >= cN_wl:                     # eos 行
                        break
                    syls = WL_SYL[pk]
                    if not syls:
                        syls = []                       # 标点等空标注行
                    stream.extend(syls)
                    e = W_out[pk].unsqueeze(0).unsqueeze(0)
                elif arm == "charsub":
                    if pk >= cN:                        # eos 行
                        break
                    sy = ROWLAB[pk]
                    if sy is None:
                        continue
                    stream.append(sy)
                    e = W_out[pk].unsqueeze(0).unsqueeze(0)
                else:
                    g0 = int(row2g[pk])
                    if g0 in hist[-2:]:                 # 两步窗回声(含隔步)
                        v = v.clone()
                        rows_to_sup = []
                        for g in {g0, *hist[-2:]}:
                            rows_to_sup.extend(rows_of_group.get(g, [g]))
                        for r in rows_to_sup:
                            v[r] -= args.delta
                        sup += 1
                    pk = int(v.argmax())
                    g = int(row2g[pk])
                    if g == K:               # eos
                        break
                    stream.append(SYLLS[g])
                    hist.append(g)
                    e = rep1[g].unsqueeze(0).unsqueeze(0)
                am = torch.cat([am, torch.ones((1, 1), dtype=torch.long)],
                               dim=1)
                try:
                    o2 = dec(inputs_embeds=e, past_key_values=past,
                             attention_mask=am, use_cache=True,
                             cache_position=torch.tensor(
                                 [cur], dtype=torch.long))
                except Exception:
                    o2 = dec(inputs_embeds=e, past_key_values=past,
                             attention_mask=am, use_cache=True)
                past = o2.past_key_values
                h = o2.last_hidden_state[:, -1:, :]
                cur += 1
        if arm == "full对照":
            from phon_match import text_stream as _ts
            stream = [tuple(x) for x in _ts("".join(stream))]
        return stream, (time.perf_counter() - t0) * 1000, sup

    rows = []
    set_list = ([(pp, LABELS.get(int(pp.stem[1:]), "")) for pp in hot]
                + [(pp, "utt") for _, pp, _ in utt])
    from phon_match import text_stream
    for p, label in set_list:
        try:
            wav, _ = read_wav(p)
        except Exception as e:
            P(f"  [!] {p.name} 读失败 {e}")
            continue
        ref_txt = asr.transcribe(wav)
        ref_sy = [tuple(x) for x in text_stream(ref_txt)]
        stream, ms, sup = swapped_stream(wav)
        sm = SequenceMatcher(None, stream, ref_sy)
        keep = (sum(b.size for b in sm.get_matching_blocks())
                / max(len(ref_sy), len(stream), 1)) if ref_sy else None
        try:
            itA, arA, sA, _ = resolve_by_rank(ref_txt, CFG)
        except Exception:
            itA, arA, sA = None, None, 0.0
        try:
            itB, arB, sB, _ = resolve_by_rank(ref_txt, CFG, ut=stream,
                                              affix=has_affix(stream))
        except Exception:
            itB, arB, sB = None, None, 0.0
        same = (itA, arA) == (itB, arB)
        rows.append({"wav": p.name, "label": label, "ref": ref_txt,
                     "ref_n": len(ref_sy),
                     "stream": [f"{i}+{f}" for i, f in stream],
                     "stream_n": len(stream),
                     "keep": None if keep is None else round(keep, 3),
                     "ms": round(ms), "suppress": sup,
                     "A": [itA, arA, round(float(sA or 0.0), 3)],
                     "B": [itB, arB, round(float(sB or 0.0), 3)],
                     "same": same})
        show = " ".join(f"{i}{f[:2]}" for i, f in stream[:8])
        P(f"  {p.name:<24s} ref「{ref_txt}」 {len(ref_sy)}音节 | "
          f"swap {len(stream)}音节(抑{sup}) keep="
          f"{'--' if keep is None else f'{keep*100:.0f}%'} "
          f"{'路由A==B ✓' if same else f'路由差 A={itA}/{arA} B={itB}/{arB}'} "
          f"[{show}…]")
    have = [r for r in rows if r["keep"] is not None]
    kr = sum(r["keep"] for r in have) / max(len(have), 1)
    same_n = sum(1 for r in rows if r["same"])
    P(f"\n  真机 {len(rows)} 条 · 标注保持率 {kr*100:.1f}% · "
      f"路由 A==B {same_n}/{len(rows)}")
    report["real"] = {"n": len(rows), "keep_rate": round(kr, 3),
                      "router_same": same_n, "rows": rows}

    # ══ Part 4 ｜每步 head 开销 ══
    with torch.no_grad():
        h1 = norm(torch.randn(1, hid))
        t0 = time.perf_counter()
        for _ in range(30):
            (h1 @ E_old.T).argmax(-1)
        t_o = (time.perf_counter() - t0) / 30 * 1000
        t0 = time.perf_counter()
        for _ in range(30):
            (h1 @ W_out.T).argmax(-1)
        t_n = (time.perf_counter() - t0) / 30 * 1000
    report["timing"] = {"head_old_ms": round(t_o, 3),
                        "head_new_ms": round(t_n, 3)}
    P(f"  head/步: 旧 {t_o:.2f} ms → 新 {t_n:.3f} ms（约 10~30 步/条）")

    json.dump(report, open(HERE / "_vocab_swap_report.json", "w",
                           encoding="utf-8"), ensure_ascii=False, indent=1)
    P(f"\n-> {HERE / '_vocab_swap_report.json'}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    P(f"\n总耗时 {time.time() - t0:.0f}s")
