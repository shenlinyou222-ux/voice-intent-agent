import json
from pathlib import Path
root = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")

# ① 放宽 VAD 起始/最短判据（治"段没生成/被丢弃"）
p = root / "intent_config.json"
d = json.loads(p.read_text(encoding="utf-8"))
b = d["behavior"]
old = {k: b.get(k) for k in ("vad_required_hits", "vad_min_speech_ms", "vad_min_run_ms")}
b["vad_required_hits"] = 2      # 连续 2 帧（64ms）就算语音起点（原 3=96ms）
b["vad_min_speech_ms"] = 200    # 最短语音 200ms（原 400ms：短句被丢 ✗）
b["vad_min_run_ms"] = 150       # 最短连续运行 150ms（原 240ms）
b["_comment_vad_relax"] = ("切断/VAD 排查：放宽起始判据（hits 3->2、min_speech 400->200ms、"
    "min_run 240->150ms），并在 handle() 里为每个段打印 时长/VAD峰概率/RMS，"
    "让'没生成段'与'段被丢'都能被观测。")
p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("  VAD 判据:", old, "->",
      {k: b[k] for k in ("vad_required_hits", "vad_min_speech_ms", "vad_min_run_ms")})

# ② 每个段进来就打一行日志（可见，不用 sub）
v = root / "voice_agent.py"
t = v.read_text(encoding="utf-8")
key = "段进入 handle"
if key not in t:
    anchor = "    def handle(self, wav):"
    assert anchor in t, "handle 锚点未匹配"
    blk = anchor + '''
        # ⭐ 段级可观测：每个进入处理的段都留痕（时长/VAD峰概率/RMS）
        #   目的：把"没生成段"（VAD 太严 ✗）和"段被丢"（门限 ✗）与"转写错"分开。
        try:
            import numpy as _np
            _dur = len(wav) / 16000.0
            _rms = float(_np.sqrt((_np.asarray(wav, dtype=_np.float32) ** 2).mean()))
            _pk = float(getattr(self.seg, "peak_prob", 0.0) or 0.0)
            _lp = float(getattr(self.seg, "last_prob", 0.0) or 0.0)
            log(f"   📥 段进入 handle: {_dur:.2f}s  VAD峰{max(_pk, _lp):.3f}  RMS {_rms:.5f}",
                "MIC")
        except Exception:
            pass
'''
    t = t.replace(anchor, blk, 1)
    v.write_text(t, encoding="utf-8")
    print("  ✅ handle() 已加段级日志")
else:
    print("  已存在，跳过")
