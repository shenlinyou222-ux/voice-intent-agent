# -*- coding: utf-8 -*-
"""验证：加新命令到底要不要重训？

假设：加一个【新程序 / 新网址 / 新文件】不需要重训。
     因为 launch_app / open_url / open_file 已经是训练好的类，
     新条目只是往【闭集】里加一行，靠 resolve() 的 snap 就能命中。

做法：
  ① 临时往 intent_config 里加一个【模型从没见过】的条目（微信）
  ② 用模型已有的 wav（「启动记事本」这类）跑一遍，看意图是否仍为 launch_app
  ③ 用 TTS 现造一句「启动微信」，跑模型看意图 + 参数能不能解析出来
  ④ 对照：如果不行，才需要重训
"""
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from actions import Config, resolve, resolve_closed

# ══════════════════════════════════════════════════════════
# ① 造一份「加了新命令」的临时配置
# ══════════════════════════════════════════════════════════
tmp = HERE / "_tmp_cfg.json"
cfg = json.loads((HERE / "intent_config.json").read_text(encoding="utf-8"))
NEW_APP = {"id": "wechat", "name": "微信", "aliases": ["微信", "wechat"],
           "path": r"C:\Program Files (x86)\Tencent\WeChat\WeChat.exe"}
NEW_URL = {"id": "taobao2", "name": "天猫", "aliases": ["天猫", "tmall"],
           "url": "https://www.tmall.com"}
cfg["apps"] = cfg["apps"] + [NEW_APP]
cfg["urls"] = cfg["urls"] + [NEW_URL]
tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
C = Config(tmp)
print("=" * 78)
print("  ① 临时配置：加了【模型从没见过】的「微信」和「天猫」")
print("=" * 78)
print(f"  apps {len(C.apps)}（原 24 + 1）   urls {len(C.urls)}（原 16 + 1）")
print(f"  「微信」在闭集里: {'微信'.lower() in C.app_by_alias}")
print(f"  「天猫」在闭集里: {'天猫'.lower() in C.url_by_alias}")

# ══════════════════════════════════════════════════════════
# ② 纯规则：新命令能不能解析（不需要模型）
# ══════════════════════════════════════════════════════════
print()
print("=" * 78)
print("  ② 纯规则解析（不需要模型）")
print("=" * 78)
cases = ["启动微信", "打开微信", "帮我打开微信", "运行微信", "开一下微信吧",
         "打开天猫", "打开天猫网页", "上天猫", "访问天猫"]
ok = 0
for t in cases:
    it, arg = __import__("actions").parse(t, C)
    hit = resolve_closed(it, resolve(it, t, C), C)
    good = hit is not None
    ok += int(good)
    print(f"  {'✅' if good else '❌'} {t:16s} → {it:12s} arg={resolve(it, t, C)!r:10s} "
          f"{'→ ' + (hit.get('path') or hit.get('url')) if hit else '解析不出来'}")
print(f"\n  规则层 {ok}/{len(cases)}")

# ══════════════════════════════════════════════════════════
# ③ TTS 造一句「启动微信」，跑真模型
# ══════════════════════════════════════════════════════════
print()
print("=" * 78)
print("  ③ 真模型测试：TTS 现造「启动微信」，看意图 + 参数")
print("=" * 78)
TESTS = [("启动微信", "launch_app"), ("帮我打开微信", "launch_app"),
         ("打开天猫", "open_url"), ("上天猫", "open_url")]
outdir = HERE / "_tmp_wavs"
outdir.mkdir(exist_ok=True)
tsv = outdir / "jobs.tsv"
tsv.write_text("\n".join(f"t{i}\t{t}" for i, (t, _) in enumerate(TESTS)), encoding="utf-8")

tts = Path(r"C:\Users\user\Desktop\ds harness\voice_intent_router\asr_internal\tts_batch.ps1")
if tts.exists():
    subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", str(tts), "-Jobs", str(tsv), "-OutDir", str(outdir)],
                   capture_output=True, text=True)
    made = sorted(outdir.glob("*.wav"))
    print(f"  TTS 生成 {len(made)} 个 wav")
else:
    made = []
    print("  找不到 tts_batch.ps1")

if made:
    from model_runner import IntentModel, read_wav
    m = IntentModel(model_dir=HERE / "model", device="cpu")
    print()
    good = 0
    for i, (t, want) in enumerate(TESTS):
        w = outdir / f"t{i}.wav"
        if not w.exists():
            print(f"  [!] 缺 {w.name}")
            continue
        r = m.predict(read_wav(str(w)))
        ri, _ = __import__("actions").parse(r["text"], C)
        fi = r["intent"]
        arg = resolve(fi, r["text"], C)
        # ⭐ 规则优先：规则命中闭集就用规则
        if ri in ("launch_app", "open_url", "open_file"):
            ra = resolve(ri, r["text"], C)
            if resolve_closed(ri, ra, C):
                fi, arg = ri, ra
        hit = resolve_closed(fi, arg, C)
        okk = (fi == want) and (hit is not None)
        good += int(okk)
        print(f"  {'✅' if okk else '❌'} 说「{t}」")
        print(f"      转写 {r['text']!r}")
        print(f"      意图 {r['intent']} (conf {r['conf']:.3f})  期望 {want}")
        print(f"      参数 {arg!r} → {('✅ ' + str(hit.get('path') or hit.get('url'))) if hit else '❌ 闭集里没有'}")
    print(f"\n  模型层 {good}/{len(TESTS)}  —— 模型完全没重训")

print()
print("=" * 78)
print("  结论")
print("=" * 78)
print("  如果上面全过 → 加新程序/网址【只需改 intent_config.json】，不用重训。")
print("  只有加【全新意图类型】（比如「把音量调到50%」）才需要重训。")

tmp.unlink(missing_ok=True)
