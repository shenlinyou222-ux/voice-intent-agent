# -*- coding: utf-8 -*-
"""一次性配置变更：①ASR 换 1.7B ②废弃 DSH（触发词清空+开关关闭）。"""
import json
from pathlib import Path

P = Path(__file__).parent / "intent_config.json"
GD17 = (r"C:\Users\user\Desktop\ds harness\aigc_demo_framework"
        r"\models\qwen3-asr-gguf-1.7b")

d = json.loads(P.read_text(encoding="utf-8"))
b, t = d["behavior"], d["triggers"]

# ① ASR 换 1.7B（用户指定）
b["asr_gguf"] = GD17 + r"\Qwen3-ASR-1.7B-Q4_K_M.gguf"
b["asr_mmproj"] = GD17 + r"\mmproj-Qwen3-ASR-1.7B-Q4_K_M.gguf"
b["_comment_asr_model"] = (
    "ASR 权重（用户指定换 1.7B）。实测代价：比 0.6B 慢 1.6~2 倍"
    "（单句 625~1138ms vs 281~660ms），内存 +768MB"
    "（1223+211MB vs 462+204MB）。回滚：把 asr_gguf/asr_mmproj 指回 "
    "qwen3-asr-gguf 目录里的 0.6B 两个文件即可。")

# ② 废弃 DSH
b["enable_dsh"] = False
t["dsh_new"] = []
t["dsh_run"] = []
b["_comment_dsh_disabled"] = (
    "用户决定废弃 DSH 会话方案，语音 agent 只做基础命令"
    "（打开应用 / 打开网址 / 搜索）。触发词已清空 + enable_dsh=false；"
    "dsh_session.py 与 act_dsh_* 代码保留未删，恢复只需把触发词和开关加回来。")

# ③ 热词去掉会话类命令词，保留通用命令词
b["asr_hotwords_extra"] = ["启动", "打开", "搜索", "精确查找", "文件", "网址"]

P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

print("config 写入完成")
print("  asr_gguf   =", Path(b["asr_gguf"]).name, Path(b["asr_gguf"]).exists())
print("  asr_mmproj =", Path(b["asr_mmproj"]).name, Path(b["asr_mmproj"]).exists())
print("  enable_dsh =", b["enable_dsh"], " dsh 触发词 =",
      t["dsh_new"], t["dsh_run"])
print("  热词 extra =", b["asr_hotwords_extra"])
