# -*- coding: utf-8 -*-
"""生成【游戏部署专用】的 intent_config（游戏意图独立部署，桌面意图完全不加载）。

产物：openra/profiles/<id>/intent_config.openra.json

用法：
    python voice_agent.py --config "openra\\profiles\\openra-v1\\intent_config.openra.json"

它做四件事：
  1. 以现有 intent_config.json 的 behavior 为底（ASR/VAD/音频参数全部继承，
     不动那些实测调出来的阈值）
  2. 把 intents / triggers 换成 profile 里的【游戏意图】
  3. apps / urls / files 清空 —— 用户明确说桌面那套意图可以完全丢掉
  4. behavior 里打开 openra_mode，并把意图后端指到【版本化的游戏 embedding 索引】
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPENRA_DIR = HERE.parent
VOICE_DIR = OPENRA_DIR.parent
PROFILES = OPENRA_DIR / "profiles"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="openra-v1")
    ap.add_argument("--base", default=str(VOICE_DIR / "intent_config.json"))
    args = ap.parse_args()

    pdir = PROFILES / args.id
    profile = json.loads((pdir / "profile.json").read_text(encoding="utf-8"))
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))

    behavior = dict(base.get("behavior", {}))
    behavior.update({
        # ── 游戏模式开关 ──
        "openra_mode": True,
        "openra_profile": args.id,
        "openra_host": profile["bridge"]["host"],
        "openra_port": int(profile["bridge"]["port"]),
        "openra_token": profile["bridge"].get("token"),
        "openra_timeout": float(profile["bridge"].get("timeout", 3.0)),
        # ── 意图后端：版本化的游戏 embedding 索引 ──
        "intent_backend": "embed",
        "intent_index": f"openra/profiles/{args.id}/{profile.get('index_file', 'index.json')}",
        "intent_embed_url": profile["embed"]["url"],
        "intent_embed_thr": float(profile["embed"].get("thr", 0.30)),
        "intent_embed_margin": float(profile["embed"].get("margin", 0.0)),
        # ── 规则直通（坐标/单位/姿态是确定性的，优先于模型）──
        "rule_direct": True,
        # ── 流式抢跑（A 路）：按住期间就边说边判边执行 ──
        #    stream_enable=流式转写；stream_exec=抢跑的动作【真的执行】
        #    B 路（松手整段权威）会跳过 A 路已执行的动作，不会做两遍
        "stream_enable": True,
        "stream_exec": True,
        "stream_url": "ws://172.20.71.101:30805/v1/realtime",
        "stream_model": "qwen3-asr",
        "stream_start_ms": 300,
        # ── ASR 热词完全来自 profile（版本化），不再用桌面闭集 ──
        "asr_hotwords": True,
        # 游戏口令是中文（矿车/电厂/展开…），必须喂中文热词，不能只喂英文名
        "asr_hotword_ascii_only": False,
        # ⚠️ 原来继承桌面的 160 字上限 → 单位名被截掉，「精炼厂」被转成「金炼厂」。
        #    游戏域的词表以单位/建筑名为主，放宽到 420 字（prompt 依然很短）。
        "asr_hotword_chars": 420,
        "asr_hotwords_extra": [],
    })

    cfg = {
        "_README": [
            "═══════════════════════════════════════════════════════════════",
            f" OpenRA 游戏部署配置（profile: {args.id}）—— 自动生成，别手改",
            " 生成脚本：openra/tools/make_game_config.py",
            "",
            " 与桌面版 intent_config.json 的区别：",
            "   · 意图只有游戏指令（19 个），桌面意图（打开程序/网址/搜索）完全不加载",
            "   · ASR 热词来自 openra/profiles/<id>/hotwords.txt（版本化）",
            "   · 意图分类用版本化的 embedding 索引（openra/profiles/<id>/index.json）",
            "   · 指令走本地 TCP → 游戏内 VoiceBridge trait → world.IssueOrder()",
            "",
            f" 版本戳：hotwords={profile['hotwords_version']}",
            f"         intent  ={profile['intent_version']}",
            f"         index   ={profile['index_version']}",
            "═══════════════════════════════════════════════════════════════",
        ],
        "version": base.get("version", 2),
        "language": base.get("language", "zh-CN"),
        "profile": args.id,
        "profile_versions": {
            "hotwords_version": profile["hotwords_version"],
            "intent_version": profile["intent_version"],
            "index_version": profile["index_version"],
        },
        # 游戏意图独立部署：这里就是全部意图
        "intents": profile["intents"],
        # 桌面闭集全部清空
        "apps": [],
        "urls": [],
        "files": [],
        "search": base.get("search", {}),
        # 触发词 = 游戏触发词（版本化 profile 里那份）
        "triggers": profile["triggers"],
        "behavior": behavior,
    }

    out = pdir / "intent_config.openra.json"
    out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written: {out}")
    print(f"  intents   : {len(cfg['intents'])}（全部是 game_*）")
    print(f"  triggers  : {len(cfg['triggers'])} 组")
    print(f"  index     : {behavior['intent_index']}")
    print(f"  bridge    : {behavior['openra_host']}:{behavior['openra_port']}")
    print(f"  versions  : {cfg['profile_versions']}")


if __name__ == "__main__":
    main()
