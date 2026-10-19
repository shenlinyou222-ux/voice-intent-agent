# -*- coding: utf-8 -*-
"""OpenRA 语音控制桥（voice agent 侧）。

目录结构
--------
    openra/
      bridge_client.py          客户端：NDJSON over TCP，和游戏内 VoiceBridge trait 对话
      game_profile.py           版本化 profile：热词 / 意图语料 / 别名 / 规则直通
      game_actions.py           执行侧动作（注册进 voice_agent/actions.py 的 ACTIONS）
      profiles/openra-v1/       一个 profile = 一整套版本化的热词+意图+索引
      tools/make_profile.py     生成 profile（内容变了就出新版本）
      tools/build_profile_index.py  重建 embedding 索引（= 换域"重训练"）
      tools/selftest_bridge.py  用假游戏验证协议
      DESIGN.md                 设计文档（含 OpenRA 玩家 API 清点统计）

游戏侧
------
    E:\\OpenRA\\OpenRA.Mods.Common\\Traits\\VoiceControl\\VoiceBridge.cs
    注册在 mods/<mod>/rules/world.yaml 的 World: 节点（ra/cnc/d2k/ts 都加了）

一句话：语音 → ASR(版本化热词) → 意图(规则直通 + 版本化 embedding 索引) → 动作
        → 本地 TCP → 游戏内 trait → world.IssueOrder() → 和玩家点击同一条路径
"""
from .game_profile import GameProfile, available, load  # noqa: F401

__all__ = ["GameProfile", "available", "load"]
