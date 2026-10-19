# -*- coding: utf-8 -*-
"""运维语音域（ops）—— 与 aoe4 / openra 同构的第三个域。

目录结构：
    ops/
      __init__.py
      game_profile.py       规则层 + 三档路由（规则 → embedding → AI）
      game_actions.py       GAME_ACTIONS = { ops_*: 动作函数 }
      render.py             rich 渲染（短输出）
      window.py             拉 Windows Terminal 窗口 + 高亮（长输出）
      tools/build_index.py  建句向量索引
      profiles/ops-v1/      profile.json + hotwords.txt + intents.jsonl + intent_config.ops.json
      run_voice_ops.ps1     Windows 启动器
      run_voice_ops.cmd     桌面快捷方式入口

接线（在 voice_agent/actions.py 里加 3 行）：
    from ops import game_actions as _ops_actions
    ACTIONS.update(_ops_actions.GAME_ACTIONS)
    _OPS = _ops_actions
"""

__all__ = ["game_profile", "game_actions", "render", "window"]
