# 语音意图识别 —— 意图域 openra（带实时转写的可见窗口）
# 由 openra\run_voice_game.cmd 或 install 时手动启动
$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = '语音意图识别 —— 意图域 openra（按住右 Ctrl）'

chcp 65001 > $null
[Console]::OutputEncoding = New-Object Text.UTF8Encoding $false

$here = $PSScriptRoot                       # ...\voice_agent\openra
$root = Split-Path -Parent $here            # ...\voice_agent
Set-Location -LiteralPath $root

Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' 语音意图识别  -  意图域 openra' -ForegroundColor Cyan
Write-Host ' 用法：按住【右 Ctrl】说话 → 松开执行（不按就不听）' -ForegroundColor Yellow
Write-Host ' 反馈：游戏内聊天框会回一行中文 + 小地图雷达标记' -ForegroundColor Yellow
Write-Host ' ASR   : vLLM http://172.20.71.101:30805' -ForegroundColor Gray
Write-Host ' intent: 规则直通 + 游戏专用索引 openra-v2（下单/放置/取消/编队/探索）' -ForegroundColor Gray
Write-Host ' bridge: 127.0.0.1:38900' -ForegroundColor Gray
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''

& 'E:\voice_agent_build\Scripts\python.exe' -X utf8 -u '.\voice_agent.py' `
    --config 'openra\profiles\openra-v2\intent_config.openra.json' *>&1 |
    Tee-Object -FilePath (Join-Path $here 'run_game_voice.log')

Write-Host ''
Write-Host '（agent 已退出，窗口留着方便你看最后的日志；关掉窗口即可）' -ForegroundColor DarkGray
