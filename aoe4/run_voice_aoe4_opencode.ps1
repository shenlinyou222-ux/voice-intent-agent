# 语音意图识别 —— 意图域 aoe4（常开监听，直接说话；开关见 intent_config.json → intent_registry）
# [OPENCODE] 与 run_voice_aoe4.ps1 的唯一区别：多设 ASR 热词偏置（不改任何既有文件）
#   匹配哪些意图：只由 voice_agent/intent_config.json 的 intent_registry（每条 enabled）决定，
#   改完保存即生效（动态热加载）。OPENCODE_FASTPATH* 只是测试/应急用的旧表开关，生产不设。
# 回滚：直接用原来的桌面快捷方式（run_voice_aoe4.cmd）
param([string]$Mode = 'config')
$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = "语音意图识别 —— 意图域 aoe4"

chcp 65001 > $null
[Console]::OutputEncoding = New-Object Text.UTF8Encoding $false

$here = $PSScriptRoot
$root = Split-Path -Parent $here
Set-Location -LiteralPath $root
$env:PYTHONWARNINGS = 'ignore'
$env:PYTHONIOENCODING = 'utf-8'
# [OPENCODE] 意图开关只认 intent_config.json → intent_registry（不设 OPENCODE_FASTPATH/TABLE）
# 仅保留 ASR 热词偏置（提升"补农"等词识别率，与路由无关）
$env:OPENCODE_ASR_BIAS = '1'

Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' 语音意图识别  -  意图域：日常/游戏/运维  [opencode]' -ForegroundColor Cyan
Write-Host ' 用法：常开监听，直接说话（不用按键）' -ForegroundColor Yellow
Write-Host ' 常说：打开B站 / 命令列表 / 切换领域（游戏/运维/日常）' -ForegroundColor Yellow
Write-Host '       调整静音判定（现场校准静音窗）｜ 开始输入 / 终止输入（听写开关）' -ForegroundColor Yellow
Write-Host ' 听写：非命令的话 → 打进当前焦点输入框（浏览器/聊天框/终端都行）' -ForegroundColor Gray
Write-Host ' 开关：全部意图的开关在 voice_agent\intent_config.json → intent_registry' -ForegroundColor Gray
Write-Host '       （每条 enabled: true/false；域 = 日常(默认)/游戏/运维/system；改完即生效）' -ForegroundColor DarkGray
Write-Host ' 日志：run_aoe4_voice_opencode.log  |  opencode_fastpath\logs\' -ForegroundColor Gray
Write-Host '       只看转写+耗时+结果；细节视角：启动前设 VOICE_AGENT_LOG=debug' -ForegroundColor DarkGray
Write-Host ' 回滚：用桌面原快捷方式（run_voice_aoe4.cmd）' -ForegroundColor DarkGray
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''

$py = 'E:\voice_agent_build\Scripts\python.exe'
# [OPENCODE] 启动探针（debug 级别才打印；INFO 只留转写/耗时/动作）
if ($env:VOICE_AGENT_LOG -eq 'debug') {
& $py -X utf8 -c "import sys; sys.path.insert(0, r'.'); from actions import Config; from opencode_fastpath import intents; c=Config(r'aoe4\profiles\aoe4-v1\intent_config.aoe4.json'); print('[OPENCODE] ' + intents.summary(c))" 2>&1 | Tee-Object -FilePath (Join-Path $here 'run_aoe4_voice_opencode.log') | Out-Null
}
# [OPENCODE] 听写状态探针（debug 级别才打印）
if ($env:VOICE_AGENT_LOG -eq 'debug') {
& $py -X utf8 -c "import json,io; b=json.load(io.open(r'aoe4\profiles\aoe4-v1\intent_config.aoe4.json',encoding='utf-8'))['behavior']; print('[OPENCODE] 听写 = %s ｜ 模式=%s ｜ 优先级=%s ｜ 逃生前缀=%s ｜ 标点→空格=%s ｜ 分段拼接=%s(%.1fs)' % ('开' if b.get('enable_dictation') else '关', b.get('dictation_mode_default'), b.get('dictation_priority'), b.get('dictation_bypass_prefix'), '开' if b.get('dictation_punct_to_space') else '关', '开' if b.get('dictation_stitch') else '关', float(b.get('dictation_stitch_gap_s') or 0)))" 2>&1 | Tee-Object -Append -FilePath (Join-Path $here 'run_aoe4_voice_opencode.log')
}
$inner = '"' + $py + '" -X utf8 -u .\voice_agent.py --config aoe4\profiles\aoe4-v1\intent_config.aoe4.json 2>&1'
cmd /c $inner | Tee-Object -Append -FilePath (Join-Path $here 'run_aoe4_voice_opencode.log')

Write-Host ''
Write-Host '（agent 已退出）' -ForegroundColor DarkGray
