# 语音意图识别 —— 意图域 aoe4（带实时转写的可见窗口）
# 由 aoe4\run_voice_aoe4.cmd 或桌面快捷方式启动
$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = '语音意图识别 —— 意图域 aoe4（常开监听）'

chcp 65001 > $null
[Console]::OutputEncoding = New-Object Text.UTF8Encoding $false

$here = $PSScriptRoot                       # ...\voice_agent\aoe4
$root = Split-Path -Parent $here            # ...\voice_agent
Set-Location -LiteralPath $root

# 屏蔽第三方库的 UserWarning（torch 的 TypedStorage/quantize 弃用提示，纯噪声，不是故障）
$env:PYTHONWARNINGS = 'ignore'
# [OPENCODE] 用配置里的 ASR 热词（从唯一意图清单自动生成；α 保持 6）
$env:OPENCODE_ASR_BIAS = '1'

Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' 语音意图识别  -  意图域 aoe4' -ForegroundColor Cyan
Write-Host ' 用法：常开监听，直接说话（不用按键）' -ForegroundColor Yellow
Write-Host ' ------------------------------------------------------------' -ForegroundColor DarkGray
Write-Host ' 分段  : Silero VAD 概率判据（prob0.4 / hits2 / misses5 / 平滑3）' -ForegroundColor Gray
Write-Host ' 预滚  : 动态 min(640ms, 段总长/5) + 静音裁剪（相对阈值 + 两者取小）' -ForegroundColor Gray
Write-Host ' ASR   : vLLM(GPU) 集群 30805  ← Qwen3-ASR-0.6B 官方 bf16，实测 ~100ms' -ForegroundColor Gray
Write-Host ' 偏置  : 只给「补农」加 logit_bias α=6（软偏置，不影响拒识）' -ForegroundColor Gray
Write-Host ' 屏蔽  : token 层屏蔽繁体字形（bad_words，保证只出简体）' -ForegroundColor Gray
Write-Host ' 闭集  : 补农一族（profile.json 的 closed_set，改这里就行）' -ForegroundColor Gray
Write-Host ' 通道  : 写 vb_cmd_*.txt（带递增序号）→ 游戏内 SCAR 脚本 0.1s 轮询执行' -ForegroundColor Gray
Write-Host ' 延迟  : 说完→游戏生效 ≈ 0.25~0.35s（检测150 + ASR~100 + 意图0 + 动作~3）' -ForegroundColor Gray
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''

# ★ 用 cmd 在【进程层】把 stderr 合并进 stdout：
#   llama.cpp 的库会把自己的信息日志（llama_context: … / clip_encode: …）写到 stderr，
#   而 PowerShell 会把任何 stderr 输出当成错误，套上一堆红字 NativeCommandError ✗。
#   经 cmd /c "... 2>&1" 合并后，PowerShell 只看到 stdout → 不再有红字 ✓
$py = 'E:\voice_agent_build\Scripts\python.exe'
$inner = '"' + $py + '" -X utf8 -u .\voice_agent.py --config aoe4\profiles\aoe4-v1\intent_config.aoe4.json 2>&1'
cmd /c $inner | Tee-Object -FilePath (Join-Path $here 'run_aoe4_voice.log')

Write-Host ''
Write-Host '（agent 已退出，窗口留着方便你看最后的日志；关掉窗口即可）' -ForegroundColor DarkGray
