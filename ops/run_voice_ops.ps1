# 运维语音（ops）—— 带实时转写的可见窗口
# 由 ops\run_voice_ops.cmd 或桌面快捷方式启动
$ErrorActionPreference = 'Continue'
$Host.UI.RawUI.WindowTitle = 'voice_agent - 运维语音（只读直通 / 变更需确认）'

chcp 65001 > $null
[Console]::OutputEncoding = New-Object Text.UTF8Encoding $false

$here = $PSScriptRoot                       # ...\voice_agent\ops
$root = Split-Path -Parent $here            # ...\voice_agent
Set-Location -LiteralPath $root

# 屏蔽第三方库的 UserWarning（torch 的 TypedStorage/quantize 弃用提示，纯噪声）
$env:PYTHONWARNINGS = 'ignore'
# 让 rich 在真实终端里着色；被 Tee 重定向时会自动降级（render.py 判 isatty）
$env:FORCE_COLOR = '1'

Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' voice_agent  -  运维语音' -ForegroundColor Cyan
Write-Host ' 用法：常开监听，直接说话（不用按键）' -ForegroundColor Yellow
Write-Host ' ------------------------------------------------------------' -ForegroundColor DarkGray
Write-Host ' 三档路由 : 规则直通(0ms) → embedding kNN(~30ms) → AI 升级(DSH)' -ForegroundColor Gray
Write-Host ' 安全分层 : 只读直通 / 变更【两步确认】/ 危险动词不进闭集' -ForegroundColor Gray
Write-Host ' 分段     : Silero VAD 概率判据（prob0.4 / hits2 / misses5 / 平滑3）' -ForegroundColor Gray
Write-Host ' ASR      : vLLM(GPU) 集群 30805 ← Qwen3-ASR-0.6B 官方 bf16，实测 ~131ms' -ForegroundColor Gray
Write-Host ' 偏置     : 热词走 logits 偏置 α=6（中英混杂：kubectl/pod/TTFT/vLLM…）' -ForegroundColor Gray
Write-Host ' 输出     : 短输出 rich 渲染；长输出拉 Windows Terminal + pygmentize 高亮' -ForegroundColor Gray
Write-Host ' ------------------------------------------------------------' -ForegroundColor DarkGray
Write-Host ' 可说（只读）:' -ForegroundColor Green
Write-Host '   查磁盘 / 查内存 / 查负载 / 查显卡 / 查端口 / 查进程 / 查服务 / 看日志' -ForegroundColor Gray
Write-Host '   查集群 / 查所有pod / 查节点 / 查服务列表' -ForegroundColor Gray
Write-Host '   查TTFT / 查TPOT / 查KV缓存 / 查前缀缓存 / 查排队 / 查吞吐' -ForegroundColor Gray
Write-Host '   打开面板 / 打开终端' -ForegroundColor Gray
Write-Host ' 可说（变更，会先问一遍）:' -ForegroundColor Yellow
Write-Host '   重启nginx / 重启docker / 停掉xxx  →  再说「确认」才真执行' -ForegroundColor Gray
Write-Host ' 可说（AI 升级）:' -ForegroundColor Magenta
Write-Host '   让 ds 看看这个 yaml / 这个 pod 怎么挂了 / 分析一下日志' -ForegroundColor Gray
Write-Host '   （会自动带上你上一条命令的输出，不用复述）' -ForegroundColor DarkGray
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''

# ★ 用 cmd 在【进程层】把 stderr 合并进 stdout：
#   llama.cpp 的库把信息日志写到 stderr，而 PowerShell 会把任何 stderr 输出当错误，
#   套上一堆红字 NativeCommandError。经 cmd /c "... 2>&1" 合并后只剩 stdout，不再有红字。
$py = 'E:\voice_agent_build\Scripts\python.exe'
$inner = '"' + $py + '" -X utf8 -u .\voice_agent.py --config ops\profiles\ops-v1\intent_config.ops.json 2>&1'
cmd /c $inner | Tee-Object -FilePath (Join-Path $here 'run_ops_voice.log')

Write-Host ''
Write-Host '（agent 已退出，窗口留着方便你看最后的日志；关掉窗口即可）' -ForegroundColor DarkGray
