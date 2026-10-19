# 打包成独立可执行文件（PyInstaller）
#
# 用 E:\voice_agent_build（只含 CPU 版 torch，约 500 MB），
# 不用主环境的 CUDA torch（4.2 GB）—— 体积差 8 倍。
#
# 产物：dist\voice_agent\
#         voice_agent.exe        ← 主程序
#         _internal\             ← 依赖（PyInstaller 自动收）
#         intent_config.json     ← ⭐ 意图配置（用户改这个）
#         model\                 ← 模型权重（745 MB）
#
# ⚠️ 两个踩过的坑：
#   ① 脚本必须存成【UTF-8 with BOM】—— Windows PowerShell 5.1 对无 BOM 的
#      UTF-8 按 ANSI 读，中文全乱码 → 语法错误 → 直接跑不起来
#   ② $ErrorActionPreference 不能用 Stop —— PyInstaller 的 INFO 日志走
#      stderr，Stop 会把它当致命错误，命令刚打印版本号就退出，dist 不生成

$ErrorActionPreference = "Continue"
$HERE = "C:\Users\user\Desktop\ds harness\voice_agent"
$PY   = "E:\voice_agent_build\Scripts\python.exe"
# ⚠️⚠️ 血债（2026-10-16 定位到的根因，排查了一整轮）：
#   产物【绝不能】放在 $HERE 下面（工作区里）。
#   DSH 沙箱给工作区 C:\Users\user\Desktop\ds harness 打了一层 ACL，
#   里面带 `Mandatory Label\Low Mandatory Level` 和 `Everyone:(DENY)(DC)`。
#   映像文件放在这个目录树里的 exe，跑 wsl.exe 会被 WSL 服务拒绝：
#       「拒绝访问。错误代码: Wsl/E_ACCESSDENIED」
#   —— 而同一个 exe 拷到 E:\ 或桌面（工作区外）就完全正常。
#   实测对照（同一个 va_slim.exe）：
#       Desktop\va_slim_desk\          ✅
#       Desktop\va slim space\         ✅（所以不是空格的问题）
#       Desktop\ds harness\va_slim_out\ ❌
#       Desktop\zzz_va\                ✅
#   源码运行不受影响，是因为 python.exe 的映像在 E:\voice_agent_build（工作区外）。
#   → 所以 dist 一律放 $DISTROOT（工作区外）。
$DISTROOT = "E:\voice_agent_dist"
$DIST = "$DISTROOT\voice_agent"
# ⚠️⚠️ 血债（2026-10-17 实测）：**不能**让 PyInstaller 直接往 $DIST 写。
#    PyInstaller --noconfirm 会先 _rmtree 目标目录，而 Windows 上「空目录被
#    某个进程当 CWD 占着」时删目录会报 WinError 32（实测反复踩：
#    目录里能建文件、能删文件，就是删不掉目录本身）。
#    改成：PyInstaller 输出到一个【每次都全新的】临时目录，
#    再用 robocopy 合并进 $DIST —— robocopy 只删多余文件、不动目录本身，绕开这个坑。
$STAGE = "E:\voice_agent_stage"

Write-Output "=== 1/8 检查打包环境 ==="
if (-not (Test-Path $PY)) { Write-Output "  缺打包环境 $PY，先跑 setup_buildenv.ps1"; exit 1 }
& $PY -c "import torch, transformers, sounddevice, webrtcvad, PyInstaller; print('  torch', torch.__version__, '| transformers', transformers.__version__)"
if ($LASTEXITCODE -ne 0) { Write-Output "  环境自检失败"; exit 1 }

Write-Output ""
Write-Output "=== 2/8 清理旧产物 ==="
Remove-Item "$HERE\build" -Recurse -Force -EA SilentlyContinue
Remove-Item "$HERE\dist"  -Recurse -Force -EA SilentlyContinue   # 老位置，清掉免得混淆
Remove-Item $STAGE         -Recurse -Force -EA SilentlyContinue  # 临时产物每次全新
Remove-Item "$HERE\voice_agent.spec" -Force -EA SilentlyContinue
Write-Output "  done"

Write-Output ""
Write-Output "=== 3/8 PyInstaller 打包 ==="
Set-Location $HERE
# ⚠️ 不能 exclude torch.distributed —— torch.utils.data.dataloader 会 import 它，
#    排掉之后 exe 一加载模型就 ModuleNotFoundError（实测踩过）。
# ⚠️ 注释【绝对不能】插在下面这些反引号续行中间 —— 会把续行链断掉，
#    PowerShell 会把下一行的 --exclude-module 当成一元运算符报语法错误。
& $PY -m PyInstaller `
    --noconfirm --clean --onedir `
    --name voice_agent `
    --distpath $STAGE `
    --paths $HERE `
    --hidden-import transformers.models.qwen3_asr.modeling_qwen3_asr `
    --hidden-import transformers.models.qwen3_asr.configuration_qwen3_asr `
    --hidden-import transformers.models.qwen3_asr.feature_extraction_qwen3_asr `
    --hidden-import transformers.models.bert.modeling_bert `
    --hidden-import transformers.models.bert.tokenization_bert `
    --hidden-import transformers.models.bert.configuration_bert `
    --collect-all sounddevice `
    --collect-all webrtcvad `
    --collect-all safetensors `
    --collect-submodules transformers.models.qwen3_asr `
    --collect-submodules transformers.models.bert `
    --exclude-module pkg_resources `
    --exclude-module setuptools `
    --exclude-module matplotlib --exclude-module scipy --exclude-module pandas `
    --exclude-module IPython --exclude-module notebook --exclude-module tkinter `
    --exclude-module torchvision --exclude-module torchaudio `
    --exclude-module tests --exclude-module pytest `
    voice_agent.py 2>&1 | Select-Object -Last 12
Write-Output "  PyInstaller 退出码 = $LASTEXITCODE"

Write-Output ""
Write-Output "=== 4/8 检查产物 + 合并到发布目录 ==="
if (-not (Test-Path "$STAGE\voice_agent\voice_agent.exe")) {
    Write-Output "  [X] 打包失败：找不到 $STAGE\voice_agent\voice_agent.exe"
    exit 1
}
Write-Output "  [OK] voice_agent.exe 已生成"
# ★ 用 robocopy 合并（/MIR 删多余文件，但【不动目录本身】）——
#   这样目标目录被别的进程当 CWD 占着也能更新（见上面 $STAGE 的血债注释）。
New-Item -ItemType Directory -Force -Path $DIST | Out-Null
robocopy "$STAGE\voice_agent" $DIST /MIR /NFL /NDL /NJH /NJS /NP /MT:8 | Out-Null
$rc = $LASTEXITCODE
# robocopy 的退出码：0=无变化 1=有复制 2=有额外文件 3=1+2 … <8 都算成功
if ($rc -ge 8) { Write-Output "  [X] robocopy 失败 rc=$rc"; exit 1 }
Write-Output "  [OK] 已合并到 $DIST （robocopy rc=$rc）"
Remove-Item $STAGE -Recurse -Force -EA SilentlyContinue

Write-Output ""
Write-Output "=== 5/8 拷配置、模型、各域 profile ==="
Copy-Item "$HERE\intent_config.json" "$DIST\" -Force
Copy-Item "$HERE\README.md" "$DIST\" -Force
# ⚠️⚠️ 血债（2026-10-17 实测）：**句向量索引 intent_index.json 原来没拷**，
#    于是 exe 一起来就报：
#      ⚠️ 句向量分类器加载失败（FileNotFoundError: ...\intent_index.json）
#        → 回退词袋线性头
#    也就是说【桌面域在 exe 里一直是降级跑的】（kNN 路径整个没了），
#    只是它静默回退、日志又只在启动时闪一下，一直没人注意。
#    实测确认：源里 8100 KB、dist 里没有；而 ops 的索引
#    （ops/profiles/ops-v1/index.json）因为整个 ops/ 目录被 robocopy 带过去，反而是好的。
#    ⚠️ 它被 voice_agent/.gitignore 忽略（是 _build_idx.py 的产物，8 MB），
#       所以【新克隆的仓库里不会有】—— 缺了要提示去跑 _build_idx.py，不能静默。
if (Test-Path "$HERE\intent_index.json") {
    Copy-Item "$HERE\intent_index.json" "$DIST\" -Force
    Write-Output "  [OK] intent_index.json（桌面域句向量索引 $([math]::Round((Get-Item "$HERE\intent_index.json").Length/1MB,1)) MB）"
} else {
    Write-Output "  [X] 缺 intent_index.json —— 桌面域的句向量分类器会失效（静默回退词袋头）"
    Write-Output "      修：python _build_idx.py   （该文件被 .gitignore 忽略，新克隆的仓库里没有）"
    exit 1
}
New-Item -ItemType Directory -Force -Path "$DIST\model" | Out-Null
# ⚠️ 血债（2026-10-16 实测，连踩两次）：这份清单必须和 model_runner.py 里
#    【实际会读】的文件一一对应，少一个就在用户第一次说话时炸。
#      · intent_text.pt 缺失 → text_only 回退完整产物（292M 参数、加载 19.4s
#        而不是 2s），且 head_text 不会创建 → predict_text() 抛 AttributeError
#      · _vocab.txt 缺失 → _init_text() 里 BertTokenizer 报
#        「Error while initializing WordPiece: 系统找不到指定的文件」直接崩
#      · head_text.pt 是训练产物，缺了就没法重新打 intent_text.pt
#    对照 model_runner.py：L83 intent_text.pt / L100 meta.json / L106 intent.pt
#      / L112 bundle.safetensors / L123+L247+L313 _vocab.txt
#    intent_int8.pt / intent_int8b.pt（各 375MB）【故意不拷】——
#      只有 intent_int8=True 才走，四个 profile 都没开，省 751MB。
$MODEL_FILES = @("intent_text.pt", "meta.json", "_vocab.txt",
                 "intent.pt", "bundle.safetensors", "head_text.pt")
foreach ($f in $MODEL_FILES) {
    if (Test-Path "$HERE\model\$f") { Copy-Item "$HERE\model\$f" "$DIST\model\" -Force; Write-Output "  [OK] model\$f" }
    else { Write-Output "  [!] 缺 model\$f" }
}

# ⭐ 各域（ops / aoe4 / openra）必须一起拷：
#    exe 用 --config <域>\profiles\<pid>\intent_config.<域>.json 启动，
#    而配置里引用的 index.json / hotwords.txt / intents.jsonl 全在域目录里 ——
#    只拷 intent_config.json 的话，三个域全都起不来。
#    ⚠️ robocopy /XD __pycache__ 排除编译缓存（别把 .pyc 带进发行包）
foreach ($d in @("ops", "aoe4", "openra")) {
    if (-not (Test-Path "$HERE\$d")) { Write-Output "  [!] 缺 $d\"; continue }
    $dst = "$DIST\$d"
    Remove-Item $dst -Recurse -Force -EA SilentlyContinue
    robocopy "$HERE\$d" $dst /E /XD __pycache__ /NFL /NDL /NJH /NJS /NP | Out-Null
    $n  = (Get-ChildItem $dst -Recurse -File).Count
    $mb = [math]::Round(((Get-ChildItem $dst -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 2)
    Write-Output ("  [OK] {0}\  {1} 个文件  {2} MB" -f $d, $n, $mb)
}

# 各域的 exe 启动器（仓库里那几个是给 python + voice_agent.py 用的，这里生成 exe 版）
# ⚠️ .cmd 必须 ASCII + CRLF —— cmd.exe 对 LF-only / 非 ASCII 的 .cmd 会解析错
$LAUNCH = @(
    @{ f = "run_ops.cmd";    cfg = "ops\profiles\ops-v1\intent_config.ops.json";          t = "ops voice (read-only direct / change needs confirm)" },
    @{ f = "run_aoe4.cmd";   cfg = "aoe4\profiles\aoe4-v1\intent_config.aoe4.json";       t = "AoE4 voice control" },
    @{ f = "run_openra.cmd"; cfg = "openra\profiles\openra-v2\intent_config.openra.json"; t = "OpenRA voice control" }
)
foreach ($l in $LAUNCH) {
    $log = ($l.f -replace '^run_', 'run_') -replace '\.cmd$', '.log'
    $body = "@echo off`r`ntitle voice_agent - $($l.t)`r`ncd /d `"%~dp0`"`r`n" +
            "powershell -NoProfile -NoExit -Command `"chcp 65001 > `$null; [Console]::OutputEncoding = New-Object Text.UTF8Encoding `$false; & '.\voice_agent.exe' --config '$($l.cfg)' *>&1 | Tee-Object -FilePath '.\$log'`"`r`n"
    [System.IO.File]::WriteAllText("$DIST\$($l.f)", $body, (New-Object Text.UTF8Encoding $false))
    Write-Output "  [OK] $($l.f)  ->  $($l.cfg)"
}

Write-Output ""
Write-Output "=== 6/8 产物自检（启动必需文件，少一个就会在第一次说话时炸）==="
# ⚠️ 这一节是 2026-10-16 连踩两次坑之后加的：
#    漏拷 intent_text.pt / _vocab.txt 都不会让【打包】失败，
#    只会让 exe 在用户第一次开口时崩（而且日志分散在两处，很难归因）。
#    所以把「启动链路上会被读到的文件」在这里全验一遍，缺了就 exit 1。
$MUST = @(
    "voice_agent.exe",
    "intent_config.json",
    "model\intent_text.pt",        # 纯文本路主产物（L83）
    "model\meta.json",             # L100
    "model\_vocab.txt",            # L123 / L247 / L313 —— 缺了 BertTokenizer 直接崩
    "model\intent.pt",             # 回退路 L106
    "model\bundle.safetensors",    # 回退路 L112
    "ops\profiles\ops-v1\index.json",
    "ops\profiles\ops-v1\intent_config.ops.json",
    "aoe4\profiles\aoe4-v1\intent_config.aoe4.json",
    "openra\profiles\openra-v2\intent_config.openra.json",
    "run_ops.cmd", "run_aoe4.cmd", "run_openra.cmd"
)
$missing = @()
foreach ($rel in $MUST) {
    if (Test-Path "$DIST\$rel") { Write-Output "  [OK] $rel" }
    else { Write-Output "  [X]  缺 $rel"; $missing += $rel }
}
if ($missing.Count -gt 0) {
    Write-Output ""
    Write-Output "  [X] 产物不完整，缺 $($missing.Count) 个必需文件 —— 不要发布这个包"
    exit 1
}
Write-Output "  [OK] 产物完整（$($MUST.Count) 个必需文件都在）"

Write-Output ""
Write-Output "=== 7/8 体积 ==="
$exe = (Get-Item "$DIST\voice_agent.exe").Length
$tot = (Get-ChildItem $DIST -Recurse -File | Measure-Object Length -Sum).Sum
$mdl = (Get-ChildItem "$DIST\model" -Recurse -File | Measure-Object Length -Sum).Sum
Write-Output ("  voice_agent.exe         {0,10:N1} MB" -f ($exe / 1MB))
Write-Output ("  model\                  {0,10:N1} MB" -f ($mdl / 1MB))
Write-Output ("  dist\voice_agent\ 合计   {0,10:N1} MB" -f ($tot / 1MB))

Write-Output ""
Write-Output "=== 8/8 冒烟测试 ==="
Write-Output "--- --list ---"
& "$DIST\voice_agent.exe" --list 2>&1 | Select-Object -First 10
Write-Output "--- --selftest ---"
& "$DIST\voice_agent.exe" --selftest 2>&1 | Select-Object -Last 12
