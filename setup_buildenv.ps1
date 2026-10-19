# 建一个只含 CPU 版 torch 的瘦环境，用于 PyInstaller 打包
# 目的：把 exe 体积从 ~2.9 GB（含 CUDA torch）压到 ~400 MB
$ErrorActionPreference = "Continue"
$VENV = "E:\voice_agent_build"
$PY   = "$VENV\Scripts\python.exe"

Write-Output "=== ① 建 venv（用 E:\py312 做 base）==="
if (Test-Path $PY) {
    Write-Output "  已存在，跳过"
} else {
    & "E:\py312\python.exe" -m venv $VENV 2>&1 | Select-Object -Last 5
}
& $PY --version

Write-Output ""
Write-Output "=== ② 装 CPU 版 torch ==="
& $PY -m pip install --quiet --disable-pip-version-check --upgrade pip 2>&1 | Select-Object -Last 3
& $PY -m pip install --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | Select-Object -Last 8

Write-Output ""
Write-Output "=== ③ 装其余依赖 ==="
& $PY -m pip install --disable-pip-version-check transformers safetensors numpy sounddevice webrtcvad-wheels pyinstaller 2>&1 | Select-Object -Last 8
# pypinyin：ASR 同音字纠错（「打开直呼」→知乎，zhihu==zhihu）见 actions._homophone_hit
& $PY -m pip install --disable-pip-version-check pypinyin 2>&1 | Select-Object -Last 8

Write-Output ""
Write-Output "=== ④ 校验 ==="
& $PY -c @"
import importlib, sys
print('  python', sys.version.split()[0])
for m in ('torch','transformers','safetensors','numpy','sounddevice','webrtcvad','PyInstaller'):
    try:
        mod = importlib.import_module(m)
        print(f'  OK   {m:14s} {getattr(mod,\"__version__\",\"?\")}')
    except Exception as e:
        print(f'  MISS {m:14s} {type(e).__name__}')
import torch
print('  cuda available:', torch.cuda.is_available())
"@ 2>&1 | Select-Object -Last 12

Write-Output ""
Write-Output "=== ⑤ 体积 ==="
$sp = "$VENV\Lib\site-packages"
foreach ($d in @("torch","transformers","numpy")) {
    $p = Join-Path $sp $d
    if (Test-Path $p) { $s = (Get-ChildItem $p -Recurse -File -EA SilentlyContinue | Measure-Object Length -Sum).Sum; Write-Output ("  {0,-14} {1,8:N0} MB" -f $d, ($s/1MB)) }
}
$tot = (Get-ChildItem $VENV -Recurse -File -EA SilentlyContinue | Measure-Object Length -Sum).Sum
Write-Output ("  {0,-14} {1,8:N0} MB" -f "TOTAL", ($tot/1MB))
