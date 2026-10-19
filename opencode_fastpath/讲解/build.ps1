# build.ps1 —— xelatex 连跑 3 遍（TOC/交叉引用需要多趟）；不要用 latexmk（本机缺 perl）
$ErrorActionPreference = 'Stop'
$miktex = $env:MIKTEX_BIN
if (-not $miktex) { $miktex = [Environment]::GetEnvironmentVariable('MIKTEX_BIN', 'User') }
if (-not $miktex) { $miktex = 'E:\MiKTeX\texmfs\install\miktex\bin\x64' }
$env:Path = "$miktex;" + $env:Path
Set-Location $PSScriptRoot
foreach ($i in 1..3) {
  Write-Host "==== pass $i ===="
  & "$miktex\xelatex.exe" -interaction=nonstopmode -file-line-error main.tex | Out-Null
}
Write-Host "==== result ===="
if (Test-Path main.pdf) { Get-Item main.pdf | Select-Object Name, Length, LastWriteTime }
Select-String -Path main.log -Pattern 'Overfull|Underfull|Undefined|Missing character' |
  Select-Object -First 40 | ForEach-Object { $_.Line }
