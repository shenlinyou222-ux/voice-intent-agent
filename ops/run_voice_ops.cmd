@echo off
rem ---------------------------------------------------------------
rem  voice_agent - ops voice control (read-only passthrough / change needs confirm)
rem  desktop shortcut target
rem  keep ASCII + CRLF -- cmd.exe mis-parses LF-only / non-ASCII .cmd
rem ---------------------------------------------------------------
title voice_agent - ops voice control
cd /d "%~dp0"

set "VOICE_AGENT_C_STDERR=%~dp0c_stderr_ops.log"
rem 不限线程时 torch/OpenMP 推理后自旋 -> 常开监听空转 CPU 220%（见 load_model 注释）
set OMP_WAIT_POLICY=PASSIVE
set KMP_BLOCKTIME=0

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_voice_ops.ps1"

echo.
echo (voice agent exited - log is above)
pause
