@echo off
rem ---------------------------------------------------------------
rem  Voice Intent Recognition (intent domain: aoe4)
rem  desktop shortcut: 语音意图识别
rem  keep ASCII + CRLF -- cmd.exe mis-parses LF-only / non-ASCII .cmd
rem ---------------------------------------------------------------
title Voice Intent Recognition - aoe4 domain
cd /d "%~dp0"

set "VOICE_AGENT_C_STDERR=%~dp0c_stderr_aoe4.log"
set OMP_WAIT_POLICY=PASSIVE
set KMP_BLOCKTIME=0

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_voice_aoe4.ps1"

echo.
echo (voice agent exited - log is above)
pause
