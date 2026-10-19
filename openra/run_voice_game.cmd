@echo off
rem ---------------------------------------------------------------
rem  voice control for OpenRA    desktop shortcut: voice control
rem  keep ASCII + CRLF -- cmd.exe mis-parses LF-only / non-ASCII .cmd
rem ---------------------------------------------------------------
title voice_agent - OpenRA voice control
cd /d "%~dp0"

set "VOICE_AGENT_C_STDERR=%~dp0c_stderr_game.log"
set OMP_WAIT_POLICY=PASSIVE
set KMP_BLOCKTIME=0

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_voice_game.ps1"

echo.
echo (voice agent exited - log is above)
pause
