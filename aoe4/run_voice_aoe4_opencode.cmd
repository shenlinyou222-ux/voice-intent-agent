@echo off
rem [OPENCODE] Voice Intent Recognition - aoe4 domain (opencode launcher)
rem usage:  run_voice_aoe4_opencode.cmd [live|shadow]     (default: live)
rem        MODE is legacy only; which intents match is decided by
rem        voice_agent\intent_config.json -> intent_registry (enabled)
rem rollback: use the original run_voice_aoe4.cmd / desktop shortcut
setlocal
set "MODE=%~1"
if "%MODE%"=="" set "MODE=live"
title Voice Intent Recognition - aoe4 (opencode %MODE%)
cd /d "%~dp0"
set "VOICE_AGENT_C_STDERR=%~dp0c_stderr_aoe4_opencode.log"
set OMP_WAIT_POLICY=PASSIVE
set KMP_BLOCKTIME=0
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_voice_aoe4_opencode.ps1" -Mode %MODE%
echo.
echo (voice agent exited - log is above)
pause
