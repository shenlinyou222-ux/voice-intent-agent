@echo off
title voice_agent - transcription monitor
cd /d "%~dp0"
set OMP_WAIT_POLICY=PASSIVE
set KMP_BLOCKTIME=0
rem C 层日志（llama.cpp/mtmd/torch）分流到独立文件，主日志只留应用层
set VOICE_AGENT_C_STDERR=%~dp0c_stderr.log
echo ============================================================
echo  语音意图识别  -  常开监听（窗口监控转写）
echo  ASR   : 进程内 llama.cpp (GGUF Q4_K_M, C++) + 热词(命令词)
echo  intent: 纯文本路 (dmeta int8) + 规则直通
echo  mode  : 真执行（会真的开浏览器/起程序/起 DSH 会话）
echo  log   : run_src.log      C层日志: c_stderr.log
echo  dry   : 想只看不执行，把下面命令末尾加上 --dry-run
echo ============================================================
echo.
powershell -NoProfile -NoExit -Command "chcp 65001 > $null; [Console]::OutputEncoding = New-Object Text.UTF8Encoding $false; & 'E:\voice_agent_build\Scripts\python.exe' -X utf8 -u '.\voice_agent.py' *>&1 | Tee-Object -FilePath '.\run_src.log'"
