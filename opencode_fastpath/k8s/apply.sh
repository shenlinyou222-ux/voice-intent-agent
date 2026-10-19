#!/bin/bash
# [OPENCODE] 启用 vLLM realtime 路由（在 WSL 里执行）
set -e
kubectl patch deploy vllm-asr --type=json \
  --patch-file "/mnt/c/Users/user/Desktop/ds harness/voice_agent/opencode_fastpath/k8s/patch_args.json"
kubectl rollout status deploy/vllm-asr --timeout=300s
