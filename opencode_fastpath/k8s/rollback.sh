#!/bin/bash
# [OPENCODE] 回滚 vLLM realtime 路由（恢复打补丁前的 args 全量数组）
set -e
kubectl patch deploy vllm-asr --type=json \
  --patch-file "/mnt/c/Users/user/Desktop/ds harness/voice_agent/opencode_fastpath/k8s/rollback_args.json"
kubectl rollout status deploy/vllm-asr --timeout=300s
