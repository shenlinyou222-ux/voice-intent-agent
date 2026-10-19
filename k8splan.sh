#!/bin/bash
OUT="/mnt/c/Users/user/Desktop/ds harness/voice_agent\k8splan.txt"
exec > "$OUT" 2>&1
K="timeout 30 k3s kubectl"
echo "############ A. 本地已有镜像（看有没有 CUDA 版 llama.cpp）############"
timeout 40 k3s ctr -n k8s.io images ls -q 2>&1 | grep -iE "llama|cuda|vllm" | head -20
echo "--- 全部镜像数 ---"; timeout 40 k3s ctr -n k8s.io images ls -q 2>&1 | wc -l
echo; echo "############ B. llama-asr 完整定义（挂载/卷/资源）############"
$K get deploy llama-asr -n default -o yaml 2>&1 | sed -n '1,120p'
echo; echo "############ C. vllm-embed 完整定义（要改成 CPU 跑）############"
$K get deploy vllm-embed -n default -o yaml 2>&1 | sed -n '/spec:/,$p' | head -80
echo; echo "############ D. /models 里到底有什么（宿主 + 集群视角）############"
ls -la /home/user/npc-models/ 2>&1 | head -15
echo "--- qwen3-asr-1.7b ---"; ls -la /home/user/npc-models/qwen3-asr-1.7b/ 2>&1 | head
echo "--- official 目录（是不是空的）---"; ls -la /home/user/npc-models/qwen3-asr-1.7b-official/ 2>&1 | head
echo "--- 0.6b official（vLLM 在用的）---"; ls /home/user/npc-models/qwen3-asr-0.6b-official/ 2>&1 | head
echo; echo "############ E. 显存现状（WSL 视角）############"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>&1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | head -10
echo; echo "############ F. GPU 设备健康（插件为什么报 unhealthy）############"
$K describe node 2>&1 | grep -iA6 "nvidia.com/gpu" | head -30
$K logs -n kube-system -l name=nvidia-device-plugin-ds --tail=25 2>&1 | tail -25
echo "=== DONE ==="