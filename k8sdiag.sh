#!/bin/bash
OUT="/mnt/c/Users/user/Desktop/ds harness/voice_agent/k8sdiag.txt"
exec > "$OUT" 2>&1
K="timeout 30 k3s kubectl"
echo "############ 1. PODS (all ns, wide) ############"
$K get pods -A -o wide 2>&1 | head -32
echo; echo "############ 2. NODE: capacity/allocatable/taints ############"
$K get nodes -o wide 2>&1
$K describe node 2>&1 | sed -n '/^Capacity:/,/^System Info:/p' | head -30
$K describe node 2>&1 | sed -n '/^Allocatable:/,/^System Info:/p' | head -22
echo "--- taints ---"; $K describe node 2>&1 | grep -A4 "^Taints:" | head -8
echo; echo "############ 3. describe: State/Reason/LastState/Events ############"
for p in $($K get pods -n default --no-headers 2>/dev/null | grep -E "vllm-asr|vllm-embed|llama-asr" | awk '{print $1}' | head -5); do
  echo "======== POD $p ========"
  $K describe pod $p -n default 2>&1 | sed -n '/^Status:/,/^IP:/p' | head -20
  echo "-- Last State / Reason --"
  $K describe pod $p -n default 2>&1 | grep -E "Last State|Reason:|Exit Code|Restart Count|State:" | head -12
  echo "-- Events --"
  $K describe pod $p -n default 2>&1 | sed -n '/^Events:/,$p' | head -18
done
echo; echo "############ 4. LOGS (current + previous) ############"
for p in $($K get pods -n default --no-headers 2>/dev/null | grep -E "vllm-asr" | awk '{print $1}' | head -2); do
  echo "---- $p --tail=20 ----"; $K logs $p -n default --tail=20 --all-containers 2>&1 | head -24
  echo "---- $p --previous --tail=20 ----"; $K logs $p -n default --previous --tail=20 --all-containers 2>&1 | head -24
done
echo; echo "############ 5. DEPLOY 要点 (image/args/resources/probes/gpu) ############"
for d in vllm-asr llama-asr vllm-embed merged-intent; do
  echo "======== DEPLOY $d ========"
  echo "replicas=$($K get deploy $d -n default -o jsonpath='{.spec.replicas}' 2>&1) ready=$($K get deploy $d -n default -o jsonpath='{.status.readyReplicas}' 2>&1)"
  $K get deploy $d -n default -o jsonpath='{range .spec.template.spec.containers[*]}image: {.image}{"\n"}command: {.command}{"\n"}args: {.args}{"\n"}resources: {.resources}{"\n"}readiness: {.readinessProbe}{"\n"}liveness: {.livenessProbe}{"\n"}{end}' 2>&1 | head -14
  echo "nodeSelector=$($K get deploy $d -n default -o jsonpath='{.spec.template.spec.nodeSelector}' 2>&1) tolerations=$($K get deploy $d -n default -o jsonpath='{.spec.template.spec.tolerations}' 2>&1)"
done
echo; echo "############ 6. ENDPOINTS + 资源用量 ############"
$K get endpoints -n default 2>&1 | head -14
echo "--- top pods ---"; $K top pods -n default 2>&1 | head -14
echo "--- top nodes ---"; $K top nodes 2>&1 | head -5
echo; echo "############ 7. EVENTS (recent) ############"
$K get events -A --sort-by=.lastTimestamp 2>&1 | tail -22
echo "=== DONE ==="