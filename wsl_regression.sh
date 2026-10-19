#!/usr/bin/env bash
# 回归检查：4 个 target 的 profile 都要能正常解析，且旧的 3 个行为不变
set -u
cd /mnt/e/habitica-setup
echo "════════ ① 4 个 target 的 profile ════════"
for t in march7 embed intent voiceagent; do
  printf '  %-12s ' "$t"
  out=$(CI_TARGET=$t bash scripts/ci/target.sh --print 2>&1)
  if [ $? -ne 0 ]; then echo "❌ 解析失败"; echo "$out" | head -3; continue; fi
  d=$(echo "$out" | grep '^DEPLOY' | sed 's/.*= *//')
  v=$(echo "$out" | grep '^VERIFY_PY' | sed 's/.*= *//')
  b=$(echo "$out" | grep '^BENCH_PY' | sed 's/.*= *//')
  echo "DEPLOY=$d  VERIFY=$v  BENCH=$b"
done
echo
echo "  未知 target 应该报错："
printf '  %-12s ' "bogus"
CI_TARGET=bogus bash scripts/ci/target.sh --print >/dev/null 2>&1 && echo "❌ 居然通过了" || echo "✅ 正确报错"
echo
echo "════════ ② 旧 3 个 target 的 profile 没变（对照提交前）════════"
cat <<'EOF'
  march7      DEPLOY=vllm-march7     VERIFY=verify-march7.py  BENCH=bench.py         UNIT=tok/s  MIN=100
  embed       DEPLOY=vllm-embed      VERIFY=verify-embed.py   BENCH=bench-embed.py   UNIT=req/s  MIN=20
  intent      DEPLOY=merged-intent   VERIFY=verify-intent.py  BENCH=bench-intent.py  UNIT=req/s  MIN=5
  voiceagent  DEPLOY=(空)            VERIFY=verify-voiceagent.py  BENCH=(空)
EOF
echo
echo "════════ ③ 现有 k8s 部署没被碰 ════════"
export KUBECONFIG=/home/user/.kube/config
kubectl -n default get deploy -o custom-columns=NAME:.metadata.name,READY:.status.readyReplicas,GEN:.metadata.generation 2>&1
echo
echo "════════ ④ voiceagent 校验脚本 ════════"
python3 scripts/ci/verify-voiceagent.py 2>&1 | tail -6
