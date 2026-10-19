#!/usr/bin/env bash
export KUBECONFIG=/home/user/.kube/config
IP=$(python3 -c "
import socket
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('10.255.255.255',1))
print(s.getsockname()[0]); s.close()")
BASE="http://$IP:30080"
JOB=hasitada-deploy
PW=$(kubectl -n cicd get secret jenkins-auth -o jsonpath='{.data.admin-password}' | base64 -d)

echo "=== Jenkins 就绪? ==="
kubectl -n cicd get pods -l app=jenkins
for i in $(seq 1 20); do
  code=$(curl -s -m 8 -o /dev/null -w '%{http_code}' "$BASE/login" 2>/dev/null)
  [ "$code" = "200" ] && { echo "  HTTP 200 ✅"; break; }
  echo "  try $i: $code"; sleep 8
done

echo
echo "=== 触发 TARGET=voiceagent ==="
curl -s -c /tmp/cjV -m 20 -u "ops:$PW" "$BASE/crumbIssuer/api/json" > /tmp/crV.json
CRUMB=$(python3 -c "import json;print(json.load(open('/tmp/crV.json'))['crumb'])")
PREV=$(curl -s -m 20 -u "ops:$PW" "$BASE/job/$JOB/api/json?tree=lastBuild%5Bnumber%5D" \
       | python3 -c "import json,sys;d=json.load(sys.stdin);print((d.get('lastBuild') or {}).get('number',0))")
echo "  触发前 lastBuild=#$PREV"
code=$(curl -s -b /tmp/cjV -m 30 -o /tmp/trV.out -w '%{http_code}' -X POST \
  -u "ops:$PW" -H "Jenkins-Crumb: $CRUMB" \
  "$BASE/job/$JOB/buildWithParameters?TARGET=voiceagent&MODE=full")
echo "  HTTP $code"

echo
echo "=== 等构建 ==="
NUM=""
for i in $(seq 1 100); do
  sleep 8
  curl -s -m 20 -u "ops:$PW" "$BASE/job/$JOB/api/json?tree=lastBuild%5Bnumber,building,result%5D" > /tmp/lbV.json
  read N B R <<< "$(python3 -c "
import json
d=json.load(open('/tmp/lbV.json')).get('lastBuild') or {}
print(d.get('number'), d.get('building'), d.get('result'))")"
  if [ "$N" != "$PREV" ]; then
    NUM=$N
    echo "  #$N building=$B result=$R"
    [ "$B" = "False" ] && break
  fi
done

[ -z "$NUM" ] && { echo "  ❌ 没等到新构建"; exit 1; }

echo
echo "=== 构建 #$NUM 结果 ==="
curl -s -m 20 -u "ops:$PW" "$BASE/job/$JOB/$NUM/api/json?tree=number,result,duration,actions%5Bparameters%5Bname,value%5D%5D" > /tmp/bnV.json
python3 - <<'PY'
import json
d=json.load(open('/tmp/bnV.json'))
ps={}
for a in d.get('actions',[]):
    for p in (a.get('parameters') or []): ps[p['name']]=p.get('value')
print('    #%s result=%s duration=%.1fs params=%s' % (d['number'], d.get('result'), (d.get('duration') or 0)/1000, ps))
PY

echo
echo "=== 关键阶段 ==="
curl -s -m 30 -u "ops:$PW" "$BASE/job/$JOB/$NUM/consoleText" > /tmp/cV.txt
grep -nE 'Stage "|语音 Agent 校验|全部判据通过|校验失败|可执行命令清单|启动程序|打开网址|Finished:|流水线' /tmp/cV.txt | sed 's/^/    /' | head -40
