#!/usr/bin/env bash
export KUBECONFIG=/home/user/.kube/config
IP=$(python3 -c "
import socket
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('10.255.255.255',1))
print(s.getsockname()[0]); s.close()")
BASE="http://$IP:30080"; JOB=hasitada-deploy
PW=$(kubectl -n cicd get secret jenkins-auth -o jsonpath='{.data.admin-password}' | base64 -d)

echo "=== ① 等没有构建在跑 ==="
for i in $(seq 1 60); do
  B=$(curl -s -m 15 -u "ops:$PW" "$BASE/job/$JOB/api/json?tree=lastBuild%5Bbuilding%5D" \
      | python3 -c "import json,sys;print((json.load(sys.stdin).get('lastBuild') or {}).get('building'))")
  [ "$B" = "False" ] && { echo "  空闲 ✅"; break; }
  echo "  try $i building=$B"; sleep 8
done

echo
echo "=== ② 这个 job 现在认哪些参数 ==="
curl -s -m 20 -u "ops:$PW" "$BASE/job/$JOB/api/json?tree=property%5BparameterDefinitions%5Bname,type,choices%5D%5D" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
for p in d.get('property',[]):
    for pd in (p.get('parameterDefinitions') or []):
        print('   %-12s %-16s %s' % (pd['name'], pd['type'], pd.get('choices','')))
"

echo
echo "=== ③ 触发（带错误体）==="
curl -s -c /tmp/cjW -m 20 -u "ops:$PW" "$BASE/crumbIssuer/api/json" > /tmp/crW.json
CRUMB=$(python3 -c "import json;print(json.load(open('/tmp/crW.json'))['crumb'])")
PREV=$(curl -s -m 20 -u "ops:$PW" "$BASE/job/$JOB/api/json?tree=lastBuild%5Bnumber%5D" \
       | python3 -c "import json,sys;print((json.load(sys.stdin).get('lastBuild') or {}).get('number',0))")
echo "  PREV=#$PREV"
code=$(curl -s -b /tmp/cjW -m 40 -o /tmp/trW.out -w '%{http_code}' -X POST \
  -u "ops:$PW" -H "Jenkins-Crumb: $CRUMB" \
  "$BASE/job/$JOB/buildWithParameters?TARGET=voiceagent&MODE=full")
echo "  HTTP $code"
if [ "$code" != "201" ]; then
  echo "  ---- 错误体（前 40 行）----"
  head -40 /tmp/trW.out | sed 's/^/    /'
fi

echo
echo "=== ④ 等构建 ==="
NUM=""
for i in $(seq 1 120); do
  sleep 8
  curl -s -m 20 -u "ops:$PW" "$BASE/job/$JOB/api/json?tree=lastBuild%5Bnumber,building,result%5D" > /tmp/lbW.json
  read N B R <<< "$(python3 -c "
import json
d=json.load(open('/tmp/lbW.json')).get('lastBuild') or {}
print(d.get('number'), d.get('building'), d.get('result'))")"
  if [ "$N" != "$PREV" ]; then
    NUM=$N; echo "  #$N building=$B result=$R"
    [ "$B" = "False" ] && break
  fi
done
[ -z "$NUM" ] && { echo "  ❌ 没等到"; exit 1; }

echo
echo "=== ⑤ #$NUM 结果 ==="
curl -s -m 20 -u "ops:$PW" "$BASE/job/$JOB/$NUM/api/json?tree=number,result,duration,actions%5Bparameters%5Bname,value%5D%5D" > /tmp/bnW.json
python3 - <<'PY'
import json
d=json.load(open('/tmp/bnW.json'))
ps={}
for a in d.get('actions',[]):
    for p in (a.get('parameters') or []): ps[p['name']]=p.get('value')
print('    #%s result=%s duration=%.1fs params=%s' % (d['number'], d.get('result'), (d.get('duration') or 0)/1000, ps))
PY

echo
echo "=== ⑥ 关键输出 ==="
curl -s -m 30 -u "ops:$PW" "$BASE/job/$JOB/$NUM/consoleText" > /tmp/cW.txt
grep -nE '语音 Agent 校验|全部判据通过|校验失败|规则分句|可执行命令清单|启动程序|打开网址|打开文件|参数化|Finished:' /tmp/cW.txt | sed 's/^/    /' | head -30
