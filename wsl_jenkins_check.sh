#!/usr/bin/env bash
export KUBECONFIG=/home/user/.kube/config
IP=$(python3 -c "
import socket
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('10.255.255.255',1))
print(s.getsockname()[0]); s.close()")
BASE="http://$IP:30080"; JOB=hasitada-deploy
PW=$(kubectl -n cicd get secret jenkins-auth -o jsonpath='{.data.admin-password}' | base64 -d)

echo "=== 最近 5 次构建 ==="
curl -s -m 25 -u "ops:$PW" "$BASE/job/$JOB/api/json?tree=builds%5Bnumber,result,duration,actions%5Bparameters%5Bname,value%5D%5D%5D%7B0,5%7D" > /tmp/bl.json
python3 - <<'PY'
import json
d = json.load(open('/tmp/bl.json'))
for b in d.get('builds', [])[:5]:
    ps = {}
    for a in b.get('actions', []):
        for p in (a.get('parameters') or []):
            ps[p['name']] = p.get('value')
    print("  #%-4s %-8s %6.1fs  TARGET=%-12s MODE=%s" % (
        b['number'], b.get('result'), (b.get('duration') or 0)/1000,
        ps.get('TARGET'), ps.get('MODE')))
PY

echo
echo "=== 最新一次构建的关键输出 ==="
N=$(python3 -c "import json;d=json.load(open('/tmp/bl.json'));print(d['builds'][0]['number'])")
curl -s -m 30 -u "ops:$PW" "$BASE/job/$JOB/$N/consoleText" > /tmp/cn.txt
grep -nE '语音 Agent 校验|规则分句|全部判据通过|校验失败|可执行命令清单|Finished:|skipped due to' /tmp/cn.txt | sed 's/^/  /' | head -25

echo
echo "=== 集群状态（确认现有部署没被碰）==="
kubectl -n default get deploy -o custom-columns=NAME:.metadata.name,READY:.status.readyReplicas,GEN:.metadata.generation 2>&1 | sed 's/^/  /'
