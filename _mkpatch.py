import json
from pathlib import Path
H = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
ids = json.loads((H / "trad_token_ids.json").read_text(encoding="utf-8"))
ops = []
for i in ids:
    ops.append({"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--logit-bias"})
    ops.append({"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "%d-inf" % i})
(H / "logitbias_patch.json").write_text(json.dumps(ops), encoding="utf-8")
print("  补丁: %d 条 op（%d 个 token 屏蔽）" % (len(ops), len(ids)))
