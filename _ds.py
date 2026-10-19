import json
from pathlib import Path
root = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
ds = root / "intent_dataset.jsonl"
rows = [json.loads(l) for l in ds.read_text(encoding="utf-8").splitlines() if l.strip()]
print("  现有 id 类型示例:", repr(rows[0].get("id")))
have = {r["intent"] for r in rows}
EX = {
 "dictation_on": ["开始输入", "开始听写", "进入输入模式", "帮我开始输入", "我要开始打字",
                  "开始语音输入", "打开听写", "现在开始输入文字"],
 "dictation_off": ["终止输入", "停止输入", "结束输入", "停止听写", "别输入了",
                   "退出输入模式", "不要打字了", "停一下输入"],
}
nums = [int(r["id"]) for r in rows if str(r.get("id", "")).isdigit()]
mx = max(nums) if nums else 0
add = 0
for it, texts in EX.items():
    if it in have:
        continue
    for txt in texts:
        mx += 1
        rows.append({"id": str(mx), "text": txt, "intent": it, "kind": "rule",
                     "split": "train", "arg": ""})
        add += 1
ds.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
print("  ✅ 语料 +%d 条（dictation_on/off）；总 %d 条，意图 %s" % (add, len(rows), sorted({r["intent"] for r in rows})))
