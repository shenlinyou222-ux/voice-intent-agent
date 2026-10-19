import json, re
from pathlib import Path
root = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
v = root / "voice_agent.py"; t = v.read_text(encoding="utf-8")
orig = t

# ① 在意图模型构造处，加入句向量分类器（开关 intent_backend）
hits = [m.start() for m in re.finditer(r"self\.model = IntentModel\(", t)]
print("  IntentModel 构造点:", len(hits))
if hits and "embed_cls" not in t:
    # 找到该行结尾
    i = hits[0]
    j = t.index("\n", t.index(")", t.index("IntentModel(", i)))
    ins = '''
        # ⭐ 意图分类后端：embed = 句向量 kNN（零训练、加意图只加示例）| torch = 原词袋线性头
        self.embed_cls = None
        if str(self.cfg.behavior.get("intent_backend", "torch")).lower() == "embed":
            try:
                from intent_embed import EmbedIntent
                _idx = root_dir if False else None
            except Exception:
                pass'''
    # 用更简单可靠的方式：直接在 self.model 构造后插入（不依赖上面的字符串）
    block = ('\n        # ⭐ 意图分类后端：embed = 句向量 kNN（零训练、加意图只加示例）'
             ' | torch = 原词袋线性头\n'
             '        self.embed_cls = None\n'
             '        if str(self.cfg.behavior.get("intent_backend", "torch")).lower() == "embed":\n'
             '            try:\n'
             '                from intent_embed import EmbedIntent\n'
             '                from pathlib import Path as _P\n'
             '                _idx = _P(__file__).parent / str(self.cfg.behavior.get(\n'
             '                    "intent_index", "intent_index.json"))\n'
             '                self.embed_cls = EmbedIntent(\n'
             '                    url=str(self.cfg.behavior.get("intent_embed_url")\n'
             '                            or "http://172.20.71.101:30802/v1/embeddings"),\n'
             '                    index=str(_idx),\n'
             '                    thr=float(self.cfg.behavior.get("intent_embed_thr") or 0.30),\n'
             '                    margin=float(self.cfg.behavior.get("intent_embed_margin") or 0.0))\n'
             '                log(f"   意图分类：句向量 kNN（{len(self.embed_cls.labels)} 示例 "\n'
             '                    f"/ {len(self.embed_cls.intents)} 类 / thr={self.embed_cls.thr}）")\n'
             '            except Exception as _e:\n'
             '                log(f"   ⚠️ 句向量分类器加载失败（{type(_e).__name__}: {_e}）"\n'
             '                    f"→ 回退词袋线性头", "WARN")\n'
             '                self.embed_cls = None\n')
    t = t[:j] + t[j:] if False else (t[:j] + block + t[j:])

# ② 统一预测入口 _predict_intent
if "_predict_intent" not in t:
    anchor = "    def handle(self, wav):"
    meth = '''    def _predict_intent(self, *args, **kwargs):
        """意图判定统一入口：embed(句向量 kNN) 优先，失败/关闭则 torch 词袋线性头。"""
        text = kwargs.get("text_override")
        if text is None:
            for a in args:
                if isinstance(a, str):
                    text = a
                    break
        if getattr(self, "embed_cls", None) is not None and text:
            try:
                r = self.embed_cls.predict(text)
                return {"text": text, "intent": r["intent"], "conf": float(r["conf"]),
                        "latency_ms": r["ms"], "margin": r["margin"], "src": "embed"}
            except Exception as _e:
                log(f"   ⚠️ 句向量分类失败（{type(_e).__name__}: {_e}）→ 回退词袋", "WARN")
        out = self.model.predict(*args, **kwargs)
        if isinstance(out, dict):
            out.setdefault("src", "torch")
        return out

''' + anchor
    t = t.replace(anchor, meth, 1)

n1 = len(re.findall(r"self\.model\.predict_text\(", t))
n2 = len(re.findall(r"self\.model\.predict\(", t))
t = t.replace("self.model.predict_text(", "self._predict_intent(")
t = t.replace("self.model.predict(", "self._predict_intent(")
v.write_text(t, encoding="utf-8")
print(f"  替换预测调用：predict_text {n1} 处 / predict {n2} 处")
print("  改动:", t != orig)

# ③ 配置：开启 embed 后端 + 砍掉 web_search
p = root / "intent_config.json"; d = json.loads(p.read_text(encoding="utf-8")); b = d["behavior"]
b["intent_backend"] = "embed"
b["intent_index"] = "intent_index.json"
b["intent_embed_url"] = "http://172.20.71.101:30802/v1/embeddings"
b["intent_embed_thr"] = 0.30
b["intent_embed_margin"] = 0.0
b["intent_drop"] = ["web_search"]      # ⭐ 用户决定：这类意图过时，砍掉
for k in ("web_search",):
    if k in d.get("triggers", {}):
        print("  删除 triggers.%s = %s" % (k, d["triggers"].pop(k)))
b["_comment_intent_backend"] = ("意图分类换成句向量 kNN（semantic-router 等价物，零训练）："
    "加意图=加示例+重建索引；unknown 用阈值(0.30)判定；web_search 已按用户要求砍掉。"
    "回退：intent_backend=torch。")
p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
print("  intent_backend =", b["intent_backend"], "| intent_drop =", b["intent_drop"])
