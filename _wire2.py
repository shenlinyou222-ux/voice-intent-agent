import json, re
from pathlib import Path
root = Path(r"C:\Users\user\Desktop\ds harness\voice_agent")
v = root / "voice_agent.py"; t = v.read_text(encoding="utf-8")

def stmt_end(s, start):
    """从 start 处的 '(' 起跟踪括号深度，返回语句结束（换行）位置——避免插进表达式中间。"""
    i = s.index("(", start); depth = 0
    while i < len(s):
        if s[i] == "(": depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                j = s.index("\n", i)
                return j
        i += 1
    raise RuntimeError("找不到语句结尾")

m = re.search(r"self\.model = IntentModel\(", t)
assert m, "找不到 IntentModel 构造"
j = stmt_end(t, m.start())
block = '''
        # ⭐ 意图分类后端：embed = 句向量 kNN（零训练：加意图只加示例+重建索引）
        #   | torch = 原词袋线性头（词嵌入+8x768）。见 intent_embed.py
        self.embed_cls = None
        if str(self.cfg.behavior.get("intent_backend", "torch")).lower() == "embed":
            try:
                from intent_embed import EmbedIntent
                from pathlib import Path as _P
                _idx = _P(__file__).parent / str(self.cfg.behavior.get(
                    "intent_index", "intent_index.json"))
                self.embed_cls = EmbedIntent(
                    url=str(self.cfg.behavior.get("intent_embed_url")
                            or "http://172.20.71.101:30802/v1/embeddings"),
                    index=str(_idx),
                    thr=float(self.cfg.behavior.get("intent_embed_thr") or 0.30),
                    margin=float(self.cfg.behavior.get("intent_embed_margin") or 0.0))
                log(f"   意图分类：句向量 kNN（{len(self.embed_cls.labels)} 示例 / "
                    f"{len(self.embed_cls.intents)} 类 / thr={self.embed_cls.thr}）")
            except Exception as _e:
                log(f"   ⚠️ 句向量分类器加载失败（{type(_e).__name__}: {_e}）"
                    f"→ 回退词袋线性头", "WARN")
                self.embed_cls = None'''
t = t[:j] + block + t[j:]

anchor = "    def handle(self, wav):"
meth = '''    def _predict_intent(self, *args, **kwargs):
        """意图判定统一入口：embed(句向量 kNN) 优先，异常/关闭则回退 torch 词袋。"""
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
assert anchor in t
t = t.replace(anchor, meth, 1)
n1 = len(re.findall(r"self\.model\.predict_text\(", t)); n2 = len(re.findall(r"self\.model\.predict\(", t))
t = t.replace("self.model.predict_text(", "self._predict_intent(").replace("self.model.predict(", "self._predict_intent(")
v.write_text(t, encoding="utf-8")
print(f"  插入位置: 字符 {j}（语句结尾）| 替换调用: {n1}+{n2} 处")
