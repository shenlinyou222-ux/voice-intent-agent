import ctypes, json
from pathlib import Path
import llama_cpp
MODEL = r"C:\Users\user\Desktop\ds harness\aigc_demo_framework\models\qwen3-asr-gguf-1.7b\Qwen3-ASR-1.7B-Q4_K_M.gguf"
T2S = "開關閉網頁個這為麼說話東學習記裏檢刪確應該價們對時間實現於與從後讓幾導觀點線終語輸啟動執資圖書檔電腦統設選擇項視訊軟體務編譯碼審螢鍵盤標頻廣樂藝術國際區轉寫"
trad = set(T2S)
llm = llama_cpp.Llama(model_path=MODEL, vocab_only=True, verbose=False)
n = llama_cpp.llama_cpp.llama_vocab_n_tokens(llama_cpp.llama_cpp.llama_model_get_vocab(llm.model))
hits = []
for tid in range(n):
    try:
        s = llm.detokenize([tid]).decode("utf-8", "ignore")
    except Exception:
        continue
    if s and any(ch in trad for ch in s):
        hits.append(tid)
print("  词表大小 =", n)
print("  含繁体字符的 token 数 =", len(hits))
print("  前 20 个 id:", hits[:20])
Path("trad_token_ids.json").write_text(json.dumps(hits), encoding="utf-8")
print("  已存 trad_token_ids.json")
llm.close()
