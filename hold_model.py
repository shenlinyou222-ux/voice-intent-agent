# -*- coding: utf-8 -*-
"""子进程用：加载指定产物后驻留，供外部采样内存。

    python hold_model.py text|full [秒数]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model_runner import IntentModel  # noqa: E402

mode = (sys.argv[1] if len(sys.argv) > 1 else "text").lower()
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
HERE = Path(__file__).parent

if mode == "full":
    m = IntentModel(model_dir=str(HERE / "model"), device="cpu", int8=True,
                    text_only=False, verbose=False)
    print("LOADED full", flush=True)
else:
    m = IntentModel(model_dir=str(HERE / "model"), device="cpu", int8=True,
                    text_only=True, verbose=False)
    print("LOADED text", flush=True)
time.sleep(secs)
