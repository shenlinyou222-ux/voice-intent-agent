import inspect, traceback, time
from pathlib import Path
import sherpa_onnx
SV = Path(r"E:\models\sensevoice-onnx\sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
print("  sherpa_onnx", sherpa_onnx.__version__)
print("  from_sense_voice 签名:", str(inspect.signature(sherpa_onnx.OfflineRecognizer.from_sense_voice))[:300])
print("  CUDA 可用标志:", getattr(sherpa_onnx, "is_gpu_available", None) and sherpa_onnx.is_gpu_available() or "n/a")
# 先 CPU 建，确保 API 用对
for provider, tag in (("cpu", "CPU"), ("cuda", "CUDA")):
    try:
        rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(SV / "model.int8.onnx"), tokens=str(SV / "tokens.txt"),
            num_threads=4, use_itn=True, language="zh", provider=provider, debug=False)
        print(f"  [{tag}] recognizer 建好了 ✓")
        f = sorted(Path("utt").glob("*mic*dictation*.wav"))[-1]
        t0 = time.time()
        s = rec.create_stream()
        s.accept_wave_file(str(f))
        rec.decode_stream(s)
        print(f"  [{tag}] {f.name[:34]} -> {(s.result.text or '').strip()[:44]}  ({(time.time()-t0)*1000:.0f}ms)")
        break
    except Exception as e:
        print(f"  [{tag}] 失败: {type(e).__name__}: {str(e)[:120]}")
        traceback.print_exc(limit=2)
