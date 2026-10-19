import time
from pathlib import Path
CANDS = [r'E:\models\qwen3-asr-1.7b-official', r'E:\models\qwen3-asr-0.6b-official']
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
for m in CANDS:
    p = Path(m)
    if not p.exists():
        continue
    has = [f.name for f in p.iterdir() if f.is_file()]
    print('try', m, '| files:', len(has))
    try:
        t0 = time.time()
        proc = AutoProcessor.from_pretrained(m, trust_remote_code=True)
        model = AutoModelForMultimodalLM.from_pretrained(m, dtype='auto',
                                                         device_map='cuda',
                                                         trust_remote_code=True)
        model.eval()
        print('  OK load %.1fs dtype=%s vram=%.2fGB' % (time.time()-t0,
              next(model.parameters()).dtype, torch.cuda.memory_allocated()/1e9))
        utts = sorted(Path(r'C:\Users\user\Desktop\ds harness\voice_agent\utt').glob('*mic*.wav'))[-5:]
        for f in utts:
            t0 = time.time()
            inp = proc.apply_transcription_request(audio=str(f), language='zh').to(model.device, model.dtype)
            ids = model.generate(**inp, max_new_tokens=128)
            txt = str(proc.decode(ids[:, inp['input_ids'].shape[1]:],
                                  return_format='transcription_only')[0]).strip()
            print('  %5.0fms  %-32s %r' % ((time.time()-t0)*1000, f.name[:32], txt))
        print('peak vram %.2fGB' % (torch.cuda.max_memory_allocated()/1e9))
        break
    except Exception as e:
        print('  FAIL:', type(e).__name__, str(e)[:200])
