import time
from pathlib import Path
MODEL = r'E:\models\qwen3-asr-1.7b-official'
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
print('torch', torch.__version__, '| cuda available =', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu =', torch.cuda.get_device_name(0),
          '| free %.1f GB' % (torch.cuda.mem_get_info()[0]/1e9))
t0 = time.time()
proc = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype='auto', device_map='cuda')
model.eval()
dt = next(model.parameters()).dtype
print('load %.1fs | dtype %s | vram %.2f GB' % (time.time()-t0, dt, torch.cuda.memory_allocated()/1e9))
utts = sorted(Path(r'C:\Users\user\Desktop\ds harness\voice_agent\utt').glob('*mic*.wav'))[-5:]
for p in utts:
    t0 = time.time()
    inputs = proc.apply_transcription_request(audio=str(p), language='zh').to(model.device, model.dtype)
    ids = model.generate(**inputs, max_new_tokens=128)
    txt = str(proc.decode(ids[:, inputs['input_ids'].shape[1]:],
                          return_format='transcription_only')[0]).strip()
    print('  %5.0fms  %-34s %r' % ((time.time()-t0)*1000, p.name[:34], txt))
print('peak vram %.2f GB' % (torch.cuda.max_memory_allocated()/1e9))
