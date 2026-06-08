"""
diag_codec_dtype.py — is the codec slow because it runs in float32, and does
casting it to bfloat16 (or running the conv stack differently) make it fast?

GPU is healthy (222 TFLOP/s matmul), attention is already 'sdpa', the conv code
has no Python time-loops. So the 12s is likely the DTYPE / kernel selection of the
conv-heavy decoder. This probe:

  1. prints the dtype of the codec's weights and of the upsample/decoder convs.
  2. times one decode AS-IS.
  3. times one decode with the codec cast to bfloat16.
  4. times one decode with the codec cast to float16.
  5. tries torch.backends tweaks (tf32) and re-times float32.

Whatever is fastest is the fix we apply in common.py / engine.py.

Run on the box (server stopped):
  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=0 \
  python diag_codec_dtype.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # inference_server/ (parent)

import torch  # noqa: E402
from engine import StreamingTTSEngine  # noqa: E402

eng = StreamingTTSEngine()
print(f"[diag] engine ready. device={eng.device}", flush=True)

tok = eng.model.speech_tokenizer
inner = tok.model
decoder = inner.decoder

# --- 1. report dtypes ---
def dtype_of(mod):
    for p in mod.parameters():
        return p.dtype
    return None

print(f"[dtype] whole codec (tok.model)      : {dtype_of(inner)}", flush=True)
print(f"[dtype] decoder.pre_transformer       : {dtype_of(decoder.pre_transformer)}",
      flush=True)
print(f"[dtype] decoder.upsample              : {dtype_of(decoder.upsample)}",
      flush=True)
print(f"[dtype] decoder.decoder (conv stack)  : {dtype_of(decoder.decoder)}",
      flush=True)

ref = eng._voice["ref_code"].to(eng.device)
dummy = torch.zeros((12, 16), dtype=ref.dtype, device=ref.device)
codes = torch.cat([ref, dummy], dim=0)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(label, n=2):
    # warm once, then average n
    tok.decode([{"audio_codes": codes}]); sync()
    t0 = time.perf_counter()
    for _ in range(n):
        tok.decode([{"audio_codes": codes}])
    sync()
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"  [{label}] {dt:8.0f} ms/decode", flush=True)
    return dt


print("\n[2] as-is:", flush=True)
timed("as-is")

print("\n[5] float32 + TF32 enabled (allow_tf32):", flush=True)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
timed("tf32+cudnn.benchmark")

print("\n[3] codec cast to bfloat16:", flush=True)
try:
    inner.to(torch.bfloat16)
    timed("bf16")
except Exception as e:
    print(f"  bf16 failed: {e!r}", flush=True)

print("\n[4] codec cast to float16:", flush=True)
try:
    inner.to(torch.float16)
    timed("fp16")
except Exception as e:
    print(f"  fp16 failed: {e!r}", flush=True)

# restore
inner.to(torch.float32)

print("\n[diag] done. Lowest ms/decode wins. If bf16/fp16 are much faster, the",
      flush=True)
print("       codec was slow in float32 and we load it in low precision.", flush=True)
print("       If tf32 alone helps a lot, just enable the tf32 flags at startup.",
      flush=True)
