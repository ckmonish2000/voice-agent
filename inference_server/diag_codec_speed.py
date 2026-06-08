"""
diag_codec_speed.py — is the codec's ~12s cost a one-time warm-up, or every call?

Confirmed so far: no hang. The codec RETURNS, but takes ~12 s per call, with or
without the kernel. 12 s on a 5090 for ~100 frames is abnormal. We must know if
the SECOND+ calls are fast (warm-up only) or still slow (real per-call cost),
because that decides the fix.

This calls speech_tokenizer.decode() FIVE times on the same input and prints the
time for each. No kernel, no generate — just the codec, repeatedly.

  call 1 slow, calls 2-5 fast  -> it's warm-up. Fix: warm the codec once at
                                  startup, then streaming is cheap.
  all 5 slow (~12s each)       -> real per-call cost. Fix: decode ONCE at the end
                                  (or in big non-overlapping chunks), not per-hop.

Also tries different sizes to see if time scales with frame count.

Run on the box (server stopped):
  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=0 \
  python diag_codec_speed.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402
from engine import StreamingTTSEngine  # noqa: E402

eng = StreamingTTSEngine()
print(f"[diag] engine ready. device={eng.device}", flush=True)

ref = eng._voice["ref_code"].to(eng.device)
tok = eng.model.speech_tokenizer


def timed(label, codes):
    t0 = time.perf_counter()
    wavs, _sr = tok.decode([{"audio_codes": codes}])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    n = codes.shape[0]
    print(f"  [{label}] frames={n:3d}  ->  {dt:8.0f} ms", flush=True)
    return dt


print("\n[A] same input, 5 times in a row (warm-up vs per-call test):", flush=True)
dummy = torch.zeros((12, 16), dtype=ref.dtype, device=ref.device)
codes = torch.cat([ref, dummy], dim=0)
for i in range(5):
    timed(f"call {i+1}", codes)

print("\n[B] does time scale with frame count? (after warm-up):", flush=True)
for n in (4, 12, 30, 60):
    d = torch.zeros((n, 16), dtype=ref.dtype, device=ref.device)
    timed(f"ref+{n}", torch.cat([ref, d], dim=0))

print("\n[C] tiny input WITHOUT the big ref prefix (just a few frames):",
      flush=True)
for n in (4, 12, 30):
    d = torch.zeros((n, 16), dtype=ref.dtype, device=ref.device)
    timed(f"{n}-only", d)

print("\n[diag] done. Read [A]: if call 1 slow and 2-5 fast = warm-up (easy fix).",
      flush=True)
print("Read [C]: if small inputs are fast, the cost is the big ref prefix.",
      flush=True)
