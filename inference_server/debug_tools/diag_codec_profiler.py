"""
diag_codec_profiler.py — DEFINITIVE: is the codec time spent on the GPU computing,
or on the CPU launching tiny ops? Uses torch.profiler (ground truth).

Ruled out so far: threads, warm-up, kernel, attention impl, GPU throttle, dtype.
A bf16 8-layer transformer on a 222 TFLOP/s GPU CANNOT take 5s to compute 100
tokens. So the time is almost certainly NOT compute — it's CPU-side overhead
(thousands of tiny kernel launches, or a sync/copy stall, or a CPU fallback op).

torch.profiler records every CPU op and every CUDA kernel with real durations.
We print:
  - top ops by total CPU time   (if a CPU op dominates -> CPU-bound / fallback)
  - top ops by total CUDA time  (if CUDA time is tiny -> GPU is idle, CPU is the wall)
  - total CUDA time vs wall time (the gap is CPU overhead / stalls)

Run on the box (server stopped):
  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=0 \
  python diag_codec_profiler.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # inference_server/ (parent)

import torch  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402
from engine import StreamingTTSEngine  # noqa: E402

eng = StreamingTTSEngine()
print(f"[diag] engine ready. device={eng.device}", flush=True)

tok = eng.model.speech_tokenizer
ref = eng._voice["ref_code"].to(eng.device)
dummy = torch.zeros((12, 16), dtype=ref.dtype, device=ref.device)
codes = torch.cat([ref, dummy], dim=0)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# warm up
tok.decode([{"audio_codes": codes}]); sync()

# wall time
t0 = time.perf_counter()
tok.decode([{"audio_codes": codes}]); sync()
wall_ms = (time.perf_counter() - t0) * 1000
print(f"\n[wall] one decode = {wall_ms:.0f} ms\n", flush=True)

# count kernel launches + profile
print("[profiler] recording one decode...\n", flush=True)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=False) as prof:
    tok.decode([{"audio_codes": codes}])
    sync()

ka = prof.key_averages()

print("===== TOP 15 OPS BY CPU TIME =====", flush=True)
print(ka.table(sort_by="cpu_time_total", row_limit=15), flush=True)

print("\n===== TOP 15 OPS BY CUDA TIME =====", flush=True)
try:
    print(ka.table(sort_by="cuda_time_total", row_limit=15), flush=True)
except Exception as e:
    print(f"(cuda sort failed: {e!r})", flush=True)

# totals
total_cpu = sum(getattr(e, "cpu_time_total", 0) for e in ka) / 1000.0  # ms
total_cuda = 0.0
n_launches = 0
for e in ka:
    cu = getattr(e, "cuda_time_total", 0) or getattr(e, "device_time_total", 0) or 0
    total_cuda += cu
    n_launches += getattr(e, "count", 0)
total_cuda /= 1000.0

print("\n========== SUMMARY ==========", flush=True)
print(f"  wall time            : {wall_ms:8.0f} ms", flush=True)
print(f"  total CUDA (GPU) time : {total_cuda:8.0f} ms", flush=True)
print(f"  total op invocations  : {n_launches}", flush=True)
print("  ---", flush=True)
print("  If CUDA time is a SMALL fraction of wall time -> CPU-bound (op launch", flush=True)
print("  overhead or a CPU fallback op). The fix is to reduce op count / avoid", flush=True)
print("  the fallback, NOT to speed up the GPU. Look at the TOP CPU OPS above.", flush=True)
print("=============================", flush=True)
