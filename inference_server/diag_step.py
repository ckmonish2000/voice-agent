"""
diag_step.py — where does each streaming STEP spend its time?

Steady-state bench showed: codec is now fast (~28 ms), but the streaming path
runs the backbone at ~4 steps/s (kernel can do 1286/s offline) and ~937 ms per
chunk. So ~200 ms/step is being lost somewhere in the live generate loop. Find
where.

We time, per talker.forward step:
  - the kernel backbone post-hook (KV seed + decode_from_hidden + norm)
  - the codec decode inside the frame hook
  - everything else in the step (generate's own forward: code_predictor, sampling)

by wrapping the relevant callables with timers. Prints a per-step table for the
first ~12 steps and an average.

Run on the box:
  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=1 \
  python diag_step.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402
from engine import StreamingTTSEngine, StreamConfig  # noqa: E402

eng = StreamingTTSEngine()
print(f"[diag] use_kernel={eng._use_kernel} device={eng.device}", flush=True)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# NOTE: we deliberately do NOT wrap talker.forward — it sits behind a
# @check_model_inputs decorator that strips extra kwargs; wrapping it bypasses
# that and raises "model_kwargs not used". Instead we measure the whole step via
# the frame hook timestamps (eng.last_metrics.frame_arrival_ts) and time only the
# inner pieces that have no such decorator: talker.model.forward and tok.decode.
step_times = []  # filled from metrics after the run (inter-frame wall time)

# --- time the codec decode (inside the frame hook) ---
tok = eng.model.speech_tokenizer
orig_codec = tok.decode
codec_times = []


def timed_codec(*a, **k):
    sync(); t0 = time.perf_counter()
    out = orig_codec(*a, **k)
    sync()
    codec_times.append((time.perf_counter() - t0) * 1000)
    return out


tok.decode = timed_codec

# --- time the talker.model backbone (where the kernel hook runs) ---
talker_model = eng.model.talker.model
orig_tm_fwd = talker_model.forward
backbone_times = []


def timed_backbone(*a, **k):
    sync(); t0 = time.perf_counter()
    out = orig_tm_fwd(*a, **k)
    sync()
    backbone_times.append((time.perf_counter() - t0) * 1000)
    return out


talker_model.forward = timed_backbone

print("[diag] warming up...", flush=True)
for _ in eng.decode_stream("warm up please", StreamConfig(max_new_tokens=24)):
    pass

codec_times.clear(); backbone_times.clear()
print("[diag] measured run...\n", flush=True)

n = 0
for _ in eng.decode_stream(
        "The quick brown fox jumps over the lazy dog and runs home.",
        StreamConfig(max_new_tokens=512)):
    n += 1

# whole-step wall time from frame arrival timestamps (gap between frames)
ts = eng.last_metrics.frame_arrival_ts
step_times = [(ts[i+1] - ts[i]) * 1000 for i in range(len(ts) - 1)]

print("========== PER-STEP TIMING ==========", flush=True)


def stats(name, arr):
    if not arr:
        print(f"  {name:20s}: (none)", flush=True)
        return
    arr2 = arr[1:] if len(arr) > 1 else arr   # drop first (warm/prefill)
    avg = sum(arr2)/len(arr2)
    print(f"  {name:20s}: n={len(arr):3d}  avg={avg:7.1f} ms  "
          f"first={arr[0]:7.1f}  min={min(arr2):7.1f}  max={max(arr2):7.1f}",
          flush=True)


stats("whole step (frame gap)", step_times)
stats("  backbone (kernel)", backbone_times)
stats("  codec decode", codec_times)
print("  ---", flush=True)
print(f"  chunks streamed     : {n}", flush=True)
print("  Interpretation: talker.forward/step is the WHOLE per-step cost.", flush=True)
print("  backbone = the kernel path. codec = decode inside the hook (only every", flush=True)
print("  HOP steps, so fewer entries). Whatever dominates talker.forward is the", flush=True)
print("  bottleneck. If backbone >> 1 ms, the kernel hook is slow in-loop; if", flush=True)
print("  talker.forward >> backbone+codec, the cost is code_predictor/sampling.", flush=True)
print("=====================================", flush=True)
