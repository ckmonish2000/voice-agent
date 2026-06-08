"""
diag_double.py — confirm the backbone is computed TWICE per step (PyTorch then
kernel), wasting the kernel's speed.

Hypothesis: the kernel runs as a forward HOOK on talker.model. A forward hook
fires AFTER the module's own forward() has already run. So each step:
  1. talker.model.forward() computes all 28 layers in PyTorch (~40 ms, WASTED)
  2. post_hook runs the kernel (~1 ms) and overwrites the result
The 41 ms/step "backbone (kernel)" we measured = mostly the discarded PyTorch
backbone, NOT the kernel.

This times, inside the post_hook, ONLY the kernel call (self._dfh + norm), and
separately times the PyTorch forward that ran just before it. If pytorch_fwd is
~40 ms and kernel_only is ~1 ms, the hypothesis is confirmed and the fix is to
bypass PyTorch's backbone (run the kernel in place of forward, not after it).

Run on the box:
  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=1 \
  python diag_double.py
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


# time the PyTorch backbone forward (the thing that runs BEFORE the hook)
tm = eng.model.talker.model
orig_fwd = tm.forward
pytorch_fwd_ms = []


def timed_fwd(*a, **k):
    sync(); t0 = time.perf_counter()
    out = orig_fwd(*a, **k)
    sync()
    # only count decode steps (seq len 1), not prefill
    try:
        if out.last_hidden_state.shape[1] == 1:
            pytorch_fwd_ms.append((time.perf_counter() - t0) * 1000)
    except Exception:
        pass
    return out


tm.forward = timed_fwd

# time ONLY the kernel call by wrapping self._dfh
orig_dfh = eng._dfh
kernel_only_ms = []


def timed_dfh(*a, **k):
    sync(); t0 = time.perf_counter()
    out = orig_dfh(*a, **k)
    sync()
    kernel_only_ms.append((time.perf_counter() - t0) * 1000)
    return out


eng._dfh = timed_dfh

print("[diag] warming up...", flush=True)
for _ in eng.decode_stream("warm up", StreamConfig(max_new_tokens=16)):
    pass
pytorch_fwd_ms.clear(); kernel_only_ms.clear()

print("[diag] measured run...\n", flush=True)
for _ in eng.decode_stream(
        "The quick brown fox jumps over the lazy dog and runs home today.",
        StreamConfig(max_new_tokens=512)):
    pass


def stats(name, arr):
    if not arr:
        print(f"  {name:28s}: (none)", flush=True); return
    a = arr[1:] if len(arr) > 1 else arr
    print(f"  {name:28s}: n={len(arr):3d} avg={sum(a)/len(a):6.2f} ms "
          f"min={min(a):6.2f} max={max(a):6.2f}", flush=True)


print("========== DOUBLE-COMPUTE CHECK ==========", flush=True)
stats("PyTorch backbone fwd/step", pytorch_fwd_ms)
stats("kernel call only (_dfh)", kernel_only_ms)
print("  ---", flush=True)
print("  If PyTorch backbone ~40 ms and kernel ~1 ms: the PyTorch backbone runs", flush=True)
print("  every step and is thrown away. Fix: run the kernel INSTEAD of PyTorch's", flush=True)
print("  forward (skip it), not as an after-the-fact hook.", flush=True)
print("==========================================", flush=True)
