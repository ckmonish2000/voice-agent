"""
diag_double.py — answers TWO questions in one run:

(1) DEVICE CHECK: is the code_predictor (makes codes 1..15, ~76% of per-step
    time) actually on the GPU, or accidentally on the CPU like the codec was?
    If it's on cpu, that alone explains the ~173 ms/step we couldn't account for.

(2) DOUBLE-COMPUTE CHECK: the kernel runs as a forward HOOK on talker.model. A
    hook fires AFTER the module's forward() already ran. So each step PyTorch may
    compute all 28 backbone layers (~40 ms, WASTED) and then the hook runs the
    kernel (~1 ms) and overwrites it. This times the PyTorch backbone fwd vs the
    kernel call alone to confirm.

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


def dev(mod):
    try:
        for p in mod.parameters():
            return p.device
    except Exception as e:
        return f"(no params: {e})"
    return "no-params"


# ---------- (1) DEVICE CHECK ----------
m = eng.model
talker = m.talker
print("\n========== DEVICE CHECK ==========", flush=True)
print(f"  talker                : {dev(talker)}", flush=True)
print(f"  talker.model (backbone): {dev(talker.model)}", flush=True)
cp = getattr(talker, "code_predictor", None)
if cp is None:
    # try common alt locations
    cp = getattr(m, "code_predictor", None)
print(f"  talker.code_predictor : {dev(cp) if cp is not None else 'NOT FOUND'}",
      flush=True)
print(f"  speech_tokenizer.model: {dev(m.speech_tokenizer.model)}", flush=True)
print("  (anything on 'cpu' here is a bug — should all be cuda)", flush=True)
print("==================================", flush=True)


# ---------- (2) DOUBLE-COMPUTE CHECK ----------
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


tm = talker.model
orig_fwd = tm.forward
pytorch_fwd_ms = []


def timed_fwd(*a, **k):
    sync(); t0 = time.perf_counter()
    out = orig_fwd(*a, **k)
    sync()
    try:
        if out.last_hidden_state.shape[1] == 1:
            pytorch_fwd_ms.append((time.perf_counter() - t0) * 1000)
    except Exception:
        pass
    return out


tm.forward = timed_fwd

orig_dfh = eng._dfh
kernel_only_ms = []


def timed_dfh(*a, **k):
    sync(); t0 = time.perf_counter()
    out = orig_dfh(*a, **k)
    sync()
    kernel_only_ms.append((time.perf_counter() - t0) * 1000)
    return out


eng._dfh = timed_dfh

print("\n[diag] warming up...", flush=True)
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
print("  If PyTorch backbone ~40 ms and kernel ~1 ms: PyTorch backbone runs each", flush=True)
print("  step and is thrown away. Fix: run the kernel INSTEAD of PyTorch forward.", flush=True)
print("==========================================", flush=True)
