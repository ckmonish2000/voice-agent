"""
diag_hang.py — verify the fix: decode_stream must stream PCM without hanging.

The hang was concurrent GPU access from two threads (generate on a worker thread,
codec decode on the foreground). Fix: codec decode now runs inside the frame hook
on the SAME thread as generate(); the foreground only drains finished PCM bytes.

This drives the REAL decode_stream() (worker thread does all GPU work) and a
heartbeat proves we get the first chunk in well under a second, with the worker
finishing cleanly (not frozen-alive like before).

Run on the box (server stopped), kernel path:
  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=1 \
  python diag_hang.py
"""

import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402
from engine import StreamingTTSEngine, StreamConfig  # noqa: E402

print(f"[diag] USE_KERNEL={os.environ.get('USE_KERNEL')!r} "
      f"QWEN_DEVICE={os.environ.get('QWEN_DEVICE')!r}", flush=True)

eng = StreamingTTSEngine()
print(f"[diag] engine ready. use_kernel={eng._use_kernel} device={eng.device}",
      flush=True)

stop_beat = threading.Event()
state = {"chunks": 0}


def _heartbeat():
    while not stop_beat.is_set():
        time.sleep(1.0)
        alive = [t.name for t in threading.enumerate()
                 if t is not threading.current_thread()]
        print(f"[HB] t={time.perf_counter():.1f} chunks={state['chunks']} "
              f"threads={alive}", flush=True)


hb = threading.Thread(target=_heartbeat, daemon=True)
hb.start()

print("[diag] calling decode_stream...", flush=True)
t0 = time.perf_counter()
first_t = None
total_bytes = 0
try:
    for pcm in eng.decode_stream("hey", StreamConfig(max_new_tokens=64)):
        state["chunks"] += 1
        total_bytes += len(pcm)
        if first_t is None:
            first_t = time.perf_counter()
            print(f"[OK] FIRST PCM chunk at {(first_t - t0)*1000:.0f} ms "
                  f"({len(pcm)} bytes)", flush=True)
except Exception as e:
    import traceback
    print(f"[diag] decode_stream raised: {e!r}", flush=True)
    traceback.print_exc()
finally:
    stop_beat.set()

dur = time.perf_counter() - t0
audio_s = (total_bytes // 2) / 24000
print("\n========== RESULT ==========", flush=True)
print(f"  chunks       : {state['chunks']}", flush=True)
print(f"  audio        : {audio_s:.2f} s in {dur:.2f} s wall "
      f"(RTF {dur/audio_s:.3f})" if audio_s else "  audio        : 0 s",
      flush=True)
if first_t:
    print(f"  TTFC         : {(first_t - t0)*1000:.0f} ms", flush=True)
print("  VERDICT      : " +
      ("STREAMING WORKS — hang fixed." if state["chunks"] > 1
       else "STILL BROKEN — investigate further."), flush=True)
print("============================", flush=True)
