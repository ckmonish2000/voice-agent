"""
diag_hang.py — boundary instrumentation for the server generation hang.

Systematic-debugging Phase 1: gather evidence at EVERY component boundary in a
single run, so we see WHERE the pipeline stalls instead of guessing.

Run on the box (server stopped), kernel path:

  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=1 \
  python diag_hang.py

It instruments four boundaries:
  [A] worker thread: did generate() even start? is it still alive?
  [B] talker.forward: did the frame hook fire? how many frames did it see?
  [C] queue: did the foreground receive any frame?
  [D] codec: did decode() of the first hop succeed / how long did it take?

Whichever boundary goes silent is the failing component. We print a verdict.
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

# ---- counters the instrumented hook will bump ----
counters = {"hook_calls": 0, "frames_seen": 0, "prefill_seen": 0,
            "eos_seen": 0, "non2tuple": 0}

# Wrap the frame hook installer so we can count what talker.forward emits,
# WITHOUT changing engine.py. We monkeypatch _install_frame_hook to also tap.
_orig_install = eng._install_frame_hook


def _tap_install(frame_q, metrics):
    talker = eng.model.talker
    eos = eng._eos

    def _hook(_module, _inputs, output):
        counters["hook_calls"] += 1
        hs = getattr(output, "hidden_states", None)
        if not (isinstance(hs, tuple) and len(hs) == 2):
            counters["non2tuple"] += 1
            return output
        codec_ids = hs[1]
        if codec_ids is None:
            counters["prefill_seen"] += 1
            print(f"[B] hook fired: PREFILL step (codec_ids None) "
                  f"call#{counters['hook_calls']}", flush=True)
            return output
        frame = codec_ids.detach().to("cpu").view(-1)[:16]
        if int(frame[0]) == eos:
            counters["eos_seen"] += 1
            print(f"[B] hook fired: EOS frame, skipping", flush=True)
            return output
        counters["frames_seen"] += 1
        if counters["frames_seen"] <= 6:
            print(f"[B] hook fired: FRAME #{counters['frames_seen']} "
                  f"code0={int(frame[0])} (pushing to queue)", flush=True)
        if metrics.first_frame_t is None:
            metrics.first_frame_t = time.perf_counter()
        metrics.frame_arrival_ts.append(time.perf_counter())
        metrics.num_frames += 1
        frame_q.put(frame.clone())
        return output

    eng._hook_handle = talker.register_forward_hook(_hook)


eng._install_frame_hook = _tap_install

# ---- A heartbeat thread: prove the worker is alive and report progress ----
stop_beat = threading.Event()


def _heartbeat():
    while not stop_beat.is_set():
        time.sleep(2.0)
        alive = [t.name for t in threading.enumerate()
                 if t is not threading.current_thread()]
        print(f"[HB] t={time.perf_counter():.1f} "
              f"hook_calls={counters['hook_calls']} "
              f"prefill={counters['prefill_seen']} "
              f"frames={counters['frames_seen']} "
              f"eos={counters['eos_seen']} non2tuple={counters['non2tuple']} "
              f"| threads={alive}", flush=True)


hb = threading.Thread(target=_heartbeat, daemon=True)
hb.start()

print("[diag] calling decode_stream (main-thread drain, worker does generate)...",
      flush=True)
t0 = time.perf_counter()
n_chunks = 0
first_chunk_t = None
try:
    for pcm in eng.decode_stream("hey", StreamConfig(max_new_tokens=32)):
        n_chunks += 1
        if first_chunk_t is None:
            first_chunk_t = time.perf_counter()
            print(f"[D] FIRST PCM CHUNK at {(first_chunk_t - t0)*1000:.0f} ms "
                  f"({len(pcm)} bytes) — codec path WORKS", flush=True)
        if n_chunks >= 8:
            print("[diag] got 8 chunks, that's enough to prove streaming.",
                  flush=True)
            break
except Exception as e:
    import traceback
    print(f"[diag] decode_stream raised: {e!r}", flush=True)
    traceback.print_exc()
finally:
    stop_beat.set()

print("\n========== VERDICT ==========", flush=True)
print(f"  hook fired total : {counters['hook_calls']}", flush=True)
print(f"  prefill steps    : {counters['prefill_seen']}", flush=True)
print(f"  real frames seen : {counters['frames_seen']}", flush=True)
print(f"  eos frames       : {counters['eos_seen']}", flush=True)
print(f"  PCM chunks out   : {n_chunks}", flush=True)
if counters["hook_calls"] == 0:
    print("  >> talker.forward NEVER ran. Hang is INSIDE generate() before the "
          "first forward — kernel pre/post-hook or prefill. (Boundary A->B)",
          flush=True)
elif counters["frames_seen"] == 0:
    print("  >> forward ran but produced NO real frame. Hang/loop is in the "
          "decode step (code_predictor / kernel backbone post-hook). (Boundary B)",
          flush=True)
elif n_chunks == 0:
    print("  >> frames were produced but NO PCM came out. Hang is in the codec "
          "decode or the HOP buffering. (Boundary C->D)", flush=True)
else:
    print("  >> FULL PIPELINE WORKS on this run. If the server still hangs, the "
          "difference is the server itself (event loop / websocket), not the "
          "engine.", flush=True)
print("=============================", flush=True)
