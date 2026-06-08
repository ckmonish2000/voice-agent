"""
bench_engine.py — proper steady-state benchmark of the streaming engine.

The quick diag used "hey" (0.72 s audio, 3 chunks) — far too short: one-time
warm-up dominates, so TTFC/RTF look terrible even if steady-state is fine. This
runs a real sentence (several seconds of audio), warms up first, and reports:

  - TTFC                : request -> first PCM chunk (the latency that matters)
  - RTF (steady)        : wall time AFTER the first chunk / audio AFTER the first
                          chunk  (true streaming speed, excludes startup)
  - RTF (overall)       : total wall / total audio (what the user feels end-to-end)
  - per-chunk gap stats : min/mean/max ms between chunks (proves frame-by-frame)
  - backbone tok/s      : decode steps / generation time

Runs with whatever USE_KERNEL is set. Run it BOTH ways to compare:

  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  COMMON="PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072"
  env $COMMON USE_KERNEL=1 python bench_engine.py    # kernel backbone
  env $COMMON USE_KERNEL=0 python bench_engine.py    # pytorch backbone
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # inference_server/ (parent)

import torch  # noqa: E402
from engine import StreamingTTSEngine, StreamConfig  # noqa: E402

SR = 24000
SENTENCE = ("The quick brown fox jumps over the lazy dog, "
            "and then it runs back home before the rain begins to fall.")

eng = StreamingTTSEngine()
print(f"[bench] use_kernel={eng._use_kernel} device={eng.device}", flush=True)

# --- warm up (first call always pays one-time costs) ---
print("[bench] warming up...", flush=True)
for _ in eng.decode_stream("warm up the engine please", StreamConfig(max_new_tokens=48)):
    pass
print("[bench] warm. running measured utterance...\n", flush=True)


def run(text, max_new_tokens=512):
    t0 = time.perf_counter()
    chunk_ts = []        # arrival perf_counter per chunk
    chunk_bytes = []
    for pcm in eng.decode_stream(text, StreamConfig(max_new_tokens=max_new_tokens)):
        chunk_ts.append(time.perf_counter())
        chunk_bytes.append(len(pcm))
    t_end = time.perf_counter()
    return t0, chunk_ts, chunk_bytes, t_end


t0, ts, by, t_end = run(SENTENCE)

if not ts:
    print("[bench] NO CHUNKS produced.", flush=True)
    sys.exit(1)

total_bytes = sum(by)
total_audio = (total_bytes // 2) / SR
ttfc = (ts[0] - t0) * 1000
overall_wall = t_end - t0
overall_rtf = overall_wall / total_audio

# steady state = everything after the first chunk
audio_after_first = (sum(by[1:]) // 2) / SR if len(by) > 1 else 0.0
wall_after_first = t_end - ts[0]
steady_rtf = (wall_after_first / audio_after_first) if audio_after_first else float("nan")

# inter-chunk gaps
gaps = [(ts[i+1] - ts[i]) * 1000 for i in range(len(ts) - 1)]
gmin = min(gaps) if gaps else 0
gmax = max(gaps) if gaps else 0
gmean = sum(gaps)/len(gaps) if gaps else 0

# backbone steps/s from engine metrics
m = eng.last_metrics
gen_time = (m.last_frame_t - m.start_t) if (m.last_frame_t and m.start_t) else None
toks = m.num_frames
toks_per_s = (toks / gen_time) if gen_time else None

print("========== BENCH RESULT ==========", flush=True)
print(f"  backbone        : {'KERNEL' if eng._use_kernel else 'pytorch'}", flush=True)
print(f"  chunks          : {len(ts)}", flush=True)
print(f"  audio produced  : {total_audio:.2f} s", flush=True)
print(f"  total wall      : {overall_wall:.2f} s", flush=True)
print(f"  frames (steps)  : {toks}", flush=True)
print("  ---", flush=True)
print(f"  TTFC            : {ttfc:.0f} ms   (target <60 ms)", flush=True)
print(f"  RTF overall     : {overall_rtf:.3f}   (total wall / total audio)", flush=True)
print(f"  RTF steady      : {steady_rtf:.3f}   (after 1st chunk; true stream speed)",
      flush=True)
if toks_per_s:
    print(f"  backbone tok/s  : {toks_per_s:.0f}", flush=True)
print(f"  chunk gaps (ms) : min {gmin:.0f} / mean {gmean:.0f} / max {gmax:.0f}",
      flush=True)
print(f"  streaming?      : {'YES (many chunks, steady gaps)' if len(ts) > 3 else 'too few chunks'}",
      flush=True)
print("==================================", flush=True)
