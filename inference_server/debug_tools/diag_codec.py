"""
diag_codec.py — the codec decode is now the prime suspect.

New evidence (diag_thread.py): generate()+kernel runs fine on BOTH the main
thread AND a worker thread when NO codec decode is involved. The hang only
appears once the codec (speech_tokenizer.decode) is in the loop. So test the
codec in isolation, with instrumentation around the decode call.

Sequential, main thread, no extra threads anywhere:
  STEP 1: generate() collecting frames via a frame hook (NO codec).
  STEP 2: feed those frames to the sliding-window codec decoder one at a time,
          printing BEFORE and AFTER each speech_tokenizer.decode() call so we
          see exactly which decode call (if any) hangs and how long it takes.

If a decode() call never returns -> the codec is the hang, independent of
threads. If all decode() calls return quickly -> the hang is the INTERACTION of
codec + generate running in the same process/loop, and we instrument that next.

Run on the box (server stopped), kernel path:
  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=1 \
  python diag_codec.py
"""

import os
import sys
import time
import queue

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # inference_server/ (parent)

import torch  # noqa: E402
from engine import (  # noqa: E402
    StreamingTTSEngine, StreamMetrics, _FRAME_SENTINEL,
)

eng = StreamingTTSEngine()
print(f"[diag] engine ready. use_kernel={eng._use_kernel} device={eng.device}",
      flush=True)

# ---- STEP 1: generate, collect frames, NO codec (we know this works) ----
print("\n[STEP 1] generate() collecting frames (no codec)...", flush=True)
q: "queue.Queue" = queue.Queue()
metrics = StreamMetrics()
metrics.start_t = time.perf_counter()

# install a minimal frame hook that ONLY pushes frames (no codec)
talker = eng.model.talker
eos = eng._eos


def _frames_only_hook(_m, _i, output):
    hs = getattr(output, "hidden_states", None)
    if not (isinstance(hs, tuple) and len(hs) == 2):
        return output
    codec_ids = hs[1]
    if codec_ids is None:
        return output
    frame = codec_ids.detach().view(-1)[:16]
    if int(frame[0]) == eos:
        return output
    q.put(frame.clone())
    return output


eng._hook_handle = talker.register_forward_hook(_frames_only_hook)
if eng._use_kernel:
    eng._install_kernel_backbone_hook()

input_ids = eng._text_to_input_ids("hey")
gen_kwargs = eng._wrapper._merge_generate_kwargs(
    max_new_tokens=32, do_sample=True, temperature=0.9, top_k=50)
with torch.no_grad():
    eng.model.generate(
        input_ids=[input_ids],
        ref_ids=[eng._voice["ref_tok"]],
        voice_clone_prompt=eng._voice["vc_prompt"],
        languages=["English"],
        non_streaming_mode=False,
        **gen_kwargs,
    )
eng._remove_frame_hook()
if eng._use_kernel:
    eng._remove_kernel_backbone_hook()

frames = []
while not q.empty():
    frames.append(q.get())
print(f"[STEP 1] collected {len(frames)} frames.", flush=True)
if not frames:
    print("[STEP 1] no frames — stop.", flush=True)
    sys.exit(0)

# ---- STEP 2: codec decode each frame, instrument every decode() call ----
print("\n[STEP 2] codec decode, sequential, instrumented...", flush=True)
ref = eng._voice["ref_code"].to(eng.device)
WINDOW, HOP, UPSAMPLE = 16, 4, 1920
buf = []
emitted = 0
tok = eng.model.speech_tokenizer

for i, f in enumerate(frames):
    buf.append(f.view(-1)[:16].to(eng.device))
    if len(buf) - emitted < HOP:
        continue
    start = max(0, len(buf) - WINDOW)
    window_frames = torch.stack(buf[start:], dim=0)
    codes = torch.cat([ref, window_frames], dim=0)
    print(f"  [decode {i+1}] calling speech_tokenizer.decode() "
          f"codes shape={tuple(codes.shape)} ...", flush=True)
    t0 = time.perf_counter()
    wavs, _sr = tok.decode([{"audio_codes": codes}])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    w = wavs[0]
    wlen = (w.shape[-1] if hasattr(w, "shape") else len(w))
    print(f"  [decode {i+1}] RETURNED in {dt:.0f} ms, {wlen} samples.",
          flush=True)
    emitted = len(buf)

print("\n========== VERDICT ==========", flush=True)
print("  If every decode() RETURNED -> codec alone is fine; the hang is the", flush=True)
print("  codec+generate interaction (re-entrancy / shared state). Instrument that.", flush=True)
print("  If a decode() never returned -> the codec call itself hangs. That's it.", flush=True)
print("=============================", flush=True)
