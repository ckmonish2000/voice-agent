"""
diag_codec2.py — does the codec hang on its own, or only AFTER the megakernel ran?

diag_codec.py showed: generate (with kernel) collects 12 frames fine, then the
FIRST speech_tokenizer.decode() of (105,16) HANGS and never returns — main
thread, sequential, no other threads. So the codec call deadlocks.

But ear_test.py / kernel_in_loop.py decode the same ref+codes and return fine.
The difference under test here: did the MEGAKERNEL run in this process before the
codec call? The megakernel launches 128 persistent blocks; if those (or the CUDA
context state they leave) starve the codec's kernels of SMs, the codec would hang
ONLY when the kernel ran first.

Two isolated probes, fresh process each time via the MODE env var:

  MODE=nokernel : decode ref+codes WITHOUT ever running the megakernel.
                  (use canned codes so we never touch the kernel.)
  MODE=kernel   : run generate WITH the kernel, THEN decode. (reproduces the hang)

Run BOTH on the box (server stopped):
  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  COMMON="PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072"

  # probe 1 — codec alone, kernel NEVER runs (USE_KERNEL=0):
  env $COMMON USE_KERNEL=0 MODE=nokernel python diag_codec2.py

  # probe 2 — kernel runs, then codec (USE_KERNEL=1):
  env $COMMON USE_KERNEL=1 MODE=kernel   python diag_codec2.py

If probe 1 RETURNS but probe 2 HANGS -> the megakernel leaves the GPU/context in
a state the codec can't run in. Root cause = kernel<->codec GPU coexistence, and
the fix is to decode on a path that doesn't share the live kernel context (e.g.
run the codec on a separate CUDA stream, or decode after a cuda synchronize/reset,
or fall back to the PyTorch backbone for the streaming server).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

MODE = os.environ.get("MODE", "nokernel")
print(f"[diag] MODE={MODE} USE_KERNEL={os.environ.get('USE_KERNEL')!r}", flush=True)

from engine import StreamingTTSEngine, StreamMetrics  # noqa: E402

eng = StreamingTTSEngine()
print(f"[diag] engine ready. use_kernel={eng._use_kernel} device={eng.device}",
      flush=True)

ref = eng._voice["ref_code"].to(eng.device)
tok = eng.model.speech_tokenizer


def decode_once(label, codes):
    print(f"  [{label}] calling speech_tokenizer.decode() "
          f"codes shape={tuple(codes.shape)} ...", flush=True)
    t0 = time.perf_counter()
    wavs, _sr = tok.decode([{"audio_codes": codes}])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000
    w = wavs[0]
    wlen = w.shape[-1] if hasattr(w, "shape") else len(w)
    print(f"  [{label}] RETURNED in {dt:.0f} ms, {wlen} samples.", flush=True)


if MODE == "nokernel":
    # Never run the kernel. Decode ref alone, then ref + a few dummy frames.
    print("\n[PROBE nokernel] codec decode WITHOUT the kernel ever running...",
          flush=True)
    decode_once("ref-only", ref)
    dummy = torch.zeros((12, 16), dtype=ref.dtype, device=ref.device)
    decode_once("ref+12dummy", torch.cat([ref, dummy], dim=0))
    print("\n[PROBE nokernel] codec works without the kernel. "
          "If MODE=kernel hangs, the kernel is the cause.", flush=True)

else:  # MODE == "kernel"
    print("\n[PROBE kernel] generate WITH kernel, then decode...", flush=True)
    import queue
    q: "queue.Queue" = queue.Queue()
    talker, eos = eng.model.talker, eng._eos

    def _frames_only_hook(_m, _i, output):
        hs = getattr(output, "hidden_states", None)
        if not (isinstance(hs, tuple) and len(hs) == 2):
            return output
        cid = hs[1]
        if cid is None:
            return output
        f = cid.detach().view(-1)[:16]
        if int(f[0]) == eos:
            return output
        q.put(f.clone())
        return output

    eng._hook_handle = talker.register_forward_hook(_frames_only_hook)
    if eng._use_kernel:
        eng._install_kernel_backbone_hook()
    input_ids = eng._text_to_input_ids("hey")
    gen_kwargs = eng._wrapper._merge_generate_kwargs(
        max_new_tokens=32, do_sample=True, temperature=0.9, top_k=50)
    with torch.no_grad():
        eng.model.generate(
            input_ids=[input_ids], ref_ids=[eng._voice["ref_tok"]],
            voice_clone_prompt=eng._voice["vc_prompt"], languages=["English"],
            non_streaming_mode=False, **gen_kwargs)
    eng._remove_frame_hook()
    if eng._use_kernel:
        eng._remove_kernel_backbone_hook()
    frames = []
    while not q.empty():
        frames.append(q.get().view(-1)[:16].to(eng.device))
    print(f"[PROBE kernel] collected {len(frames)} frames. Now decoding...",
          flush=True)
    if frames:
        gen = torch.stack(frames, dim=0)
        decode_once("ref+gen(after kernel)", torch.cat([ref, gen], dim=0))
    print("\n[PROBE kernel] if this RETURNED, the kernel is NOT the cause; "
          "if it HUNG, kernel<->codec coexistence is the bug.", flush=True)

print("\n[diag] done.", flush=True)
