"""
diag_thread.py — isolate the ONE variable that differs from the working
kernel_in_loop.py: is generation hanging because it runs on a NON-main thread?

Evidence so far:
  - kernel_in_loop.py (generate on MAIN thread)         -> WORKS
  - server / decode_stream (generate on a WORKER thread) -> hangs ~13 frames in,
    worker thread alive-but-frozen, even after all GPU work was moved onto it.

This runs the SAME generate() call twice, with a step counter wired into the
kernel post-hook so we see exactly how many decode steps complete:

  RUN A: generate() on the MAIN thread.
  RUN B: generate() on a WORKER thread (daemon), main thread watches a heartbeat.

If A completes and B freezes -> the custom megakernel op (decode_from_hidden)
cannot be launched safely from a non-main Python thread. That's the root cause,
and the fix is to run generation on the main thread (or pin the CUDA context).

Run on the box (server stopped), kernel path:
  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=1 \
  python diag_thread.py
"""

import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402
from engine import StreamingTTSEngine  # noqa: E402

eng = StreamingTTSEngine()
print(f"[diag] engine ready. use_kernel={eng._use_kernel} device={eng.device}",
      flush=True)

steps = {"n": 0, "where": "init"}


def _instrument_post_hook():
    """Wrap the kernel post-hook so each phase bumps a counter we can watch."""
    talker_model = eng.model.talker.model
    dec = eng._kdec
    st = {"seeded": False, "emb": None}

    def _layer_kv(pkv, L):
        if hasattr(pkv, "layers"):
            return pkv.layers[L].keys, pkv.layers[L].values
        return pkv.key_cache[L], pkv.value_cache[L]

    def _seed(pkv):
        kc, vc = dec._k_cache, dec._v_cache
        for L in range(kc.shape[0]):
            k, v = _layer_kv(pkv, L)
            n = min(k.shape[2], kc.shape[2])
            kc[L, :, :n, :] = k[0, :, :n, :].to(kc.dtype)
            vc[L, :, :n, :] = v[0, :, :n, :].to(vc.dtype)

    def pre_hook(_m, args, kwargs):
        emb = kwargs.get("inputs_embeds")
        if emb is None and args:
            emb = args[0]
        st["emb"] = emb

    def post_hook(_m, _a, _kw, output):
        hs = output.last_hidden_state
        if hs.shape[1] != 1:
            return output
        pkv = output.past_key_values
        pos = pkv.get_seq_length() - 1
        if not st["seeded"]:
            steps["where"] = "seeding"
            _seed(pkv)
            st["seeded"] = True
        steps["where"] = f"pre-kernel(step {steps['n']})"
        emb = st["emb"].detach().to(torch.bfloat16).reshape(-1).contiguous()
        dec._position = pos
        eng._dfh(
            dec._out_token, emb,
            dec._embed_weight, dec._layer_weights_packed,
            dec._final_norm_weight, dec._lm_head_weight,
            dec._cos_table, dec._sin_table, dec._k_cache, dec._v_cache,
            dec._hidden, dec._act, dec._res, dec._q, dec._k, dec._v,
            dec._attn_out, dec._mlp_inter, dec._norm_out,
            dec._bmax_vals, dec._bmax_idxs,
            28, pos, dec._k_cache.shape[2], dec._attn_scale,
        )
        steps["where"] = f"post-kernel(step {steps['n']})"
        k_pre = dec._hidden.detach().to(torch.bfloat16).view(1, 1, -1)
        output.last_hidden_state = talker_model.norm(k_pre)
        steps["n"] += 1
        steps["where"] = f"done step {steps['n']}"
        return output

    eng._k_pre_handle = talker_model.register_forward_pre_hook(
        pre_hook, with_kwargs=True)
    eng._k_post_handle = talker_model.register_forward_hook(
        post_hook, with_kwargs=True)


def _run_generate():
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


# ============ RUN A: generate on the MAIN thread ============
print("\n[RUN A] generate() on the MAIN thread (the kernel_in_loop case)...",
      flush=True)
_instrument_post_hook()
steps["n"] = 0
ta = time.perf_counter()
try:
    _run_generate()
    print(f"[RUN A] COMPLETED on main thread: {steps['n']} kernel steps in "
          f"{(time.perf_counter()-ta)*1000:.0f} ms", flush=True)
except Exception as e:
    import traceback
    print(f"[RUN A] raised: {e!r}", flush=True)
    traceback.print_exc()
finally:
    eng._remove_kernel_backbone_hook()

# ============ RUN B: generate on a WORKER thread ============
print("\n[RUN B] generate() on a WORKER thread; main thread watches...",
      flush=True)
_instrument_post_hook()
steps["n"] = 0
steps["where"] = "starting worker"
done = threading.Event()


def _worker():
    try:
        _run_generate()
    except Exception as e:
        print(f"[RUN B] worker raised: {e!r}", flush=True)
    finally:
        done.set()


w = threading.Thread(target=_worker, name="gen-worker", daemon=True)
tb = time.perf_counter()
w.start()
last_n = -1
stalls = 0
while not done.wait(timeout=1.0):
    n = steps["n"]
    print(f"[RUN B] t={time.perf_counter()-tb:.1f}s steps={n} where={steps['where']!r} "
          f"worker_alive={w.is_alive()}", flush=True)
    if n == last_n:
        stalls += 1
    else:
        stalls = 0
    last_n = n
    if stalls >= 6:
        print(f"\n[RUN B] STALLED at step {n}, phase {steps['where']!r}. "
              "Worker is frozen here.", flush=True)
        break
else:
    print(f"[RUN B] COMPLETED on worker thread: {steps['n']} steps in "
          f"{(time.perf_counter()-tb)*1000:.0f} ms", flush=True)
eng._remove_kernel_backbone_hook()

print("\n========== VERDICT ==========", flush=True)
print("  If RUN A completed but RUN B stalled -> the megakernel op is unsafe on a", flush=True)
print("  non-main Python thread. Fix: run generation on the MAIN thread.", flush=True)
print("  If BOTH stalled -> not a thread issue; look at the stalled phase above.", flush=True)
print("=============================", flush=True)
