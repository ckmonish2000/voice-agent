"""
benchmark.py — REAL end-to-end TTS speed: is the kernel fast enough for Pipecat?

The "slow" you saw was kernel_in_loop.py in VERIFY mode (runs PyTorch AND the
kernel + a cuda.synchronize() every step). That is NOT the kernel's speed. This
script measures the actual numbers that decide real-time viability:

  1. Kernel backbone alone   — raw decode_from_hidden throughput (no PyTorch, no
                               per-step sync stall). This is what the kernel buys.
  2. PyTorch talker.generate — the current production path end-to-end (backbone +
                               code_predictor + the loop), for the baseline RTF.
  3. The per-frame budget    — frames/sec needed for real-time (12.5) vs achieved.

Why this matters: the kernel only accelerates the 28-layer BACKBONE. Each frame
also needs the 5-layer code_predictor x15 + codec. If the predictor dominates,
the backbone speedup is capped (Amdahl). This script shows the split so you know
whether Pipecat integration will hit RTF < 1, and what to accelerate next.

Run:  LDG_VOCAB_SIZE=3072 python benchmark.py "A medium length sentence to time."
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

TEXT = sys.argv[1] if len(sys.argv) > 1 else "The quick brown fox jumps over the lazy dog."
MAX_NEW_TOKENS = 256
WARMUP = 3
BENCH_STEPS = 100        # backbone-alone timing iterations
REALTIME_FPS = 12.5      # codec frame rate -> 80 ms/frame budget for RTF<1

REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)


def _sync():
    torch.cuda.synchronize()


def bench_kernel_backbone(dec):
    """Raw kernel throughput: decode_from_hidden, no PyTorch, ONE sync at the end
    of the timed batch (not per step) — the kernel's true steady-state speed."""
    _dfh = torch.ops.qwen_tts_megakernel_C.decode_from_hidden
    h = torch.randn(1024, dtype=torch.bfloat16, device="cuda").contiguous()

    def one(pos):
        _dfh(
            dec._out_token, h,
            dec._embed_weight, dec._layer_weights_packed,
            dec._final_norm_weight, dec._lm_head_weight,
            dec._cos_table, dec._sin_table, dec._k_cache, dec._v_cache,
            dec._hidden, dec._act, dec._res, dec._q, dec._k, dec._v,
            dec._attn_out, dec._mlp_inter, dec._norm_out,
            dec._bmax_vals, dec._bmax_idxs,
            28, pos, dec._k_cache.shape[2], dec._attn_scale,
        )

    dec.reset()
    for i in range(WARMUP):
        one(i)
    _sync()

    t0 = time.perf_counter()
    for i in range(BENCH_STEPS):
        one(i)
    _sync()
    dt = time.perf_counter() - t0
    per_step_ms = dt / BENCH_STEPS * 1000
    return BENCH_STEPS / dt, per_step_ms   # steps/sec, ms/step


def bench_pytorch_e2e(tts, model, voice, text):
    """Full PyTorch talker.generate (backbone + code_predictor loop). Returns
    (frames, seconds, frames_per_sec)."""
    input_ids = tts._tokenize_texts([tts._build_assistant_text(text)])[0]
    gk = tts._merge_generate_kwargs(max_new_tokens=MAX_NEW_TOKENS)
    # warmup
    with torch.no_grad():
        tts_gen(model, input_ids, voice, gk)
    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        codes = tts_gen(model, input_ids, voice, gk)
    _sync()
    dt = time.perf_counter() - t0
    T = codes.shape[0]
    return T, dt, T / dt


def tts_gen(model, input_ids, voice, gk):
    codes_list, _ = model.generate(
        input_ids=[input_ids], ref_ids=[voice["ref_tok"]],
        voice_clone_prompt=voice["vc"], languages=["English"],
        non_streaming_mode=False, **gk,
    )
    return codes_list[0]


def main():
    if os.environ.get("LDG_VOCAB_SIZE") != "3072":
        print("WARNING: set LDG_VOCAB_SIZE=3072\n")

    from qwen_tts import Qwen3TTSModel
    from qwen_tts_megakernel.model_tts import build_talker_decoder

    print("[bench] loading kernel decoder + model...")
    dec, _ = build_talker_decoder(verbose=False)
    tts = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base", dtype=torch.bfloat16, device_map="cuda")
    model = tts.model

    pi = tts.create_voice_clone_prompt(ref_audio=REF_AUDIO, ref_text=REF_TEXT,
                                       x_vector_only_mode=False)
    voice = {
        "vc": tts._prompt_items_to_voice_clone_prompt(pi),
        "ref_tok": tts._tokenize_texts([tts._build_ref_text(REF_TEXT)])[0],
    }

    print("\n" + "=" * 64)
    print("1) KERNEL BACKBONE ALONE (decode_from_hidden, steady-state)")
    print("=" * 64)
    sps, ms = bench_kernel_backbone(dec)
    print(f"  {sps:8.1f} backbone steps/sec   ({ms:.3f} ms/step)")
    print(f"  budget for real-time: 1 backbone/frame, {REALTIME_FPS} frames/sec")
    print(f"  -> backbone alone is {sps/REALTIME_FPS:.0f}x faster than real-time")

    print("\n" + "=" * 64)
    print("2) FULL PyTorch talker.generate (backbone + code_predictor x15/frame)")
    print("=" * 64)
    T, dt, fps = bench_pytorch_e2e(tts, model, voice, TEXT)
    audio_s = T / REALTIME_FPS
    rtf = dt / audio_s
    print(f"  text: {TEXT!r}")
    print(f"  {T} frames in {dt:.2f}s  ->  {fps:.1f} frames/sec")
    print(f"  audio length {audio_s:.2f}s  ->  RTF = {rtf:.2f}  "
          f"({'REAL-TIME OK' if rtf < 1 else f'{rtf:.1f}x too slow'})")

    print("\n" + "=" * 64)
    print("3) READING THE RESULT")
    print("=" * 64)
    print(f"  - Kernel backbone: {ms:.3f} ms/step (≈{sps:.0f}/s). Negligible vs the")
    print(f"    {1000/REALTIME_FPS:.0f} ms/frame real-time budget.")
    print("  - PyTorch full path RTF above includes the code_predictor (5-layer x15)")
    print("    and codec, which the kernel does NOT accelerate. If that RTF is still")
    print("    >1, the bottleneck is the predictor/codec, not the backbone — so the")
    print("    next win is accelerating the predictor, not the backbone.")
    print("  - For Pipecat: swap the backbone for the kernel behind the engine seam;")
    print("    expect the backbone's share of per-frame time to ~vanish. Net RTF")
    print("    depends on what's left (predictor+codec). Measure with kernel_in_loop")
    print("    VERIFY=0 once a fast (no-sync) loop is wired.")


if __name__ == "__main__":
    main()
