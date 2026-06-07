"""
Phase 4c — kernel-in-the-loop end-to-end audio (drop-in correctness path).

The KERNEL runs the 28-layer talker backbone each decode step; PyTorch does
prefill + code_predictor + codec. Produces kernel_out.wav.

How it plugs in (uses the new decode_from_hidden op + compatible KV layouts):
  - pre-hook on talker.model: capture this step's inputs_embeds (the summed
    code-embedding, (1,1,1024)) and position, and on the first decode step seed
    the kernel KV cache from PyTorch's prefill K/V (both are (.,8,seq,128)).
  - post-hook on talker.model: run kernel decode_from_hidden(inputs_embeds) at the
    current position, take its PRE-norm hidden (_hidden), apply PyTorch's
    talker.model.norm to it, and OVERWRITE output.last_hidden_state.
    (diag_hidden.py showed the kernel's own _norm_out drifts; _hidden + PyTorch
     norm is the faithful path — same fix as parity_frame16.)

Modes (env VERIFY):
  VERIFY=1 (default): compare kernel-substituted hidden vs PyTorch hidden each
    step, print running max-diff; do NOT substitute (audio = pure PyTorch). Proves
    the cache seeding + per-step wiring over a real utterance.
  VERIFY=0: actually substitute -> kernel-driven audio in kernel_out.wav.

Run:  LDG_VOCAB_SIZE=3072 VERIFY=1 python kernel_in_loop.py "Hello there."
then: LDG_VOCAB_SIZE=3072 VERIFY=0 python kernel_in_loop.py "Hello there."
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEXT = sys.argv[1] if len(sys.argv) > 1 else "Hello there, this is the kernel."
VERIFY = os.environ.get("VERIFY", "1") == "1"
MAX_NEW_TOKENS = 256
HERE = os.path.dirname(os.path.abspath(__file__))

REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)


def main():
    if os.environ.get("LDG_VOCAB_SIZE") != "3072":
        print("WARNING: set LDG_VOCAB_SIZE=3072\n")

    import numpy as np
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel
    from model_tts import build_talker_decoder

    print(f"[4c] loading; text={TEXT!r}  VERIFY={VERIFY}")
    # Kernel decoder (talker weights packed for the kernel).
    dec, _ = build_talker_decoder(verbose=False)
    # Wrapper around the PyTorch model for the generate() helpers + the loop.
    # Force CUDA + bf16 to match the kernel decoder (default load is CPU -> device
    # mismatch when our CUDA hook tensors meet the model).
    tts = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base", dtype=torch.bfloat16, device_map="cuda"
    )
    model = tts.model
    talker_model = model.talker.model
    tok = model.speech_tokenizer
    _dfh = torch.ops.qwen_megakernel_C.decode_from_hidden

    prompt_items = tts.create_voice_clone_prompt(
        ref_audio=REF_AUDIO, ref_text=REF_TEXT, x_vector_only_mode=False)
    vc_prompt = tts._prompt_items_to_voice_clone_prompt(prompt_items)
    ref_code = prompt_items[0].ref_code
    ref_tok = tts._tokenize_texts([tts._build_ref_text(REF_TEXT)])[0]
    input_ids = tts._tokenize_texts([tts._build_assistant_text(TEXT)])[0]
    gen_kwargs = tts._merge_generate_kwargs(max_new_tokens=MAX_NEW_TOKENS)

    st = {"seeded": False, "emb": None, "pos": None, "steps": 0, "maxdiff": 0.0}

    def _layer_kv(pkv, L):
        """Return (keys, values) for layer L across transformers cache APIs.
        5.x: pkv.layers[L].keys/.values ; older: pkv.key_cache[L]/value_cache[L]."""
        if hasattr(pkv, "layers"):
            layer = pkv.layers[L]
            return layer.keys, layer.values
        return pkv.key_cache[L], pkv.value_cache[L]

    def seed_cache(pkv, upto):
        kc, vc = dec._k_cache, dec._v_cache       # (L,8,MAX,128)
        for L in range(kc.shape[0]):
            k, v = _layer_kv(pkv, L)              # (1,8,seq,128)
            n = min(k.shape[2], kc.shape[2])
            kc[L, :, :n, :] = k[0, :, :n, :].to(kc.dtype)
            vc[L, :, :n, :] = v[0, :, :n, :].to(vc.dtype)

    def pre_hook(_m, args, kwargs):
        emb = kwargs.get("inputs_embeds")
        if emb is None and args:
            emb = args[0]
        st["emb"] = emb
        st["pkv"] = kwargs.get("past_key_values")

    def post_hook(_m, _args, _kwargs, output):
        hs = output.last_hidden_state
        if hs.shape[1] != 1:        # prefill — leave alone
            return output
        pkv = output.past_key_values
        pos = pkv.get_seq_length() - 1   # this step's absolute position
        if not st["seeded"]:
            seed_cache(pkv, pos)         # cache now holds prefill + this token's KV
            st["seeded"] = True

        emb = st["emb"].detach().to(torch.bfloat16).reshape(-1).contiguous()  # (1024,)
        dec._position = pos
        _dfh(
            dec._out_token, emb,
            dec._embed_weight, dec._layer_weights_packed,
            dec._final_norm_weight, dec._lm_head_weight,
            dec._cos_table, dec._sin_table, dec._k_cache, dec._v_cache,
            dec._hidden, dec._act, dec._res, dec._q, dec._k, dec._v,
            dec._attn_out, dec._mlp_inter, dec._norm_out,
            dec._bmax_vals, dec._bmax_idxs,
            28, pos, dec._k_cache.shape[2], dec._attn_scale,
        )
        torch.cuda.synchronize()
        k_pre = dec._hidden.detach().to(torch.bfloat16).view(1, 1, -1)
        k_hidden = talker_model.norm(k_pre)     # post-norm via PyTorch (faithful)

        st["steps"] += 1
        d = (k_hidden.float() - hs.float()).abs().max().item()
        st["maxdiff"] = max(st["maxdiff"], d)

        if not VERIFY:
            output.last_hidden_state = k_hidden  # SUBSTITUTE kernel hidden
        return output

    h1 = talker_model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = talker_model.register_forward_hook(post_hook, with_kwargs=True)
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            codes_list, _ = model.generate(
                input_ids=[input_ids], ref_ids=[ref_tok],
                voice_clone_prompt=vc_prompt, languages=["English"],
                non_streaming_mode=False, **gen_kwargs,
            )
    finally:
        h1.remove(); h2.remove()

    codes = codes_list[0]
    print(f"\n[4c] steps={st['steps']}  seeded={st['seeded']}  "
          f"max hidden diff (kernel vs PyTorch over the utterance) = {st['maxdiff']:.4f}")
    print(f"[4c] generated {codes.shape[0]} frames  (~{codes.shape[0]/12.5:.2f}s)")

    # render audio (PyTorch codes in VERIFY mode; kernel-influenced codes if VERIFY=0)
    full = torch.cat([ref_code.to(codes.device), codes], dim=0)
    wavs, sr = tok.decode([{"audio_codes": full}])
    wav = wavs[0]
    wav = wav.detach().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
    wav = wav.reshape(-1)
    cut = int(ref_code.shape[0] / full.shape[0] * wav.shape[0])
    out = "kernel_out.wav" if not VERIFY else "kernel_verify.wav"
    sf.write(os.path.join(HERE, out), wav[cut:], sr)
    print(f"[4c] wrote {out} ({len(wav[cut:])/sr:.2f}s)")
    if VERIFY:
        print("[4c] VERIFY mode: audio is PyTorch; the diff above is the real signal.")
        print("     If diff is small (~0.2, the 28-layer bf16 floor), re-run with VERIFY=0")
        print("     to render kernel-driven audio.")


if __name__ == "__main__":
    main()
