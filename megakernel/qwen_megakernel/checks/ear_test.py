"""
Ear test — does the 13/16 tail-codebook drift actually change the audio?

Our hybrid kernel path matches the first ~12 codebooks of each frame and drifts
on the last few (bf16 accumulation over 28 layers). This asks the only question
that matters for TTS: can you HEAR that difference?

Method (mirrors the working stages/ pipeline exactly — stage2 + stage4):
  1. model.generate(...) -> talker_codes_list[0] = clean (T,16) codes.
  2. ref copy decoded as-is; "drift" copy = last K codebooks per frame randomly
     resampled (WORSE than the kernel's near-misses = upper bound on damage).
  3. decode both WITH the ref_code voice prefix prepended, trim the ref portion
     off the waveform, write two wavs. Listen.

If even this worse-than-kernel perturbation sounds the same, the real 13/16
kernel output is audibly safe.

Run:  python ear_test.py "your sentence here"
Outputs: ear_ref.wav (PyTorch), ear_drift.wav (tail-codebooks perturbed)
"""

import os
import sys
import torch

TEXT = sys.argv[1] if len(sys.argv) > 1 else "Hello, this is a test of the speech kernel."
DRIFT_LAST_K = 4
SEED = 0
MAX_NEW_TOKENS = 512

REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    import numpy as np
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    print(f"[ear] loading model; text = {TEXT!r}")
    tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    model = tts.model
    tok = model.speech_tokenizer

    # ---- build voice-clone prompt + input ids (same as stages/stage2) ----
    prompt_items = tts.create_voice_clone_prompt(
        ref_audio=REF_AUDIO, ref_text=REF_TEXT, x_vector_only_mode=False
    )
    vc_prompt = tts._prompt_items_to_voice_clone_prompt(prompt_items)
    ref_code = prompt_items[0].ref_code            # (R,16) reference voice codes
    ref_tok = tts._tokenize_texts([tts._build_ref_text(REF_TEXT)])[0]
    input_ids = tts._tokenize_texts([tts._build_assistant_text(TEXT)])[0]
    gen_kwargs = tts._merge_generate_kwargs(max_new_tokens=MAX_NEW_TOKENS)

    # ---- real generate() -> clean (T,16) codes (THE working path) ----
    torch.manual_seed(SEED)
    print("[ear] running model.generate() ...")
    with torch.no_grad():
        talker_codes_list, _ = model.generate(
            input_ids=[input_ids],
            ref_ids=[ref_tok],
            voice_clone_prompt=vc_prompt,
            languages=["English"],
            non_streaming_mode=False,
            **gen_kwargs,
        )
    codes = talker_codes_list[0].to(model.device)   # (T,16)
    T, G = codes.shape
    print(f"[ear] generated {T} frames x {G} codebooks  (~{T/12.5:.2f}s of audio)")

    # ---- drift copy: randomly resample the last K codebooks per frame ----
    g = torch.Generator(device="cpu").manual_seed(SEED)
    drift = codes.clone().cpu()
    vocab_per_group = [3072] + [2048] * (G - 1)
    for j in range(G - DRIFT_LAST_K, G):
        drift[:, j] = torch.randint(0, vocab_per_group[j], (T,), generator=g)
    drift = drift.to(model.device)
    print(f"[ear] perturbed last {DRIFT_LAST_K} codebooks/frame "
          f"({int((drift != codes).sum())} of {T*G} codes changed)")

    # ---- decode both WITH ref prefix, trim ref portion (same as stage4) ----
    def render(gen_codes):
        full = torch.cat([ref_code.to(gen_codes.device), gen_codes], dim=0)
        wavs, sr = tok.decode([{"audio_codes": full}])
        wav = wavs[0]
        wav = wav.detach().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
        wav = wav.reshape(-1)
        cut = int(ref_code.shape[0] / full.shape[0] * wav.shape[0])  # drop ref prefix
        return wav[cut:], sr

    ref_wav, sr = render(codes)
    drift_wav, _ = render(drift)
    sf.write(os.path.join(HERE, "ear_ref.wav"), ref_wav, sr)
    sf.write(os.path.join(HERE, "ear_drift.wav"), drift_wav, sr)

    print(f"\nwrote ear_ref.wav ({len(ref_wav)/sr:.2f}s) and "
          f"ear_drift.wav ({len(drift_wav)/sr:.2f}s)")
    print("Listen to both. ear_ref = clean PyTorch; ear_drift = tail-codebooks")
    print("perturbed (RANDOM, worse than the kernel). If they sound the same, the")
    print("kernel's 13/16 tail drift is audibly safe.")


if __name__ == "__main__":
    main()
