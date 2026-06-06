"""
Ear test — does the 13/16 tail-codebook drift actually change the audio?

Our hybrid kernel path matches the first ~12 codebooks of each frame exactly and
drifts on the last few (bf16 accumulation over 28 layers). This script answers the
only question that matters for TTS: can you HEAR that difference?

Method (no kernel-in-the-loop / no KV-cache translation needed):
  1. Generate a real utterance with stock PyTorch -> reference codes (T,16) + wav.
  2. Make a "drifted" copy: for each frame, RANDOMLY resample the LAST K codebooks
     within the codebook's valid range — mimicking the kernel's tail drift, but
     WORSE (random, not the kernel's near-misses) so it's an upper bound on damage.
  3. Decode both code tensors -> two wavs. Listen.

If even this worse-than-kernel perturbation sounds the same, the real 13/16 kernel
output is audibly safe.

Run ON THE BOX:  python ear_test.py "your sentence here"
Outputs: ear_ref.wav (PyTorch), ear_drift.wav (tail-codebooks perturbed)
"""

import os
import sys
import torch

TEXT = sys.argv[1] if len(sys.argv) > 1 else "Hello, this is a test of the speech kernel."
DRIFT_LAST_K = 4     # perturb the last K of 16 codebooks per frame (kernel drifts ~3-4)
SEED = 0

REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    print(f"[ear] loading model; text = {TEXT!r}")
    tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    model = tts.model
    tok = model.speech_tokenizer

    # 1. Real PyTorch utterance -> codes (T,16) via the wrapper's voice-clone path.
    torch.manual_seed(SEED)
    with torch.no_grad():
        result = tts.generate_voice_clone(
            text=TEXT, language="English",
            ref_audio=REF_AUDIO, ref_text=REF_TEXT,
            x_vector_only_mode=False, non_streaming_mode=False,
            do_sample=False, max_new_tokens=512,
        )
    # result shape varies by version; normalize to a (T,16) LongTensor.
    codes = _extract_codes(result).to(model.device)
    print(f"[ear] generated {codes.shape[0]} frames x {codes.shape[1]} codebooks")

    # 2. Drifted copy: randomly resample the last K codebooks per frame.
    g = torch.Generator(device="cpu").manual_seed(SEED)
    drift = codes.clone().cpu()
    T, G = drift.shape
    # codebook 0 vocab 3072; codebooks 1..15 vocab 2048 (per config). Stay in range.
    vocab_per_group = [3072] + [2048] * (G - 1)
    for j in range(G - DRIFT_LAST_K, G):
        rand = torch.randint(0, vocab_per_group[j], (T,), generator=g)
        drift[:, j] = rand
    drift = drift.to(model.device)
    nchanged = (drift != codes).sum().item()
    print(f"[ear] perturbed last {DRIFT_LAST_K} codebooks/frame "
          f"({nchanged} of {T*G} codes changed)")

    # 3. Decode both -> wav
    ref_wav = _decode(tok, codes)
    drift_wav = _decode(tok, drift)

    sf.write(os.path.join(HERE, "ear_ref.wav"), ref_wav, 24000)
    sf.write(os.path.join(HERE, "ear_drift.wav"), drift_wav, 24000)
    print("\nwrote ear_ref.wav (PyTorch) and ear_drift.wav (tail-codebooks perturbed)")
    print("Listen to both. If they sound the same, the kernel's 13/16 tail drift is")
    print("audibly safe (this perturbation is RANDOM = worse than the kernel's near-misses).")


def _extract_codes(result, num_groups=16):
    """Pull a (T,16) LongTensor out of whatever generate_voice_clone returned.

    Handles: tuple/list wrapping, (T,16), (16,T), and FLAT (T*16,) — the version
    on this box returns a flat 1-D tensor (e.g. 65280 = 4080 frames x 16)."""
    import torch
    while isinstance(result, (tuple, list)):
        result = result[0]
    t = result if isinstance(result, torch.Tensor) else torch.as_tensor(result)
    t = t.squeeze()

    if t.dim() == 1:
        n = t.numel()
        if n % num_groups != 0:
            raise SystemExit(
                f"flat codes length {n} not divisible by {num_groups}; inspect output"
            )
        # frames are interleaved as [c0..c15, c0..c15, ...] -> (T, 16)
        t = t.view(-1, num_groups)
    elif t.dim() == 2:
        if t.shape[0] == num_groups and t.shape[1] != num_groups:
            t = t.t()  # (16,T) -> (T,16)
    else:
        raise SystemExit(f"unexpected codes shape {tuple(t.shape)}; inspect output")
    return t.long().contiguous()


def _decode(tok, codes):
    import numpy as np
    import torch
    wavs, _sr = tok.decode([{"audio_codes": codes}])
    wav = wavs[0]
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().to("cpu").numpy()
    return np.asarray(wav).reshape(-1)


if __name__ == "__main__":
    main()
