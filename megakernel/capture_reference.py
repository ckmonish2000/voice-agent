"""
Phase 4a — capture stock-PyTorch talker REFERENCE hidden states.  (SELF-CONTAINED)

The measuring tool. Before trusting any ported-megakernel output, we capture the
ground truth: the hidden states the *stock PyTorch* talker backbone produces,
step by step, for a fixed greedy input. The ported kernel must reproduce these
(within bf16 tolerance) to pass Phase 4a.

This file imports nothing from the rest of the repo — it uses the official
`qwen_tts` wrapper directly, so you only copy THIS file to the box.

What we capture
---------------
`model.talker` is an nn.Module driven by generate(). Its forward returns
`hidden_states = (layer_hidden_states, codec_ids)`. Per generated step we save:
  - hidden[0]  : backbone hidden state (pre-codec_head)  <-- the 4a parity target
  - codec_ids  : the (1,16) emitted code frame           <-- sanity / 4b later

Greedy (do_sample=False) -> deterministic, directly comparable to the kernel's
argmax decode. A handful of steps is enough for parity.

Run ON THE BOX (needs GPU + model + `pip install qwen-tts`):
    python capture_reference.py
Outputs (next to this file): artifacts/ref_hidden.pt, artifacts/ref_codes.pt
"""

import os
import torch

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
TEXT = "Hello"
LANG = "English"
STEPS = 8

# Official voice-clone reference clip (same one the project's engine uses).
REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)

    # The wrapper registers the qwen3_tts arch and loads model+processor for us.
    from qwen_tts import Qwen3TTSModel

    print(f"[capture] loading {MODEL_ID} ...")
    tts = Qwen3TTSModel.from_pretrained(MODEL_ID)
    model = tts.model
    talker = model.talker

    hidden_steps = []   # (hidden_size,) per decode step
    code_steps = []     # (16,) per decode step

    def _describe(x, depth=0):
        pad = "  " * depth
        if isinstance(x, torch.Tensor):
            return f"{pad}Tensor{tuple(x.shape)} {x.dtype}"
        if isinstance(x, (tuple, list)):
            head = f"{pad}{type(x).__name__}[len={len(x)}]"
            kids = "\n".join(_describe(e, depth + 1) for e in x[:4])
            return head + ("\n" + kids if kids else "")
        return f"{pad}{type(x).__name__}={x!r}"

    printed = {"done": False}

    def _last_tensor(x):
        """Return the final per-layer hidden tensor from a tuple-of-tensors,
        or x itself if it's already a tensor."""
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (tuple, list)) and x:
            return _last_tensor(x[-1])
        return None

    def hook(_module, _inputs, output):
        hs = getattr(output, "hidden_states", None)
        if not (isinstance(hs, tuple) and len(hs) == 2):
            return output
        layer_hidden, codec_ids = hs
        if codec_ids is None:
            return output  # prefill step

        if not printed["done"]:
            print("\n=== talker output.hidden_states structure (first real step) ===")
            print("hidden_states[0] (layer_hidden):")
            print(_describe(layer_hidden, 1))
            print("hidden_states[1] (codec_ids):")
            print(_describe(codec_ids, 1))
            print("=== end structure ===\n")
            printed["done"] = True

        h_t = _last_tensor(layer_hidden)   # final-layer hidden state
        if h_t is None:
            return output
        h = h_t.detach().to("cpu", torch.float32).reshape(-1)
        hidden_steps.append(h)
        code_steps.append(codec_ids.detach().to("cpu").reshape(-1)[:16].clone())
        return output

    handle = talker.register_forward_hook(hook)
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            # greedy + short; only the first STEPS frames are needed for parity
            tts.generate_voice_clone(
                text=TEXT,
                language=LANG,
                ref_audio=REF_AUDIO,
                ref_text=REF_TEXT,
                x_vector_only_mode=False,
                non_streaming_mode=False,
                do_sample=False,
                max_new_tokens=STEPS,
            )
    finally:
        handle.remove()

    n = min(len(hidden_steps), STEPS)
    if n == 0:
        raise SystemExit(
            "No talker steps captured. The hook saw no (hidden, codec_ids) tuple — "
            "check the qwen_tts version / talker.forward return shape."
        )
    hidden_steps = hidden_steps[:n]
    code_steps = code_steps[:n]

    torch.save(hidden_steps, os.path.join(ARTIFACTS, "ref_hidden.pt"))
    torch.save(code_steps, os.path.join(ARTIFACTS, "ref_codes.pt"))

    print(f"\ncaptured {n} steps")
    print(f"  hidden_size = {tuple(hidden_steps[0].shape)} dtype={hidden_steps[0].dtype}")
    for i in range(n):
        h = hidden_steps[i]
        c = code_steps[i].tolist()
        print(f"  step {i}: hidden[:5]={[round(v,4) for v in h[:5].tolist()]}  codes={c}")
    print(f"\nsaved -> {ARTIFACTS}/ref_hidden.pt, ref_codes.pt")


if __name__ == "__main__":
    main()
