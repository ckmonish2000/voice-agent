"""
parity_kernel_replace.py — verify the kernel-REPLACES-backbone change is correct.

The new path skips PyTorch's 28 backbone layers and runs the kernel instead,
advancing the KV cache length with dummy tokens. We must confirm this produces
the SAME codes as the plain PyTorch backbone (greedy + same seed) — otherwise we
broke generation for speed.

Method: run the same greedy generation twice in ONE process:
  A) PyTorch backbone   (USE_KERNEL effectively off — restore original forward)
  B) kernel-replace path (the new forward)
and compare the generated code-0 sequences frame by frame.

Greedy (do_sample=False) so both are deterministic and directly comparable.

Run on the box:
  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=1 \
  python parity_kernel_replace.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # inference_server/ (parent)

import torch  # noqa: E402
from engine import StreamingTTSEngine, StreamConfig  # noqa: E402

# build with kernel available
os.environ["USE_KERNEL"] = "1"
eng = StreamingTTSEngine()
assert eng._use_kernel, "USE_KERNEL must be 1 for this test"
print(f"[parity] engine ready (kernel available). device={eng.device}", flush=True)

TEXT = "hello there friend"
CFG = StreamConfig(max_new_tokens=24, do_sample=False, seed=0)


def collect_code0(use_kernel):
    """Run greedy generation and collect code-0 of each frame (the backbone's
    output token). Returns a list[int]."""
    eng._use_kernel = use_kernel
    codes0 = []
    q_frames = []

    talker = eng.model.talker
    eos = eng._eos

    def hook(_m, _i, output):
        hs = getattr(output, "hidden_states", None)
        if not (isinstance(hs, tuple) and len(hs) == 2):
            return output
        cid = hs[1]
        if cid is None:
            return output
        f = cid.detach().view(-1)[:16]
        if int(f[0]) == eos:
            return output
        codes0.append(int(f[0]))
        return output

    hh = talker.register_forward_hook(hook)
    if use_kernel:
        eng._install_kernel_backbone_hook()
    torch.manual_seed(0)
    input_ids = eng._text_to_input_ids(TEXT)
    gk = eng._wrapper._merge_generate_kwargs(
        max_new_tokens=CFG.max_new_tokens, do_sample=False)
    with torch.no_grad():
        eng.model.generate(
            input_ids=[input_ids], ref_ids=[eng._voice["ref_tok"]],
            voice_clone_prompt=eng._voice["vc_prompt"], languages=["English"],
            non_streaming_mode=False, **gk)
    hh.remove()
    if use_kernel:
        eng._remove_kernel_backbone_hook()
    return codes0


print("\n[parity] run A: PyTorch backbone (greedy)...", flush=True)
a = collect_code0(use_kernel=False)
print(f"  code0[:12] = {a[:12]}  (n={len(a)})", flush=True)

print("\n[parity] run B: kernel-replace backbone (greedy)...", flush=True)
b = collect_code0(use_kernel=True)
print(f"  code0[:12] = {b[:12]}  (n={len(b)})", flush=True)

print("\n========== PARITY ==========", flush=True)
n = min(len(a), len(b))
match = sum(1 for i in range(n) if a[i] == b[i])
print(f"  compared {n} frames, {match} match ({match/n*100:.0f}%)" if n else
      "  no frames", flush=True)
first_div = next((i for i in range(n) if a[i] != b[i]), None)
if first_div is None:
    print("  IDENTICAL — kernel-replace is correct.", flush=True)
else:
    print(f"  first divergence at frame {first_div}: pytorch={a[first_div]} "
          f"kernel={b[first_div]}", flush=True)
    print("  (a few late divergences can be bf16 rounding; early divergence = bug)",
          flush=True)
print("============================", flush=True)
