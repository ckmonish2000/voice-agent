"""
diag_codec_profile.py — find WHICH part of the codec decode eats the ~12 seconds.

The codec forward (Qwen3TTSTokenizerV2Decoder.forward) has 5 stages:
  1. quantizer.decode(codes)      -> hidden
  2. pre_conv(hidden)             -> conv
  3. pre_transformer(hidden)      -> attention transformer   <-- flash/sdpa matters
  4. upsample blocks (transpose-conv + convnext)
  5. decoder blocks (conv + snake) -> waveform

We monkey-wrap each stage with a timer (and torch.cuda.synchronize so the timing
is real, not just async-launch time). Whichever stage dominates is the target.

We also print the attention implementation the pre_transformer is actually using
(eager vs sdpa vs flash) — if it's 'eager', switching to 'sdpa' is a likely big
win and needs NO new install.

Run on the box (server stopped):
  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=0 \
  python diag_codec_profile.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402
from engine import StreamingTTSEngine  # noqa: E402

eng = StreamingTTSEngine()
print(f"[diag] engine ready. device={eng.device}", flush=True)

tok = eng.model.speech_tokenizer
# tok is the inference wrapper; tok.model is Qwen3TTSTokenizerV2Model;
# tok.model.decoder is Qwen3TTSTokenizerV2Decoder (the thing with .forward we profile)
inner = tok.model
decoder = inner.decoder
print(f"[diag] decoder class: {type(decoder).__name__}", flush=True)

# --- report the attention implementation actually in use ---
try:
    pt = decoder.pre_transformer
    impl = getattr(pt.config, "_attn_implementation", "?")
    print(f"[diag] pre_transformer attn implementation = {impl!r}", flush=True)
    print(f"[diag] config sliding_window = "
          f"{getattr(pt.config, 'sliding_window', '?')}, "
          f"num_layers = {getattr(pt.config, 'num_hidden_layers', '?')}", flush=True)
except Exception as e:
    print(f"[diag] could not read pre_transformer config: {e!r}", flush=True)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --- wrap each stage of decoder.forward with a timer ---
times = {}


def wrap(obj, name):
    orig = obj.forward

    def timed(*a, **k):
        sync()
        t0 = time.perf_counter()
        out = orig(*a, **k)
        sync()
        times[name] = times.get(name, 0.0) + (time.perf_counter() - t0) * 1000
        return out

    obj.forward = timed
    return orig


# stage 3: the transformer
orig_pt = wrap(decoder.pre_transformer, "3_pre_transformer")
# stage 2: pre_conv
orig_pc = wrap(decoder.pre_conv, "2_pre_conv")
# stage 4: each upsample block group
for i, blocks in enumerate(decoder.upsample):
    for j, b in enumerate(blocks):
        wrap(b, f"4_upsample[{i}][{j}]")
# stage 5: each decoder block
for i, b in enumerate(decoder.decoder):
    wrap(b, f"5_decoder[{i}]")

# build a representative input: ref + 12 frames
ref = eng._voice["ref_code"].to(eng.device)
dummy = torch.zeros((12, 16), dtype=ref.dtype, device=ref.device)
codes = torch.cat([ref, dummy], dim=0)

print("\n[run] warm-up call...", flush=True)
times.clear()
sync(); t0 = time.perf_counter()
tok.decode([{"audio_codes": codes}])
sync(); warm = (time.perf_counter() - t0) * 1000
print(f"[run] warm-up total = {warm:.0f} ms", flush=True)

print("\n[run] measured call (per-stage breakdown):", flush=True)
times.clear()
sync(); t0 = time.perf_counter()
tok.decode([{"audio_codes": codes}])
sync(); total = (time.perf_counter() - t0) * 1000

# aggregate upsample + decoder groups for readability
agg = {}
for k, v in times.items():
    grp = k.split("[")[0]
    agg[grp] = agg.get(grp, 0.0) + v

print(f"\n  TOTAL decode = {total:.0f} ms", flush=True)
for k in sorted(agg):
    print(f"    {k:20s} = {agg[k]:8.0f} ms  ({agg[k]/total*100:4.1f}%)", flush=True)
accounted = sum(agg.values())
print(f"    {'(unaccounted)':20s} = {total-accounted:8.0f} ms  "
      f"({(total-accounted)/total*100:4.1f}%)", flush=True)

print("\n[diag] done. The biggest % is where the time goes -> that's the target.",
      flush=True)
