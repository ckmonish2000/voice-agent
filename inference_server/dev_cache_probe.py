"""
dev_cache_probe.py — figure out how to advance the PyTorch KV cache length by 1
WITHOUT running the 28 backbone layers, so the kernel can replace the backbone.

Writes findings to inference_server/_cache_probe.txt.

We need to know, for the talker.model's past_key_values during a decode step:
  - the cache class (DynamicCache? layers[]? key_cache[]?)
  - how get_seq_length() is computed (so we know what to bump)
  - whether HF's generate reads anything else between steps that the kernel path
    must keep valid (cache_position is derived from cache length).

Run on the box:
  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=0 \
  python dev_cache_probe.py
Then: cat _cache_probe.txt
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402
from engine import StreamingTTSEngine, StreamConfig  # noqa: E402

eng = StreamingTTSEngine()
talker_model = eng.model.talker.model

lines = []
def w(s=""): lines.append(str(s))

# capture the past_key_values object on the FIRST decode step via a hook
captured = {}

def cap_hook(_m, _a, _kw, output):
    pkv = getattr(output, "past_key_values", None)
    hs = getattr(output, "last_hidden_state", None)
    if hs is not None and hs.shape[1] == 1 and "pkv" not in captured:
        captured["pkv"] = pkv
        captured["hs_shape"] = tuple(hs.shape)
    return output

h = talker_model.register_forward_hook(cap_hook, with_kwargs=True)

# run a couple of decode steps
for i, _ in enumerate(eng.decode_stream("hello there", StreamConfig(max_new_tokens=6))):
    if i > 2:
        break
h.remove()

pkv = captured.get("pkv")
w("=" * 60)
w("PyTorch KV cache (past_key_values) on a decode step")
w("=" * 60)
w(f"last_hidden_state shape on decode step: {captured.get('hs_shape')}")
if pkv is None:
    w("NO past_key_values captured.")
else:
    w(f"cache class            : {type(pkv).__name__}")
    w(f"has .layers            : {hasattr(pkv, 'layers')}")
    w(f"has .key_cache         : {hasattr(pkv, 'key_cache')}")
    try:
        w(f"get_seq_length()       : {pkv.get_seq_length()}")
    except Exception as e:
        w(f"get_seq_length() error : {e!r}")
    # shapes of layer 0
    try:
        if hasattr(pkv, "layers"):
            k = pkv.layers[0].keys
            w(f"layers[0].keys shape   : {tuple(k.shape)} dtype={k.dtype} dev={k.device}")
            w(f"num layers in cache    : {len(pkv.layers)}")
        elif hasattr(pkv, "key_cache"):
            k = pkv.key_cache[0]
            w(f"key_cache[0] shape     : {tuple(k.shape)} dtype={k.dtype} dev={k.device}")
            w(f"num layers in cache    : {len(pkv.key_cache)}")
    except Exception as e:
        w(f"layer shape error      : {e!r}")
    # what methods does it expose for appending?
    methods = [m for m in dir(pkv) if not m.startswith("__") and callable(getattr(pkv, m, None))]
    w(f"cache methods          : {methods}")

# how does get_seq_length read length? show DynamicCache.update / get_seq_length source
import inspect
w("\n" + "=" * 60)
w("cache class get_seq_length / update source")
w("=" * 60)
for meth in ("get_seq_length", "update"):
    try:
        w(f"\n--- {type(pkv).__name__}.{meth} ---")
        w(inspect.getsource(getattr(type(pkv), meth)))
    except Exception as e:
        w(f"(no source for {meth}: {e!r})")

OUT = os.path.join(os.path.dirname(__file__), "_cache_probe.txt")
text = "\n".join(lines)
with open(OUT, "w") as f:
    f.write(text)
print(text)
print(f"\n[wrote] {OUT}\nCopy with: cat _cache_probe.txt")
