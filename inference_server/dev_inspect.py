"""
dev_inspect.py — inspect talker.model.forward internals so we can replace the
PyTorch backbone with the kernel safely (without breaking the KV cache / the
code_predictor that runs next).

Run on the box:
  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=0 USE_KERNEL=0 \
  python dev_inspect.py
"""

import sys
import os
import inspect

sys.path.insert(0, os.path.dirname(__file__))

from engine import StreamingTTSEngine  # noqa: E402

eng = StreamingTTSEngine()
tm = eng.model.talker.model
print("talker.model class:", type(tm).__name__)
print("num layers:", len(tm.layers) if hasattr(tm, "layers") else "?")

cp = eng.model.talker.code_predictor
print("code_predictor class:", type(cp).__name__)

print("\n---- talker.model.forward source ----")
try:
    print(inspect.getsource(type(tm).forward))
except Exception as e:
    print(f"(could not get talker.model.forward source: {e!r})")

print("\n---- talker.forward source (how it calls model + code_predictor) ----")
try:
    src = inspect.getsource(type(eng.model.talker).forward)
    print(src)
except Exception as e:
    print(f"(could not get talker.forward source: {e!r})")
