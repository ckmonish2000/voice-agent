"""
dev_check.py — confirm which part of the codec is on CPU vs GPU.

The profiler proved the codec does all its work on the CPU (GPU time ~0). The
speech_tokenizer is a wrapper with no .parameters(); the real network is inside
.model / .model.decoder. This prints the device of each real piece so we can
write the precise one-line fix (move the codec to cuda).

Run on the box (server stopped):
  cd /workspace/qwen-tts-0.6b-megakernel/inference_server
  PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
  QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=0 \
  python dev_check.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from engine import StreamingTTSEngine  # noqa: E402

eng = StreamingTTSEngine()
m = eng.model


def dev(mod):
    try:
        for p in mod.parameters():
            return p.device
    except Exception as e:
        return f"(no params: {e})"
    return "no-params"


st = m.speech_tokenizer
print("talker                 device:", dev(m.talker))
print("speech_tokenizer type        :", type(st).__name__)
print("speech_tokenizer.model device:", dev(st.model))
print("  .model.decoder       device:", dev(st.model.decoder))
print("  .model.decoder.pre_transformer:", dev(st.model.decoder.pre_transformer))
print("wrapper .device attr         :", getattr(st, "device", "none"))
print("eng.device                   :", eng.device)
