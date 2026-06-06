"""
Runbook Step 5 — dump the TTS talker's weight key names + shapes.

The kernel port (Phase 4a) has to point the megakernel's weight loader at the
talker's tensors. This prints their exact names and shapes so we don't guess on
GPU time. Mirrors common.py: register the custom arch, then load via AutoModel.
"""
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor
from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor,
)

AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

# bfloat16 on the 5090 (this is the kernel's dtype). device_map cuda is fine.
model = AutoModel.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    attn_implementation="eager",
)

sd = model.state_dict()
keys = list(sd.keys())
print("total tensors:", len(keys))

# --- 1. Find the talker backbone prefix + its layer-0 weights (the 11 per-layer) ---
print("\n=== talker backbone layer-0 tensors (expect 11 per layer) ===")
for k in keys:
    kl = k.lower()
    if "talker" in kl and "code_predictor" not in kl and (".layers.0." in kl):
        print(k, tuple(sd[k].shape))

# --- 2. Talker non-layer weights: embed, final norm, codec_head ---
print("\n=== talker embed / norm / codec_head (output projection) ===")
for k in keys:
    kl = k.lower()
    if "talker" in kl and "code_predictor" not in kl and ".layers." not in kl:
        if any(s in kl for s in ["embed", "norm", "codec_head", "lm_head", "head"]):
            print(k, tuple(sd[k].shape))

# --- 3. code_predictor (the separate 5-layer transformer, PyTorch-side for now) ---
print("\n=== code_predictor tensors (layer-0 + heads) ===")
for k in keys:
    kl = k.lower()
    if "code_predictor" in kl and (".layers.0." in kl or ".layers." not in kl):
        print(k, tuple(sd[k].shape))

# --- 4. Sanity: confirm embed != codec_head (untied, per KERNEL_CHANGES #2) ---
print("\n=== untied check (embed vs codec_head) ===")
for k in keys:
    kl = k.lower()
    if "talker" in kl and "code_predictor" not in kl and (
        "embed_tokens" in kl or "codec_head" in kl
    ):
        print(k, tuple(sd[k].shape))
