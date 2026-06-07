"""
Shared helpers for the Qwen3-TTS stage-by-stage walkthrough.

This module centralizes the three things every stage script needs:
  1. Picking the right device (MPS on your M4, CPU fallback).
  2. Loading the model + processor *once* and reusing the same objects.
  3. A small `describe()` helper that prints a tensor the way you'd want to
     inspect it between layers: shape, dtype, device, and a few sample values.

Why one shared loader? Each "stage" of Qwen3-TTS is a sub-module that lives
*inside* one big `Qwen3TTSForConditionalGeneration` object:

    model (Qwen3TTSForConditionalGeneration)
    ├── talker            -> Qwen3TTSTalkerForConditionalGeneration  (Stage 2)
    │   ├── model         -> 28-layer transformer backbone (codebook 0)
    │   └── code_predictor-> 5-layer transformer           (codebooks 1..15) (Stage 3)
    ├── speaker_encoder   -> ECAPA-TDNN x-vector extractor  (voice cloning)
    └── speech_tokenizer  -> codec: 16 codes -> 24kHz waveform (Stage 4)

A real inference server would load this object once and route requests through
those sub-modules. These scripts do exactly that, just slowly and out loud.
"""

import os
import torch

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")


def pick_device() -> str:
    """CUDA (RTX 5090 box) if available, else MPS (Apple GPU), else CPU.
    Override with QWEN_DEVICE=cuda|mps|cpu."""
    forced = os.environ.get("QWEN_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device: str | None = None):
    """
    Load the full Qwen3-TTS model + processor once.

    Uses the low-level path the official wrapper uses (AutoModel after
    registering the custom class). Dtype is chosen by device:
      - CUDA  -> bfloat16 (matches the megakernel; required for USE_KERNEL=1)
      - MPS/CPU -> float32 (the safe choice on Apple Silicon)
    Eager attention everywhere (no flash-attn dependency).

    Returns (model, processor, device).
    """
    import torch
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from qwen_tts.core.models import (
        Qwen3TTSConfig,
        Qwen3TTSForConditionalGeneration,
        Qwen3TTSProcessor,
    )

    device = device or pick_device()
    # bf16 on the GPU (kernel dtype); float32 on MPS/CPU.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # The qwen3_tts architecture is NOT in stock transformers' auto-mapping;
    # the qwen-tts package registers it manually. We do the same here so
    # AutoModel/AutoProcessor know how to build it.
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    print(f"[common] loading {MODEL_ID} on device={device} ({dtype}, eager attn)...")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        attn_implementation="eager",     # no flash-attn dependency
    )
    model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID, fix_mistral_regex=True)
    print("[common] model + processor ready.")
    return model, processor, device


def describe(name: str, x) -> None:
    """Print a tensor/array the way you'd inspect it between layers."""
    if x is None:
        print(f"  {name:<28} = None")
        return
    if isinstance(x, (list, tuple)):
        print(f"  {name:<28} = {type(x).__name__}[len={len(x)}]")
        for i, item in enumerate(x[:3]):
            describe(f"{name}[{i}]", item)
        return
    if isinstance(x, torch.Tensor):
        flat = x.detach().to("cpu").flatten()
        sample = flat[:6].tolist()
        sample = [round(v, 4) if isinstance(v, float) else v for v in sample]
        print(
            f"  {name:<28} shape={tuple(x.shape)} dtype={x.dtype} "
            f"device={x.device} sample={sample}"
        )
        return
    # numpy or scalar
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            print(
                f"  {name:<28} shape={x.shape} dtype={x.dtype} "
                f"sample={np.round(x.flatten()[:6], 4).tolist()}"
            )
            return
    except ImportError:
        pass
    print(f"  {name:<28} = {x!r}")


def banner(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n  {title}\n{line}")
