"""
Phase 4a — Qwen3-TTS TALKER port of the megakernel weight loader.

This file does NOT modify the original kernel `model.py`. It provides a talker
weight loader + a Decoder built from those weights, reusing the kernel's existing
`Decoder` class (the per-step machinery is identical; only the weights, rope_theta,
vocab, and the untied output head change).

Drop this file next to the cloned kernel repo on the box so it can
`from qwen_megakernel.model import Decoder, _pack_layer_weights, ...`.

Verified facts this port relies on (see megakernel/README.md):
  - talker dims == kernel dims (28 / 1024 / 3072 / 16 / 8 / 128 / eps 1e-6)  [GO]
  - rope_theta = 1_000_000  (was 10_000)
  - MRoPE collapses to plain 1D for this path -> kernel's plain RoPE is correct
  - input embed  = talker.model.codec_embedding (3072, 1024)
  - output head  = talker.codec_head            (3072, 1024)   *** UNTIED ***
  - final norm   = talker.model.norm
  - per-layer prefix = talker.model.layers.{i}.   (same 11 tensors)
"""

import torch

# Talker constants. Core dims match the kernel; only vocab + theta differ.
NUM_LAYERS = 28
HEAD_DIM = 128
MAX_SEQ_LEN = 2048
ROPE_THETA = 1_000_000.0      # talker (was 10_000 for text)
CODEC_VOCAB = 3072            # codebook-0 vocab (was 151936 text vocab)

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def _register_qwen3_tts():
    """Make Auto* understand the qwen3_tts arch (not in stock transformers)."""
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from qwen_tts.core.models import (
        Qwen3TTSConfig,
        Qwen3TTSForConditionalGeneration,
        Qwen3TTSProcessor,
    )
    try:
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
    except Exception:
        pass  # already registered


def load_talker_weights(model_name: str = MODEL_ID, verbose: bool = True):
    """Load the Qwen3-TTS talker backbone weights into GPU tensors, packed the
    way the megakernel expects. Returns (weights_dict, full_model).

    The full model is returned too so a parity test can run stock-PyTorch layers
    on the exact same weights.
    """
    _register_qwen3_tts()
    from transformers import AutoModel

    if verbose:
        print(f"[model_tts] loading {model_name} (bf16, cuda)...")
    model = AutoModel.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda")
    model.eval()
    state = model.state_dict()

    # RoPE tables — plain 1D rotate-half (MRoPE collapses to this for the talker
    # decode path), with the talker's theta = 1e6.
    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

    # Per-layer weights — SAME 11 tensors, talker prefix.
    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"talker.model.layers.{i}."
        layer_weights.extend(
            [
                state[p + "input_layernorm.weight"].contiguous(),
                state[p + "self_attn.q_proj.weight"].contiguous(),
                state[p + "self_attn.k_proj.weight"].contiguous(),
                state[p + "self_attn.v_proj.weight"].contiguous(),
                state[p + "self_attn.q_norm.weight"].contiguous(),
                state[p + "self_attn.k_norm.weight"].contiguous(),
                state[p + "self_attn.o_proj.weight"].contiguous(),
                state[p + "post_attention_layernorm.weight"].contiguous(),
                state[p + "mlp.gate_proj.weight"].contiguous(),
                state[p + "mlp.up_proj.weight"].contiguous(),
                state[p + "mlp.down_proj.weight"].contiguous(),
            ]
        )

    # *** UNTIED *** — input embed and output head are SEPARATE tensors.
    codec_embedding = state["talker.model.codec_embedding.weight"].contiguous()
    codec_head = state["talker.codec_head.weight"].contiguous()

    weights = dict(
        embed_weight=codec_embedding,                      # input: codes -> hidden
        layer_weights=layer_weights,
        final_norm_weight=state["talker.model.norm.weight"].contiguous(),
        lm_head_weight=codec_head,                         # output: hidden -> code0 (UNTIED)
        cos_table=cos_table,
        sin_table=sin_table,
    )
    return weights, model


def build_talker_decoder(verbose: bool = True):
    """Build a kernel Decoder wired to the talker weights, reusing the kernel's
    Decoder machinery unchanged. Returns (decoder, full_model)."""
    # Make `import qwen_megakernel` resolve whether we're run from megakernel/ or
    # from inside the vendored kernel dir. The vendored kernel lives at
    # megakernel/qwen_megakernel/ (a repo dir containing the qwen_megakernel pkg).
    import os, sys
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "qwen_megakernel"), here):
        if os.path.isdir(os.path.join(cand, "qwen_megakernel")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            break

    from qwen_megakernel.model import Decoder

    weights, model = load_talker_weights(verbose=verbose)
    # Decoder(weights=...) skips its own loader and uses ours. tokenizer not
    # needed for hidden-state parity (we feed token ids directly).
    dec = Decoder(weights=weights, tokenizer=None, verbose=verbose)
    return dec, model
