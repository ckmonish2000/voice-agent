"""
Unit test for the new decode_from_hidden op (Phase 4c, step 1).

Proves: feeding a token's EMBEDDING via decode_from_hidden produces the SAME
post-layer hidden as feeding the token ID via decode. If they match, the new
op (layer 0 reads input_hidden instead of embed[token]) is correct, and we can
build the kernel-in-the-loop on top of it.

PREREQUISITE: recompiled kernel (the .cu now has decode_from_hidden).
Run:  LDG_VOCAB_SIZE=3072 python test_decode_from_hidden.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOK = 100
POS = 0


def main():
    if os.environ.get("LDG_VOCAB_SIZE") != "3072":
        print("WARNING: set LDG_VOCAB_SIZE=3072\n")

    from qwen_tts_megakernel.model_tts import build_talker_decoder
    dec, _ = build_talker_decoder(verbose=True)

    # The embedding row for TOK = exactly what decode() feeds layer 0 internally.
    embed_row = dec._embed_weight[TOK].clone().contiguous()  # (1024,) bf16

    # --- path A: decode() by token id ---
    dec.reset(); dec._position = POS
    _decode = torch.ops.qwen_tts_megakernel_C.decode
    _decode(
        dec._out_token, TOK,
        dec._embed_weight, dec._layer_weights_packed,
        dec._final_norm_weight, dec._lm_head_weight,
        dec._cos_table, dec._sin_table, dec._k_cache, dec._v_cache,
        dec._hidden, dec._act, dec._res, dec._q, dec._k, dec._v,
        dec._attn_out, dec._mlp_inter, dec._norm_out,
        dec._bmax_vals, dec._bmax_idxs,
        28, POS, dec._k_cache.shape[2], dec._attn_scale,
    )
    torch.cuda.synchronize()
    code_a = int(dec._out_token.item())
    hidden_a = dec._hidden.detach().float().clone()

    # --- path B: decode_from_hidden() by embedding vector ---
    dec.reset(); dec._position = POS
    _dfh = torch.ops.qwen_tts_megakernel_C.decode_from_hidden
    _dfh(
        dec._out_token, embed_row,
        dec._embed_weight, dec._layer_weights_packed,
        dec._final_norm_weight, dec._lm_head_weight,
        dec._cos_table, dec._sin_table, dec._k_cache, dec._v_cache,
        dec._hidden, dec._act, dec._res, dec._q, dec._k, dec._v,
        dec._attn_out, dec._mlp_inter, dec._norm_out,
        dec._bmax_vals, dec._bmax_idxs,
        28, POS, dec._k_cache.shape[2], dec._attn_scale,
    )
    torch.cuda.synchronize()
    code_b = int(dec._out_token.item())
    hidden_b = dec._hidden.detach().float().clone()

    diff = (hidden_a - hidden_b).abs().max().item()
    print("\n=== decode vs decode_from_hidden (same input, two entry points) ===")
    print(f"  decode()            code0={code_a}")
    print(f"  decode_from_hidden  code0={code_b}")
    print(f"  max abs diff (hidden) = {diff:.6f}")
    if code_a == code_b and diff < 1e-3:
        print("\n  PASS — decode_from_hidden matches decode. The op is correct.")
    else:
        print("\n  FAIL — paths differ; the new op is not feeding layer 0 correctly.")


if __name__ == "__main__":
    main()
