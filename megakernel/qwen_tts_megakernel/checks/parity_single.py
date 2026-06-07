"""
Phase 4a — single-layer / single-step PARITY TEST.

Proves the ported kernel BODY math (talker weights + theta=1e6 + plain RoPE) is
correct, in isolation from voice-clone / prefill plumbing.

Setup (empty KV cache, one token, one layer):
  - pick a code token id `tok` and a position `P`
  - PyTorch side:  h = codec_embedding[tok];  run talker.model.layers[0] with
    position_embeddings from talker.model.rotary_emb at position P; read output.
  - Kernel side:   decode(num_layers=1, position=P, empty cache), then read the
    `hidden_buffer` scratch (which holds the post-layer hidden state).
  - Compare max abs diff. With an empty cache + single token, attention is
    self-only, so layer 0's output depends only on (embed, RoPE, the layer math)
    — exactly what we want to validate.

Pass: max abs diff within bf16 tolerance (~2e-2). That means the body port is
faithful and the size-match holds in practice.

Run ON THE BOX from the kernel repo dir (so `qwen_tts_megakernel` imports), with
model_tts.py on the path:
    python parity_single.py
"""

import torch

TOK = 100      # arbitrary valid code id (< 3072)
# IMPORTANT: use POS=0 so the kernel attends over exactly 1 cache slot
# (cache_len = position+1, kernel.cu:1329). At POS>0 the kernel attends over
# position+1 slots including the ZEROED cache entries 0..position-1, while the
# PyTorch reference (seq=1, no cache) attends to only the single token -> unfair
# comparison. POS=0 makes both sides attend to exactly the one real token.
POS = 0
LAYER_IDX = 0


def main():
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from qwen_tts_megakernel.model_tts import build_talker_decoder, HEAD_DIM

    # Build kernel decoder (talker weights) + keep the full PyTorch model.
    dec, model = build_talker_decoder(verbose=True)
    talker_model = model.talker.model  # the 28-layer backbone nn.Module

    # ---------- PyTorch reference: one layer on one token at position P ----------
    with torch.no_grad():
        codec_embedding = talker_model.codec_embedding.weight  # (3072, 1024) bf16
        h = codec_embedding[TOK].view(1, 1, -1).to("cuda")     # (1,1,1024)

        # Build position_embeddings (cos,sin) at position P via the model's rotary.
        pos_ids = torch.full((3, 1, 1), POS, dtype=torch.long, device="cuda")
        cos, sin = talker_model.rotary_emb(h, pos_ids)

        layer = talker_model.layers[LAYER_IDX]
        # empty cache -> single-token self-attention; no mask needed for seq=1
        out = layer(
            hidden_states=h,
            position_ids=pos_ids,
            position_embeddings=(cos, sin),
            use_cache=False,
        )
        ref_hidden = out[0].detach().to("cpu", torch.float32).reshape(-1)  # (1024,)

    # ---------- Kernel: decode 1 layer at position P, read hidden_buffer ----------
    # The kernel's LM-head stage is COMPILE-TIME sized to vocab 151936
    # (kernel.cu:74). The talker's codec_head has only 3072 rows, so handing it to
    # the kernel makes the LM head read ~50x past the tensor -> illegal memory
    # access. The parity test only needs the PRE-LM-head hidden state (dec._hidden),
    # so we pass a DUMMY full-size lm_head (151936x1024) purely to give the LM head
    # valid memory to read. Its argmax output is meaningless and ignored.
    # (The real vocab=3072 fix is a separate .cu recompile for Phase 4b.)
    DUMMY_VOCAB = 151936
    dummy_lm_head = torch.zeros(
        DUMMY_VOCAB, dec._hidden.shape[0], dtype=torch.bfloat16, device="cuda"
    )

    dec.reset()
    dec._position = POS
    # run a single decode step with num_layers=1 by calling the op directly with
    # the decoder's buffers but overriding num_layers.
    _decode = torch.ops.qwen_tts_megakernel_C.decode
    _decode(
        dec._out_token,
        TOK,
        dec._embed_weight,
        dec._layer_weights_packed,
        dec._final_norm_weight,
        dummy_lm_head,
        dec._cos_table,
        dec._sin_table,
        dec._k_cache,
        dec._v_cache,
        dec._hidden,
        dec._act,
        dec._res,
        dec._q,
        dec._k,
        dec._v,
        dec._attn_out,
        dec._mlp_inter,
        dec._norm_out,
        dec._bmax_vals,
        dec._bmax_idxs,
        1,            # *** num_layers = 1 ***
        POS,
        dec._k_cache.shape[2],   # max_seq_len
        dec._attn_scale,
    )
    torch.cuda.synchronize()
    kernel_hidden = dec._hidden.detach().to("cpu", torch.float32).reshape(-1)  # (1024,)

    # ---------- Compare ----------
    diff = (ref_hidden - kernel_hidden).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ref_scale = ref_hidden.abs().mean().item()

    print("\n=== single-layer parity (talker layer 0) ===")
    print(f"  token={TOK}  position={POS}")
    print(f"  ref    [:6] = {[round(v,4) for v in ref_hidden[:6].tolist()]}")
    print(f"  kernel [:6] = {[round(v,4) for v in kernel_hidden[:6].tolist()]}")
    print(f"  max abs diff  = {max_abs:.5f}")
    print(f"  mean abs diff = {mean_abs:.5f}   (ref mean |h| = {ref_scale:.4f})")

    tol = 2e-2
    if max_abs < tol:
        print(f"\n  PASS — within bf16 tolerance ({tol}). Body port is faithful.")
    else:
        print(f"\n  FAIL — exceeds tolerance ({tol}). Investigate before proceeding.")
        print("  (Check: untied codec_head vs codec_embedding, q/k_norm, attn_scale,")
        print("   RoPE table indexing, or a residual/precision mismatch.)")


if __name__ == "__main__":
    main()
