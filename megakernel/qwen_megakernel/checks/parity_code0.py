"""
Phase 4b — code0 PARITY TEST (requires the kernel recompiled with vocab=3072).

Phase 4a proved one layer matches. This proves the FULL talker decode through the
kernel produces the same codebook-0 token as stock PyTorch — i.e. the kernel's
output stage (final norm + codec_head + argmax) is correct for the talker.

Setup (single token, position 0, empty KV cache -> self-only attention, fair):
  - PyTorch: embed code via codec_embedding -> run full talker.model (28 layers)
    -> final norm -> codec_head -> argmax = code0_ref
  - Kernel:  decode(num_layers=28, position=0) with lm_head = codec_head (3072)
    -> the op's argmax output token = code0_kernel
  - Assert code0_ref == code0_kernel, and (diagnostic) compare the top logits.

PREREQUISITE: the kernel MUST be compiled with LDG_VOCAB_SIZE=3072, e.g.:
    LDG_VOCAB_SIZE=3072 python parity_code0.py
(build.py reads LDG_VOCAB_SIZE from the env; the JIT recompiles on flag change.)

Run ON THE BOX from the kernel repo dir, with model_tts.py on the path.
"""

import os
import torch

TOK = 100
POS = 0


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # Safety: warn loudly if the env flag wasn't set (kernel would be vocab 151936
    # and read OOB on the 3072-row codec_head -> illegal memory access).
    vocab_env = os.environ.get("LDG_VOCAB_SIZE")
    if vocab_env != "3072":
        print(
            "WARNING: LDG_VOCAB_SIZE env is not '3072' (got "
            f"{vocab_env!r}). The kernel will compile for vocab 151936 and read "
            "past the 3072-row codec_head -> CUDA illegal memory access.\n"
            "Re-run as:  LDG_VOCAB_SIZE=3072 python parity_code0.py\n"
        )

    from model_tts import build_talker_decoder

    dec, model = build_talker_decoder(verbose=True)
    talker_model = model.talker.model
    codec_head = model.talker.codec_head  # nn.Linear(1024, 3072)

    # ---------- PyTorch reference: full backbone -> codec_head -> argmax ----------
    with torch.no_grad():
        h = talker_model.codec_embedding.weight[TOK].view(1, 1, -1).to("cuda")
        pos_ids = torch.full((3, 1, 1), POS, dtype=torch.long, device="cuda")
        cos, sin = talker_model.rotary_emb(h, pos_ids)

        hidden = h
        for layer in talker_model.layers:
            hidden = layer(
                hidden_states=hidden,
                position_ids=pos_ids,
                position_embeddings=(cos, sin),
                use_cache=False,
            )[0]
        hidden = talker_model.norm(hidden)              # final RMSNorm
        logits = codec_head(hidden).float().reshape(-1)  # (3072,)
        code0_ref = int(torch.argmax(logits).item())
        topv, topi = torch.topk(logits, 5)

    # ---------- Kernel: full decode (28 layers) -> argmax code0 ----------
    dec.reset()
    dec._position = POS
    _decode = torch.ops.qwen_megakernel_C.decode
    _decode(
        dec._out_token, TOK,
        dec._embed_weight, dec._layer_weights_packed,
        dec._final_norm_weight, dec._lm_head_weight,   # lm_head = codec_head (3072)
        dec._cos_table, dec._sin_table,
        dec._k_cache, dec._v_cache,
        dec._hidden, dec._act, dec._res,
        dec._q, dec._k, dec._v, dec._attn_out,
        dec._mlp_inter, dec._norm_out,
        dec._bmax_vals, dec._bmax_idxs,
        28,                       # full backbone
        POS,
        dec._k_cache.shape[2],
        dec._attn_scale,
    )
    torch.cuda.synchronize()
    code0_kernel = int(dec._out_token.item())

    print("\n=== code0 parity (full 28-layer talker decode) ===")
    print(f"  token={TOK}  position={POS}")
    print(f"  PyTorch code0 = {code0_ref}")
    print(f"  Kernel  code0 = {code0_kernel}")
    print(f"  PyTorch top-5 logits: "
          f"{[(int(i), round(float(v),3)) for v, i in zip(topv.tolist(), topi.tolist())]}")
    if code0_ref == code0_kernel:
        print("\n  PASS — kernel output stage produces the correct codebook-0 token.")
    else:
        print("\n  MISMATCH — kernel code0 != PyTorch code0.")
        print("  If they're both plausible and the top-2 logits are near-tied, this can be a")
        print("  bf16 argmax tie-break; otherwise check final norm / codec_head load / vocab.")


if __name__ == "__main__":
    main()
