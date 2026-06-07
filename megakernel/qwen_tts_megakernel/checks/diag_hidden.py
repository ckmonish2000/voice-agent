"""
Diagnostic — which kernel buffer equals PyTorch's talker hidden?

parity_frame16 showed code0 matches but the predictor's input hidden was off by
1.375 (not bf16 noise). So the buffer we read (_norm_out) isn't PyTorch's
past_hidden. This finds the right one by comparing BOTH kernel buffers against
BOTH PyTorch hiddens (pre-norm and post-norm).

Run:  LDG_VOCAB_SIZE=3072 python diag_hidden.py
"""

import os
import torch

TOK = 100
POS = 0


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.environ.get("LDG_VOCAB_SIZE") != "3072":
        print("WARNING: set LDG_VOCAB_SIZE=3072\n")

    from qwen_tts_megakernel.model_tts import build_talker_decoder
    dec, model = build_talker_decoder(verbose=True)
    tm = model.talker.model

    # PyTorch: capture pre-norm (last layer out) AND post-norm (last_hidden_state)
    with torch.no_grad():
        h = tm.codec_embedding.weight[TOK].view(1, 1, -1).to("cuda")
        pos_ids = torch.full((3, 1, 1), POS, dtype=torch.long, device="cuda")
        cos, sin = tm.rotary_emb(h, pos_ids)
        hidden = h
        for layer in tm.layers:
            hidden = layer(hidden_states=hidden, position_ids=pos_ids,
                           position_embeddings=(cos, sin), use_cache=False)[0]
        pt_prenorm = hidden.float().reshape(-1)            # before final norm
        pt_postnorm = tm.norm(hidden).float().reshape(-1)  # = last_hidden_state

    # Kernel
    dec.reset(); dec._position = POS
    _decode = torch.ops.qwen_tts_megakernel_C.decode
    _decode(
        dec._out_token, TOK,
        dec._embed_weight, dec._layer_weights_packed,
        dec._final_norm_weight, dec._lm_head_weight,
        dec._cos_table, dec._sin_table,
        dec._k_cache, dec._v_cache,
        dec._hidden, dec._act, dec._res,
        dec._q, dec._k, dec._v, dec._attn_out,
        dec._mlp_inter, dec._norm_out,
        dec._bmax_vals, dec._bmax_idxs,
        28, POS, dec._k_cache.shape[2], dec._attn_scale,
    )
    torch.cuda.synchronize()

    k_hidden = dec._hidden.detach().float().reshape(-1)    # bf16 buffer -> pre-norm?
    k_norm = dec._norm_out.detach().float().reshape(-1)    # float buffer -> post-norm?
    k_act = dec._act.detach().float().reshape(-1)          # g_activations (pre-norm copy)

    def cmp(name_a, a, name_b, b):
        d = (a - b).abs().max().item()
        print(f"  {name_a:14} vs {name_b:16} max|diff| = {d:.5f}")

    print("\n=== which kernel buffer matches which PyTorch hidden? ===")
    print(f"  pt_prenorm [:4] = {[round(v,4) for v in pt_prenorm[:4].tolist()]}")
    print(f"  pt_postnorm[:4] = {[round(v,4) for v in pt_postnorm[:4].tolist()]}")
    print(f"  k_hidden   [:4] = {[round(v,4) for v in k_hidden[:4].tolist()]}")
    print(f"  k_norm     [:4] = {[round(v,4) for v in k_norm[:4].tolist()]}")
    print(f"  k_act      [:4] = {[round(v,4) for v in k_act[:4].tolist()]}")
    print()
    cmp("k_hidden", k_hidden, "pt_prenorm", pt_prenorm)
    cmp("k_hidden", k_hidden, "pt_postnorm", pt_postnorm)
    cmp("k_norm", k_norm, "pt_prenorm", pt_prenorm)
    cmp("k_norm", k_norm, "pt_postnorm", pt_postnorm)
    cmp("k_act", k_act, "pt_prenorm", pt_prenorm)
    print("\n  -> the pair with the smallest diff tells us which kernel buffer to feed")
    print("     the code_predictor (it wants PyTorch's POST-norm last_hidden_state).")


if __name__ == "__main__":
    main()
