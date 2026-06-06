"""
Phase 4b (cont.) — full 16-CODE FRAME parity.

code0 parity proved the kernel's output stage. A talker frame is 16 codes:
  - code0      : kernel  (codec_head argmax)            [proven in parity_code0.py]
  - codes 1-15 : PyTorch code_predictor (5-layer)       [kept in PyTorch by design]

This test builds ONE full frame the hybrid way (kernel code0 + kernel hidden ->
PyTorch predictor for 1..15) and compares it to the all-PyTorch frame.

The seam (from modeling_qwen3_tts.py:1670-1683):
  past_hidden = talker final hidden (POST final-norm = last_hidden_state[:, -1:])
  last_id_hidden = codec_embedding[code0]
  preds = code_predictor.generate(inputs_embeds=cat(past_hidden, last_id_hidden),
                                  max_new_tokens=15, do_sample=False)
  frame = [code0, *preds]

Kernel supplies past_hidden via the `_norm_out` buffer (post-final-norm; the same
tensor the LM head consumed) and code0 via the argmax output. We run the predictor
in PyTorch for both the reference and the hybrid path, so codes 1-15 differ only if
the kernel's hidden/code0 differ from PyTorch's.

PREREQUISITE: kernel compiled with LDG_VOCAB_SIZE=3072.
Run:  LDG_VOCAB_SIZE=3072 python parity_frame16.py
"""

import os
import torch

TOK = 100
POS = 0


def main():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if os.environ.get("LDG_VOCAB_SIZE") != "3072":
        print("WARNING: set LDG_VOCAB_SIZE=3072 (else kernel reads OOB on codec_head).\n")

    from model_tts import build_talker_decoder

    dec, model = build_talker_decoder(verbose=True)
    talker = model.talker
    talker_model = talker.model
    code_predictor = talker.code_predictor
    num_groups = model.config.talker_config.num_code_groups  # 16

    def run_predictor(past_hidden, code0):
        """codes 1..15 from the PyTorch predictor, given the talker hidden + code0.
        Mirrors modeling_qwen3_tts.py:1670-1681."""
        with torch.no_grad():
            code0_t = torch.tensor([[code0]], device="cuda")
            last_id_hidden = talker.get_input_embeddings()(code0_t)  # (1,1,1024)
            inp = torch.cat((past_hidden, last_id_hidden), dim=1)    # (1,2,1024)
            res = code_predictor.generate(
                inputs_embeds=inp,
                max_new_tokens=num_groups - 1,   # 15
                do_sample=False,
                return_dict_in_generate=True,
            )
            return res.sequences.reshape(-1).tolist()  # codes 1..15

    # ---------- All-PyTorch reference frame ----------
    with torch.no_grad():
        h = talker_model.codec_embedding.weight[TOK].view(1, 1, -1).to("cuda")
        pos_ids = torch.full((3, 1, 1), POS, dtype=torch.long, device="cuda")
        cos, sin = talker_model.rotary_emb(h, pos_ids)
        hidden = h
        for layer in talker_model.layers:
            hidden = layer(hidden_states=hidden, position_ids=pos_ids,
                           position_embeddings=(cos, sin), use_cache=False)[0]
        ref_past_hidden = talker_model.norm(hidden)               # (1,1,1024) post-norm
        ref_code0 = int(torch.argmax(talker.codec_head(ref_past_hidden).float().reshape(-1)))
        ref_rest = run_predictor(ref_past_hidden, ref_code0)
    ref_frame = [ref_code0] + ref_rest

    # ---------- Hybrid frame: kernel code0 + kernel hidden -> PyTorch predictor ----------
    dec.reset()
    dec._position = POS
    _decode = torch.ops.qwen_megakernel_C.decode
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
    kernel_code0 = int(dec._out_token.item())
    # _norm_out (g_normalized) = post-final-norm hidden the LM head consumed.
    kernel_hidden = dec._norm_out.detach().to(torch.bfloat16).view(1, 1, -1)
    kernel_rest = run_predictor(kernel_hidden, kernel_code0)
    kernel_frame = [kernel_code0] + kernel_rest

    # ---------- Compare ----------
    print("\n=== 16-code frame parity ===")
    print(f"  token={TOK}  position={POS}")
    print(f"  PyTorch frame: {ref_frame}")
    print(f"  Hybrid  frame: {kernel_frame}")
    matches = sum(int(a == b) for a, b in zip(ref_frame, kernel_frame))
    print(f"  matching codes: {matches}/16")
    # also report kernel-hidden vs ref-hidden distance (diagnostic)
    d = (kernel_hidden.float() - ref_past_hidden.float()).abs().max().item()
    print(f"  max abs diff (kernel hidden vs ref post-norm hidden) = {d:.5f}")
    if kernel_frame == ref_frame:
        print("\n  PASS — full 16-code frame identical. The hybrid kernel+predictor")
        print("  path reproduces the stock-PyTorch talker frame exactly.")
    else:
        print("\n  PARTIAL/FAIL — frames differ. If code0 matches but later codes drift,")
        print("  the predictor amplified a small kernel-hidden diff; inspect max abs diff above.")


if __name__ == "__main__":
    main()
