"""
The 2-minute test: do the talker's MRoPE position_ids collapse to plain 1D
during per-step decode?

If, for every decode step, all 3 MRoPE sections hold the SAME position value,
then MRoPE == plain RoPE for the kernel's job, and the EASY port (theta + weights
+ vocab, no .cu RoPE change) can match the reference. If the 3 sections differ,
MRoPE is genuinely required (HARD port).

How: hook the talker submodule's rotary embedding call and print the position_ids
tensor it actually receives each step. Self-contained (only needs qwen-tts).

Run ON THE BOX:  python check_positions.py
"""

import torch

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
TEXT = "Hello"
REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? "
    "You blew it! And thanks to you."
)
STEPS = 6


def main():
    from qwen_tts import Qwen3TTSModel

    print(f"[positions] loading {MODEL_ID} ...")
    tts = Qwen3TTSModel.from_pretrained(MODEL_ID)
    model = tts.model

    # The talker backbone's rotary embedding module. Its forward(x, position_ids)
    # receives the position_ids of shape (3, batch, seq) -> the 3 MRoPE sections.
    rotary = model.talker.model.rotary_emb

    records = []

    def hook(_m, inputs, _out):
        # inputs = (x, position_ids)
        if len(inputs) >= 2 and isinstance(inputs[1], torch.Tensor):
            pid = inputs[1].detach().to("cpu")
            records.append(pid)
        return None  # don't modify

    h = rotary.register_forward_hook(hook)
    try:
        with torch.no_grad():
            tts.generate_voice_clone(
                text=TEXT,
                language="English",
                ref_audio=REF_AUDIO,
                ref_text=REF_TEXT,
                x_vector_only_mode=False,
                non_streaming_mode=False,
                do_sample=False,
                max_new_tokens=STEPS,
            )
    finally:
        h.remove()

    print(f"\ncaptured {len(records)} rotary calls (prefill + decode steps)\n")
    all_collapsed = True
    for i, pid in enumerate(records):
        shape = tuple(pid.shape)
        # pid shape is (3, batch, seq). Check if the 3 sections are identical.
        if pid.dim() == 3 and pid.shape[0] == 3:
            sec0, sec1, sec2 = pid[0], pid[1], pid[2]
            collapsed = torch.equal(sec0, sec1) and torch.equal(sec1, sec2)
            vals = sec0.reshape(-1).tolist()
            tag = "PLAIN-1D (3 sections identical)" if collapsed else "MRoPE (sections DIFFER)"
            if not collapsed:
                all_collapsed = False
            # show last few positions (the decode tip)
            show = vals[-6:] if len(vals) > 6 else vals
            print(f"call {i}: shape={shape}  {tag}")
            print(f"        section0 last positions: {show}")
            if not collapsed:
                print(f"        section1 last: {sec1.reshape(-1).tolist()[-6:]}")
                print(f"        section2 last: {sec2.reshape(-1).tolist()[-6:]}")
        else:
            print(f"call {i}: shape={shape}  (unexpected shape, raw): {pid.reshape(-1).tolist()[:8]}")

    print("\n" + "=" * 60)
    if all_collapsed:
        print("VERDICT: positions COLLAPSE to plain 1D every call.")
        print("  -> MRoPE == plain RoPE for the kernel. EASY port can match.")
    else:
        print("VERDICT: at least one call has DIFFERING MRoPE sections.")
        print("  -> MRoPE is genuinely active. HARD port (MRoPE) required for parity.")
    print("=" * 60)


if __name__ == "__main__":
    main()
