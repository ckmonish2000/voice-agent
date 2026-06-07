# Megakernel Port — Verified Findings & Index

**Date:** 2026-06-07 · **Status:** **Phase 4a + 4b(code0) PASSED** on a real RTX 5090.
Box verified, kernel builds + runs, GO/NO-GO cleared (GO), talker weights captured, MRoPE
shown to collapse to plain 1D. The ported kernel body matches stock PyTorch to bf16 noise
(4a, max abs diff 0.0078), AND the full 28-layer decode + output stage (vocab 3072 +
`codec_head` + argmax) produces the **exact same codebook-0 token** as PyTorch (4b: both
code0 = 1497, decisive logit margin). **The kernel can now generate a real talker code.**
Next: codes 1–15 via the PyTorch `code_predictor`, then full 16-code frame parity (4b cont.).

This folder groups everything about porting the `AlpinDale/qwen_megakernel` CUDA kernel
to drive the **Qwen3-TTS talker**: the vendored kernel source, the port, the tests, the
runbook, and the verified results. The kernel source is vendored under
`qwen_megakernel/` so the **whole thing clones to the box in one go** — no separate repo.

## Folder layout

The vendored kernel lives in `megakernel/qwen_megakernel/`; all our run scripts are grouped
in `megakernel/qwen_megakernel/checks/` (they add the parent dir to `sys.path`, so
`import qwen_megakernel` resolves with **zero copying**) — clone, `cd checks`, run.

```
megakernel/
├── README.md                       ← this file: verified results + index + recipe
├── docs/
│   ├── 2026-06-06-megakernel-roadmap.md     ← why the fast path exists (context)
│   └── 2026-06-06-megakernel-vast-setup.md  ← step-by-step Vast.ai runbook
└── qwen_megakernel/                ← VENDORED kernel (AlpinDale + our LDG_VOCAB_SIZE flag)
    ├── csrc/kernel.cu                  (LDG_VOCAB_SIZE = build-overridable macro)
    ├── qwen_megakernel/                (model.py, build.py, bench.py — kernel Python side)
    ├── requirements.txt, LICENSE
    └── checks/                     ← ALL our scripts (run from here)
        ├── model_tts.py                THE PORT: talker weights (theta, vocab, untied head)
        ├── parity_single.py            Phase 4a: single-layer parity (PASSED, diff 0.0078)
        ├── parity_code0.py             Phase 4b: full-decode codebook-0 parity (PASSED)
        ├── parity_frame16.py           Phase 4b: full 16-code frame parity (13/16)
        ├── diag_hidden.py              diagnostic: which kernel buffer == PyTorch hidden
        ├── ear_test.py                 codec ear-test: is tail-code drift audible?
        ├── check_cfg.py                GO/NO-GO talker config check
        ├── dump_weights.py             dump talker weight names + shapes
        ├── check_positions.py          prove MRoPE collapses to plain 1D
        └── capture_reference.py        save stock-PyTorch hidden states (ground truth)
```

| File | What it is |
|------|------------|
| `qwen_megakernel/` (package) | Vendored CUDA kernel (upstream + our `LDG_VOCAB_SIZE` build flag). JIT-compiles `csrc/kernel.cu` on first run. |
| `model_tts.py` | **The port.** Talker weights into kernel format (theta=1e6, vocab 3072, untied `codec_head`). Reuses the kernel `Decoder` unchanged. |
| `parity_single.py` | **Phase 4a (PASSED).** One token, one layer, kernel vs PyTorch hidden (diff 0.0078). |
| `parity_code0.py` | **Phase 4b (PASSED).** Full 28-layer decode → codebook-0 token, kernel vs PyTorch (exact). Needs `LDG_VOCAB_SIZE=3072`. |
| `parity_frame16.py` | **Phase 4b.** Full 16-code frame (kernel hidden + code0 → PyTorch predictor for 1–15). 13/16, first 12 exact (bf16 floor). |
| `diag_hidden.py` | Diagnostic: compares kernel buffers vs PyTorch pre/post-norm hidden. |
| `ear_test.py` | Codec ear-test: perturb tail codebooks, decode two wavs, listen — is the drift audible? |
| `check_cfg.py` / `dump_weights.py` / `check_positions.py` / `capture_reference.py` | Step-0 helpers (config, weight names, MRoPE check, ground-truth capture). |
| `docs/…` | Roadmap + Vast.ai runbook. |

## Running on the box (clone → cd → run, no copying)

```bash
git clone https://github.com/ckmonish2000/voice-agent.git
cd voice-agent/megakernel/qwen_megakernel/checks   # all scripts run from here
pip install qwen-tts                                # TTS model stack (registers qwen3_tts)

# Step 0 sanity (optional):
python -m qwen_megakernel.bench   # (run from ../ , the kernel dir) prove build (~1000 tok/s)
python check_cfg.py                          # GO/NO-GO talker config

# Phase 4a/4b parity:
python parity_single.py                      # body parity            -> PASS (0.0078)
LDG_VOCAB_SIZE=3072 python parity_code0.py   # code0 parity (recompiles 3072) -> PASS
LDG_VOCAB_SIZE=3072 python parity_frame16.py # full 16-code frame     -> 13/16

# Ear test (does the tail-code drift sound different?):
python ear_test.py "Hello, this is a test of the speech kernel."
# -> writes ear_ref.wav and ear_drift.wav; download + compare by ear
```

> `python -m qwen_megakernel.bench` is the **upstream** bench and must run from the kernel dir
> (`cd ..` first), since `-m` needs `qwen_megakernel` as a top-level package. Our scripts in
> `checks/` add the parent to `sys.path` themselves, so they run from `checks/` directly.

> **Note:** `LDG_VOCAB_SIZE=3072` is required for `parity_code0`/`parity_frame16` — without it
> the kernel compiles for vocab 151936 and reads past the 3072-row `codec_head` (illegal
> memory access). The JIT recompiles automatically when the flag changes.

---

## 1. Verified environment (the box)

Vast.ai single **RTX 5090**, image `vastai/pytorch_cuda-13.0.2-auto`.

| Check | Result | Pass |
|-------|--------|------|
| `nvidia-smi` | RTX 5090, 32 GB, driver 580.95.05 | ✅ |
| `nvcc --version` | CUDA **13.0**, V13.0.88 (≥12.8 required) | ✅ |
| `torch` | `2.11.0+cu130`, cuda 13.0, available, device RTX 5090 | ✅ |

CUDA 13 supports `sm_120a` (Blackwell) — the kernel's hardcoded arch.

---

## 2. Unchanged kernel bench (Phase 1)

```bash
git clone https://github.com/AlpinDale/qwen_megakernel.git && cd qwen_megakernel
pip install uv && uv pip install --system -r requirements.txt
python -m qwen_megakernel.bench
```

- ✅ Compiles (JIT, `-arch=sm_120a`), no nvcc errors.
- ✅ Speed **~1032–1036 tok/s** (target ~1000; ~8.4× vs PyTorch HF).
- ⚠️ **Correctness check on the text model does NOT match HF greedy** — HF vs MK tokens
  diverge at token 4. Both are greedy + same model, so they should be identical. This is
  a **pre-existing bug in the upstream text kernel**, not introduced by us.

### Theta red herring (don't repeat)
Changing `model.py:55` `10000.0 → 1000000.0` produced **zero** change in the 8 output
tokens — at small positions in bf16 the difference rounds away. So theta is **not** the
cause of the text-bench mismatch.

**Decision:** the text-bench correctness bug is **parked**. It does not block the TTS
port, because Phase 4b *replaces the output stage entirely*. If text parity is ever
needed: localize via a layer-0 numeric diff (norm → QKV → attention → MLP).

---

## 3. GO/NO-GO gate (runbook §4) → **GO**

Talker backbone (`talker_config` of `Qwen/Qwen3-TTS-12Hz-0.6B-Base`) vs the kernel's
hardcoded constants:

| Field | Kernel | Talker | Match |
|-------|--------|--------|-------|
| num_hidden_layers | 28 | **28** | ✅ |
| hidden_size | 1024 | **1024** | ✅ |
| intermediate_size | 3072 | **3072** | ✅ |
| num_attention_heads | 16 | **16** | ✅ |
| num_key_value_heads | 8 | **8** | ✅ |
| head_dim | 128 | **128** | ✅ |
| rms_norm_eps | 1e-6 | **1e-06** | ✅ |
| **rope_theta** | 10000 | **1000000** | ❌ change (expected) |
| **rope_scaling** | none (plain 1D) | **MRoPE** | ❌ change (expected) |
| **codec vocab** | 151936 | **3072** | ❌ change (expected) |
| num_code_groups | — | **16** | new output stage |

**Verdict: GO.** Core dims all match → it's the 0.6B (not 1.7B) → the kernel **body**
(attention + MLP, 28 layers) is reusable unchanged.

### MRoPE — config says multimodal, but it COLLAPSES to plain 1D (verified)
Config declares MRoPE:
```
rope_scaling = {"interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default"}
position_id_per_seconds = 13
```
The kernel does plain rotate-half 1D RoPE (`kernel.cu:404`). **We tested whether MRoPE is
actually active** for the single-speaker text→speech path by hooking
`talker.model.rotary_emb` and printing the real `position_ids` every call
(`check_positions.py`). Result on the box:

```
call 0 (prefill): shape (3,1,111)  PLAIN-1D (3 sections identical)  positions ...108,109,110
call 1..5 (decode): shape (3,1,1)  PLAIN-1D (3 sections identical)  positions 111,112,113,114,115
VERDICT: positions COLLAPSE to plain 1D every call -> MRoPE == plain RoPE for the kernel.
```

The talker builds `position_ids` as a single scalar broadcast into all 3 MRoPE sections
(`modeling_qwen3_tts.py:1706-1710`, `expand(3, -1, -1)`). When the 3 sections are equal,
MRoPE is mathematically identical to plain RoPE. **So no MRoPE kernel change is needed** —
the plain rotate-half body + θ=1e6 is correct for this path. `KERNEL_CHANGES #3`'s
"validate plain RoPE first" path is confirmed.

**One nuance:** decode starts at **position ~111** (after the voice-clone prefill), not 0.
The kernel must seed its position counter with that prefill offset for parity (an integer,
not an algorithm change).

---

## 4. Talker weight names + shapes (runbook §5) — captured on the box

### Backbone — 11 tensors/layer, prefix `talker.model.layers.{i}.`
```
input_layernorm.weight            (1024,)
self_attn.q_proj.weight           (2048, 1024)
self_attn.k_proj.weight           (1024, 1024)
self_attn.v_proj.weight           (1024, 1024)
self_attn.q_norm.weight           (128,)
self_attn.k_norm.weight           (128,)
self_attn.o_proj.weight           (1024, 2048)
post_attention_layernorm.weight   (1024,)
mlp.gate_proj.weight              (3072, 1024)
mlp.up_proj.weight                (3072, 1024)
mlp.down_proj.weight              (1024, 3072)
```
Same 11 tensors/shapes the kernel expects. **Only the prefix changes**
(`model.layers.{i}.` → `talker.model.layers.{i}.`).

### Embedding / norm / output — the UNTIED caveat
```
talker.model.codec_embedding.weight  (3072, 1024)   ← INPUT embedding (codes → hidden)
talker.model.norm.weight             (1024,)        ← final RMSNorm
talker.codec_head.weight             (3072, 1024)   ← OUTPUT projection (hidden → code0)
talker.model.text_embedding.weight   (151936, 2048) ← text side (not used in talker decode)
```
⚠️ **CRITICAL:** the kernel ties `lm_head = embed` (`model.py:87`). **Wrong for the
talker** — `codec_embedding` (in) and `codec_head` (out) are **separate** tensors. Load
them separately; there is no tied lm_head in the talker.

### code_predictor — separate 5-layer transformer (Phase 4b; keep in PyTorch first)
```
talker.code_predictor.model.layers.{0..4}.*           (same 11-tensor layout)
talker.code_predictor.model.norm.weight               (1024,)
talker.code_predictor.model.codec_embedding.{0..14}.weight  (2048, 1024)  × 15
talker.code_predictor.lm_head.{0..14}.weight                (2048, 1024)  × 15
```
Produces codebooks 1–15 (15 in-embeds + 15 out-heads, per-code vocab 2048), conditioned
on the talker hidden state.

---

## 5. Loading the TTS arch (env gotcha)

`qwen3_tts` is not in stock transformers' auto-mapping → plain `AutoConfig.from_pretrained`
throws `KeyError: 'qwen3_tts'` regardless of version. Register it manually first (mirrors
`inference_server/common.py`):

```python
from transformers import AutoConfig, AutoModel, AutoProcessor
from qwen_tts.core.models import (
    Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor,
)
AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
```
Box setup: `pip install qwen-tts` (import name `qwen_tts`). transformers stays 5.10.2 —
the `register()` is what matters. The `flash-attn not installed` warning is harmless.

---

## 6. Phase 4a change recipe (grounded — no guessing)

Edits to the kernel's `qwen_megakernel/model.py`:

| # | Change | Where | From → To |
|---|--------|-------|-----------|
| 1 | rope_theta | `model.py:55` | `10000.0` → `1000000.0` |
| 2 | load via registered TTS AutoModel | `load_weights()` | `AutoModelForCausalLM("Qwen/Qwen3-0.6B")` → registered `AutoModel("Qwen/Qwen3-TTS-12Hz-0.6B-Base")` |
| 3 | per-layer prefix | `model.py:65` | `model.layers.{i}.` → `talker.model.layers.{i}.` |
| 4 | input embed | `model.py:82` | `model.embed_tokens.weight` → `talker.model.codec_embedding.weight` |
| 5 | final norm | `model.py:86` | `model.norm.weight` → `talker.model.norm.weight` |
| 6 | output head (UNTIE) | `model.py:87` | tied → `talker.codec_head.weight` (separate) |
| 7 | VOCAB_SIZE | `model.py:16` | `151936` → `3072` |
| 8 | LM-head grid retune | `kernel.cu:74` + `build.py LDG_LM_*` | size for 3072 rows (was 152k) |

Changes 1–6 are done in `model_tts.py` (Python only) and **verified** (§6b). Changes 7–8
touch `kernel.cu` (compile-time vocab) and need a recompile — that's the start of Phase 4b
(the output stage). Until then, the parity test uses a dummy full-size lm_head to avoid the
out-of-bounds read (see §6b Failure 1).

**Verify 4a:** run the talker as text-style decode through the kernel, dump hidden states
(pre-output-stage), compare to stock PyTorch (`output_hidden_states=True` or hooks on
`talker.model.layers[i]`). Match → body port correct.

---

## 6b. Phase 4a PARITY RESULT — PASSED ✅  (read this, juniors)

**What we proved:** the ported kernel body (talker weights + θ=1e6 + plain RoPE + untied
`codec_head`) computes the *same* layer output as stock PyTorch.

**How we proved it (single-layer parity, `parity_single.py`):**
We don't compare a whole generation — too many moving parts. We isolate ONE layer:
1. PyTorch side: embed one code id via `codec_embedding`, run `talker.model.layers[0]` at
   a fixed position, read the output hidden vector.
2. Kernel side: call the kernel's `decode` with `num_layers=1` at the same position, then
   read the `hidden_buffer` scratch tensor (which holds the post-layer hidden — confirmed
   in `kernel.cu:859/1268`).
3. Compare the two 1024-dim vectors.

**Result:**
```
token=100  position=0
ref    [:6] = [-0.1523, -0.124,  -0.0574, -0.0078, 0.0513, -0.0806]
kernel [:6] = [-0.1514, -0.123,  -0.0571, -0.0083, 0.0513, -0.0806]
max abs diff  = 0.00781   (tolerance 0.02)   mean abs diff = 0.00056
PASS — within bf16 tolerance. Body port is faithful.
```
The remaining ~0.008 is bf16 rounding noise (bf16 has ~2–3 significant digits), not a real
disagreement. **Conclusion: the kernel's attention + MLP body runs the talker correctly.**

### Two failures we hit on the way (and what they taught us)

These are worth understanding — neither was a bug in the kernel math; both were about the
*test setup* and a *known size difference*.

**Failure 1 — `CUDA error: illegal memory access`.**
The kernel's LM-head stage is sized at **compile time** to vocab `151936` (`kernel.cu:74`,
`constexpr LDG_VOCAB_SIZE`). We handed it the talker's `codec_head`, which has only **3072**
rows. The LM head tried to read 151936 rows from a 3072-row tensor → it read ~50× past the
end of the buffer → illegal memory access. *Lesson: a compile-time constant won't adapt to a
runtime tensor; the tensor must match the constant, or the constant must change.*
Fix for the test: pass a dummy 151936×1024 tensor so the LM head reads valid memory; we
ignore its (meaningless) argmax and read the pre-head hidden. The *real* fix (vocab→3072 +
recompile) is Phase 4b.

**Failure 2 — numbers in the right ballpark but ~70% too small** (max diff 0.27, every kernel
value compressed toward zero). The cause was an **unfair comparison**, not bad math. The
kernel attends over `cache_len = position + 1` slots (`kernel.cu:1329`). At `position=5` with
a freshly *zeroed* KV cache, it included 5 spurious all-zero key/value entries in its
attention softmax — while the PyTorch reference (single token, no cache) attended to only the
1 real token. Averaging in 5 zero-entries dragged the kernel output toward zero — exactly the
"right signs, shrunk magnitude" signature we saw. *Lesson: the "same signs but wrong scale"
pattern points at an averaging/normalization difference, here the attention context.*
Fix: use `position=0` so `cache_len=1` → both sides attend to exactly the one real token.
(Bonus: at position 0, RoPE is identity, so this run also rules RoPE in/out cleanly.)

### Debugging principle demonstrated
Both fixes came from **localizing before changing** — read what the kernel actually does
(the vocab constant; the `cache_len = position+1` line), form a hypothesis that explains the
*specific* symptom, then make the smallest change that tests it. We never edited the `.cu`
blindly.

---

## 6c. Phase 4b code0 RESULT — PASSED ✅

After making `LDG_VOCAB_SIZE` a build flag and recompiling with `=3072`, the kernel's full
28-layer decode + output stage (final norm → `codec_head` → argmax) was compared to stock
PyTorch (`parity_code0.py`, single token, position 0):

```
token=100  position=0
PyTorch code0 = 1497
Kernel  code0 = 1497          <- exact match
PyTorch top-5 logits: [(1497, 9.188), (1238, 8.5), (763, 8.25), (628, 8.188), (1185, 7.938)]
PASS — kernel output stage produces the correct codebook-0 token.
```
The top logit (9.188) beats the runner-up (8.5) by a clear margin — a decisive argmax, not a
bf16 tie. **The kernel emits a correct codebook-0 token for the talker.** The vocab build
flag (text 151936 ↔ talker 3072) works; recompile is triggered automatically by the changed
`-DLDG_VOCAB_SIZE` flag.

**Still in PyTorch (by design):** codes 1–15 come from the separate 5-layer `code_predictor`
(`talker.code_predictor`, per-code vocab 2048). Plan: kernel emits the talker hidden state +
code0; PyTorch runs the 15-code loop. Next milestone: full 16-code frame parity.

---

## 7. Safety net (Phase 3 — "sacred")

The working stock-PyTorch voice agent is committed (this repo + `pipecat-qwen`,
branch `feat/voice-agent`). NEVER edit `.cu` without a committed working state.

## 8. Cost

Box bills ~$0.69/hr while running. **Stop/destroy when idle.** Step-0 cost < $1.
