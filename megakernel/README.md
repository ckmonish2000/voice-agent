# Megakernel Port — Verified Findings & Index

**Date:** 2026-06-07 · **Status:** **Phases 4a + 4b + 4c COMPLETE + audibly verified** on a
real RTX 5090. The ported megakernel runs the Qwen3-TTS talker backbone end-to-end inside the
real generation loop and produces correct speech:
- **4a** body parity to bf16 noise (max abs diff 0.0078).
- **4b code0** — full 28-layer decode + output stage (vocab 3072 + `codec_head` + argmax)
  produces the **exact** same codebook-0 token as PyTorch (both 1497, decisive margin).
- **4b frame** — full 16-code frame is **13/16** (first 12 exact; last few drift from bf16
  accumulation over 28 layers — the kernel's only approximation).
- **Ear test** — the tail-codebook drift is **inaudible**: randomizing the last 4 codebooks
  per frame (far worse than the kernel's near-misses) is indistinguishable from the clean
  PyTorch reference.
- **4c kernel-in-the-loop** — the kernel runs the 28-layer backbone **every decode step** of
  a full voice-clone utterance (KV cache seeded from PyTorch's prefill, fed via the new
  `decode_from_hidden` op). VERIFY mode confirmed per-step hidden parity over the whole
  utterance; VERIFY=0 substitutes the kernel hidden and renders `kernel_out.wav` — **real,
  kernel-driven, correct-sounding TTS audio.**

**The port is done:** the megakernel correctly drives the talker from a single layer all the
way to end-to-end spoken audio. Remaining work is optional productionization (speed tuning;
wiring into the Pipecat voice agent in `inference_server/`).

This is **`qwen_tts_megakernel`** — the `AlpinDale/qwen_megakernel` CUDA kernel ported to drive
the **Qwen3-TTS talker**. It's one self-contained package (kernel + port + tests + docs) that
clones to the box in one go. The original text-only kernel is not kept separately; its files
live here, edited for TTS (vocab build flag + `decode_from_hidden` op).

## Folder layout

Mirrors the upstream `qwen_megakernel` layout (`csrc/` + inner package), with the TTS port
(`model_tts.py`) inside the package and all run scripts under `checks/`.

```
megakernel/
├── README.md                       ← this file: verified results + usage
├── docs/
│   ├── 2026-06-06-megakernel-roadmap.md     ← why the fast path exists (context)
│   ├── 2026-06-06-megakernel-vast-setup.md  ← step-by-step Vast.ai runbook
│   └── 2026-06-07-inference-integration-and-deploy.md  ← wire kernel into the server + deploy
└── qwen_tts_megakernel/            ← THE package (clone target)
    ├── csrc/
    │   ├── kernel.cu                   fused 28-layer kernel (LDG_VOCAB_SIZE flag + decode_from_hidden)
    │   └── torch_bindings.cpp          torch ops: decode, decode_from_hidden, generate_nosync
    ├── qwen_tts_megakernel/         ← the importable Python package
    │   ├── __init__.py
    │   ├── build.py                    JIT compile flags (-arch=sm_120a, LDG_VOCAB_SIZE)
    │   ├── model.py                    kernel Decoder + KV cache + buffers
    │   └── model_tts.py                THE PORT: load talker weights (theta=1e6, vocab 3072, untied head)
    ├── checks/                      ← all run scripts (run from here)
    │   ├── setup_and_verify.sh         one-shot: deps + box check + parity suite
    │   ├── check_cfg.py                GO/NO-GO talker config
    │   ├── dump_weights.py             talker weight names + shapes
    │   ├── check_positions.py          MRoPE collapses to plain 1D
    │   ├── capture_reference.py        save PyTorch hidden states (ground truth)
    │   ├── parity_single.py            4a: single-layer parity (PASS, 0.0078)
    │   ├── parity_code0.py             4b: full-decode code0 parity (PASS, exact)
    │   ├── parity_frame16.py           4b: full 16-code frame (13/16)
    │   ├── test_decode_from_hidden.py  4c: decode_from_hidden == decode (bit-exact)
    │   ├── kernel_in_loop.py           4c: kernel drives the real loop -> kernel_out.wav
    │   ├── diag_hidden.py              diagnostic: which buffer == PyTorch hidden
    │   └── ear_test.py                 is the tail-code drift audible? (no)
    ├── LICENSE                       (upstream Apache-2.0, preserved)
    └── requirements.txt              torch, transformers, ninja, qwen-tts, soundfile, ...
```

### The files that matter most
| File | What it is |
|------|------------|
| `qwen_tts_megakernel/model_tts.py` | **The port.** Talker weights into kernel format. Start here. |
| `csrc/kernel.cu` | The CUDA kernel + our edits (`LDG_VOCAB_SIZE` flag, `decode_from_hidden`). |
| `checks/kernel_in_loop.py` | **The payoff.** Kernel drives a full utterance → `kernel_out.wav`. |
| `csrc/torch_bindings.cpp` | Exposes `decode` / `decode_from_hidden` to Python. |
| `qwen_tts_megakernel/model.py` / `build.py` | Kernel `Decoder` + JIT build flags. |

## Running on the box

### Prerequisite — the box itself
Rent an **RTX 5090** on Vast.ai with a **CUDA ≥ 12.8 `-devel`** PyTorch image (ships `nvcc`;
the kernel JIT-compiles on first run). Verified-good image: `vastai/pytorch_cuda-13.0.2-auto`.
Disk ≥ 40 GB. See `docs/2026-06-06-megakernel-vast-setup.md` for the full rent runbook.

### Fastest path — one script (recommended)
```bash
git clone https://github.com/ckmonish2000/voice-agent.git
cd voice-agent/megakernel/qwen_tts_megakernel/checks
bash setup_and_verify.sh
```
`setup_and_verify.sh` installs deps, **verifies the box** (RTX 5090 + nvcc ≥ 12.8 + torch sees
CUDA), JIT-builds the kernel via `parity_single`, then runs the output-stage parity suite. It
stops with a clear error if the box is wrong (not a 5090 / no nvcc / CUDA too old).

### Manual path (same steps, run individually)
```bash
git clone https://github.com/ckmonish2000/voice-agent.git
cd voice-agent/megakernel/qwen_tts_megakernel/checks
pip install qwen-tts ninja soundfile           # or: pip install -r ../requirements.txt

# --- verify the box (do this first on any new box) ---
nvidia-smi | grep 5090                          # must show RTX 5090
nvcc --version                                  # must be release >= 12.8
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: ... True NVIDIA GeForce RTX 5090

# --- parity suite (run from checks/; first run JIT-compiles the kernel) ---
python parity_single.py                         # 4a body parity        -> PASS (0.0078)
LDG_VOCAB_SIZE=3072 python parity_code0.py      # 4b code0 parity        -> PASS (recompiles)
LDG_VOCAB_SIZE=3072 python parity_frame16.py    # 4b full 16-code frame  -> 13/16
LDG_VOCAB_SIZE=3072 python test_decode_from_hidden.py   # 4c op check     -> PASS (bit-exact)

# --- end-to-end kernel-driven audio (Phase 4c) ---
LDG_VOCAB_SIZE=3072 VERIFY=1 python kernel_in_loop.py "Hi."   # per-step hidden parity (slow)
LDG_VOCAB_SIZE=3072 VERIFY=0 python kernel_in_loop.py "Hi."   # renders kernel_out.wav

# --- ear test (is the tail-code drift audible? no) ---
python ear_test.py "Hello, this is a test of the speech kernel."
# -> writes ear_ref.wav and ear_drift.wav; download + compare by ear
```

> **`LDG_VOCAB_SIZE=3072` is required** for everything that uses the output stage
> (`parity_code0`, `parity_frame16`, `test_decode_from_hidden`, `kernel_in_loop`) — without it
> the kernel compiles for vocab 151936 and reads past the 3072-row `codec_head` (illegal memory
> access). The JIT recompiles automatically when the flag changes.

> The scripts in `checks/` add the package dir to `sys.path` and import
> `qwen_tts_megakernel.*`, so they run directly from `checks/` — no copying.

> `kernel_in_loop.py` is **slow in VERIFY=1** (runs PyTorch *and* the kernel each step); it
> prints per-step progress so you can see it advance. Use a short input first.

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

## 2. Unchanged kernel bench (Phase 1) — historical

This was the original Phase-1 step against the **upstream** repo (the text-only `bench.py` is
not vendored here, since the TTS port doesn't use it). Recorded for provenance:

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

These are the changes the port makes vs. the original text kernel. In this repo they live in
**`qwen_tts_megakernel/model_tts.py`** (a separate loader that reuses the kernel `Decoder`),
plus the `LDG_VOCAB_SIZE` build flag in `csrc/kernel.cu` — the original `model.py` is kept
unchanged. Table below maps each change to the original `model.py` line for reference:

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
code0; PyTorch runs the 15-code loop.

---

## 6d. Phase 4b frame RESULT — 13/16, and the EAR TEST that settles it

`parity_frame16.py` builds a full frame the hybrid way (kernel hidden + code0 → PyTorch
`code_predictor` for codes 1–15) and compares to the all-PyTorch frame:

```
PyTorch frame: [1497, 1767, 503, 708, 840, 613, 1362, 317, 1459, 551, 1860, 281, 181,  986, 1606, 86]
Hybrid  frame: [1497, 1767, 503, 708, 840, 613, 1362, 317, 1459, 551, 1860, 281, 2042, 986, 62,  348]
               <-------------- first 12 identical -------------->            drift in the tail
matching codes: 13/16
```

The first 12 codebooks match exactly; the last few drift. Root cause (via `diag_hidden.py`):
the kernel's hidden after **28 layers** differs from PyTorch's by ~0.19 (bf16 rounding
*accumulates* across layers — single-layer diff was only 0.0078). That small input diff is
absorbed by the predictor for 12 steps, then flips a few tail codes. This is the kernel's only
approximation — and it's inherent to bf16 (bit-exact 16/16 would need fp32 accumulation,
which defeats the kernel's speed purpose).

### Does the drift matter? Ear test → NO (`ear_test.py`)
Rather than chase bit-exact integers, we tested what actually matters for TTS — **the sound.**
`ear_test.py` generates a real utterance, then makes a copy with the **last 4 codebooks per
frame fully RANDOMIZED** (far worse than the kernel's close-miss drift), decodes both through
the codec (with the voice-clone prefix), and writes two wavs.

**Result: `ear_ref.wav` and `ear_drift.wav` sound the same** — same words, same cloned voice,
no audible artifacts. Since even *random* tail codebooks are inaudible, the kernel's *near-miss*
13/16 drift is **definitively audibly safe**. The later codebooks encode fine residual detail
the codec is largely insensitive to.

**Conclusion:** the megakernel produces **audibly correct** Qwen3-TTS talker output. The body
math, the output stage, and the end-to-end sound are all verified.

---

## 6e. Phase 4c — KERNEL IN THE LOOP (end-to-end audio) — DONE ✅

4a/4b proved the kernel in *isolation* (one token, one frame, fed by PyTorch). 4c puts it in
the **real generation loop**: the kernel runs the 28-layer backbone for **every decode step**
of a full voice-clone utterance.

### The blocker and how it was solved
- The talker feeds each step a **summed code-embedding vector**, not a token id — but the
  kernel's `decode` only took a token id (it did its own embedding lookup). **Fix:** added a
  `decode_from_hidden` op (`kernel.cu` + binding) — an optional `input_hidden` pointer; when
  set, layer 0 reads it directly instead of `embed[token_id]`. Verified bit-identical to
  `decode` (`test_decode_from_hidden.py`: max diff 0.000000).
- The kernel needs the **prefill context** (voice-clone prompt) in its KV cache, but has no
  multi-token prefill path. **Fix:** the kernel KV cache `(L,8,MAX,128)` and PyTorch's
  `DynamicCache` per-layer K/V `(1,8,seq,128)` are layout-compatible — `kernel_in_loop.py`
  **copies PyTorch's post-prefill K/V into the kernel cache** on the first decode step.

### Wiring (`kernel_in_loop.py`)
A pre-hook on `talker.model` captures each step's `inputs_embeds`; a post-hook runs
`decode_from_hidden` at the current position, takes the kernel's pre-norm `_hidden`, applies
PyTorch's `talker.model.norm`, and (VERIFY=0) overwrites `last_hidden_state`. PyTorch still
runs the `code_predictor` (codes 1–15) + codec.

### Result
- **VERIFY=1** — per-step hidden parity over the whole utterance confirmed (matches the
  ~0.2 28-layer bf16 floor; the cache seeding + position bookkeeping are correct).
- **VERIFY=0** — substitutes the kernel hidden and renders `kernel_out.wav`: **real,
  kernel-driven TTS audio that speaks the sentence correctly.**

### Gotchas hit (documented for the next run / juniors)
- transformers 5.x renamed the cache API: use `cache.layers[L].keys/.values`, not the old
  `key_cache`/`value_cache` (added a shim).
- Load the PyTorch model on **CUDA + bf16** (`device_map="cuda"`) — default is CPU → device
  mismatch with the kernel's CUDA tensors.
- The loop is **slow in VERIFY mode** (runs PyTorch *and* the kernel + a per-step
  `cuda.synchronize()`); per-step progress prints show it's advancing. Use a short input first.

---

## 7. Safety net (Phase 3 — "sacred")

The working stock-PyTorch voice agent is committed (this repo + `pipecat-qwen`,
branch `feat/voice-agent`). NEVER edit `.cu` without a committed working state.

## 8. Cost

Box bills ~$0.69/hr while running. **Stop/destroy when idle.** Step-0 cost < $1.
