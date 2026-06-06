# Megakernel Port — Verified Findings & Index

**Date:** 2026-06-07 · **Status:** Step 0 complete on a real RTX 5090. Box verified,
unchanged kernel builds + runs, **GO/NO-GO cleared (GO)**, talker weight names + shapes
captured. Ready for Phase 4a (kernel body port).

This folder groups everything about porting the `AlpinDale/qwen_megakernel` CUDA kernel
to drive the **Qwen3-TTS talker**. The kernel source itself lives in the separate
`pipecat-qwen` repo under `qwen_megakernel/`; this folder is the plan, the runbook, the
verified results, and the helper scripts.

## Folder layout

```
megakernel/
├── README.md            ← this file: verified results + index + Phase 4a recipe
├── check_cfg.py         ← helper: GO/NO-GO talker config check
├── dump_weights.py      ← helper: dump talker weight names + shapes
└── docs/
    ├── 2026-06-06-megakernel-roadmap.md     ← why the fast path exists (context)
    └── 2026-06-06-megakernel-vast-setup.md  ← step-by-step Vast.ai runbook
```

| File | What it is |
|------|------------|
| `README.md` (this file) | **Verified results** from actually running the runbook, plus exact weight names and the grounded Phase 4a recipe. |
| `docs/2026-06-06-megakernel-roadmap.md` | Why the fast path exists, how it connects to this repo, the change summary. Start here for context. |
| `docs/2026-06-06-megakernel-vast-setup.md` | Step-by-step Vast.ai runbook (rent 5090 → verify → compile → GO/NO-GO). |
| `check_cfg.py` | Helper: registers the `qwen3_tts` arch and prints the talker config (the GO/NO-GO check). |
| `dump_weights.py` | Helper: loads the model and prints the talker's weight key names + shapes (needed for the port). |

> The helper scripts run **on the rented box**, not the Mac (they need the GPU + model).
> Copy their contents over, run, paste output back.

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

### MRoPE (confirmed real)
```
rope_scaling = {"interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default"}
position_id_per_seconds = 13
```
Kernel does plain rotate-half (`kernel.cu:404`); talker uses multimodal RoPE. Plan: try
plain RoPE + θ=1e6 first, compare hidden states, build MRoPE only if parity fails.

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

**Verify 4a:** run the talker as text-style decode through the kernel, dump hidden states
(pre-output-stage), compare to stock PyTorch (`output_hidden_states=True` or hooks on
`talker.model.layers[i]`). Match → body port correct.

---

## 7. Safety net (Phase 3 — "sacred")

The working stock-PyTorch voice agent is committed (this repo + `pipecat-qwen`,
branch `feat/voice-agent`). NEVER edit `.cu` without a committed working state.

## 8. Cost

Box bills ~$0.69/hr while running. **Stop/destroy when idle.** Step-0 cost < $1.
