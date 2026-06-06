# Megakernel Roadmap — the planned fast path

**Date:** 2026-06-06
**Status:** Plan / not yet implemented. The code in this repo runs the slow
(CPU/Mac-GPU) path today. This doc explains the fast path and links to the
detailed Vast.ai runbook.

This repo (`voice-agent/`) contains only the three runnable apps. The CUDA
megakernel itself is a separate, large research effort and its source lives in
the original `pipecat-qwen` repo under `qwen_megakernel/`. This doc is the pointer
+ context so the fast-path plan isn't lost.

---

## Why this exists (the problem you've already felt)

The voice agent works, but on Apple Silicon (MPS) the Qwen3-TTS model generates
audio about **12x slower than real time** (RTF ≈ 11–13). That is the root cause
of every latency and choppiness issue documented in
`2026-06-05-voice-agent-debugging-log.md` (Bug 6) — and the reason
`pipecat_server/qwen_ws_tts.py` has to buffer / pre-roll audio instead of
streaming it cleanly.

No software trick in this repo fixes that. The only real fix is to make the model
**generate faster than real time** (RTF < 1, ideally < 0.1). That is what the
megakernel does.

---

## What the megakernel is (plain)

A GPU runs many small math operations to produce each step of model output.
Normally each operation is a separate "kernel" launched one after another, with
overhead and memory shuffling between every one. A **megakernel** fuses the whole
chain of operations into a **single** GPU launch, so the GPU wastes far less time
starting/stopping and moving data. For this model it turns the slow per-step
audio-token generation into a fast one.

- It runs on **NVIDIA CUDA cores** (specifically an **RTX 5090** — the reference
  kernel is hardcoded to that GPU, `sm_120a`, and needs CUDA ≥ 12.8).
- It does **not** run on a Mac. All work and testing happen on a rented GPU.
- It replaces only the slow part — the talker's autoregressive decode loop. The
  rest of the pipeline (this repo) is unchanged.

Reference implementation we adapt: `AlpinDale/qwen_megakernel` (built for
Qwen3-0.6B text). The lucky break: the TTS **talker** backbone has the *same*
core dimensions as Qwen3-0.6B-text, so the kernel's heavy machinery is reusable.

---

## How it connects to THIS repo (the seam)

The inference server (`inference_server/`) has a clean seam: text in → talker
decode → codec → audio out. Only the **talker decode** step needs the GPU
megakernel. Swapping it in does not change:

- `pipecat_server/` — the voice pipeline (unchanged)
- `frontend/` — the React client (unchanged)
- the inference server's WebSocket protocol (unchanged)

When the megakernel lands (RTF < 1), the buffering/pre-roll in
`pipecat_server/qwen_ws_tts.py` can be reduced to near zero, giving low-latency
**and** smooth streaming at the same time. (The code there has a comment marking
exactly where to switch back to per-chunk streaming.)

---

## What changes in the kernel (summary)

Adapting the text kernel to the TTS talker, in increasing difficulty. Full,
code-grounded detail is in the original repo's `KERNEL_CHANGES.md`.

| # | Change | Difficulty |
|---|--------|-----------|
| 1 | `rope_theta` 10000 → 1000000 (Python constant) | trivial |
| 2 | Load talker weights instead of text weights; untie lm_head | mechanical |
| 6 | Vocab 151936 → 3072 (codebook-0 size) | trivial |
| 3 | Plain RoPE → MRoPE (multimodal position encoding) | medium-hard |
| 5 | Input embedding: 1 lookup → sum of 16 code embeddings | medium |
| 4 | Output: 1 token → code0 from the kernel, then 15 more codes via a separate 5-layer `code_predictor` (kept in PyTorch at first) | hard (the real work) |

Core dims that need **no** change (the reusable body): 28 layers, hidden 1024,
intermediate 3072, 16 Q heads, 8 KV heads, head_dim 128, RMS eps 1e-6.

---

## How we test it (the discipline)

Never edit the CUDA blind. Each increment: change the smallest thing → recompile
on the GPU → **compare numbers (hidden states, then audio codes) against stock
PyTorch** → only then proceed. "Codes match → audio matches."

---

## Where to start

See **`2026-06-06-megakernel-vast-setup.md`** in this same `docs/` folder — the
step-by-step Vast.ai runbook: rent an RTX 5090, verify the box, compile the
unchanged kernel, and clear the GO/NO-GO gate (confirm the talker's dimensions
match the kernel) **before** writing any port code.

Realistic outcome for a first pass: kernel compiles + bench passes + hidden-state
parity proven (the θ + weights + vocab changes), with the output stage partial.
That is a legitimate, documentable milestone.
