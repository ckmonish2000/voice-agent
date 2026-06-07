# Take-home: gap analysis + execution plan to close everything

**Date:** 2026-06-07
**Purpose:** map the take-home requirements to what's done vs missing, and the
ordered plan to finish. Honest about the hard number (RTF) up front.

---

## Scorecard vs the brief

| Requirement | Target | Status | Evidence / gap |
|---|---|---|---|
| Step 1 — adapt kernel to TTS talker | — | ✅ done + verified | `model_tts.py`, `decode_from_hidden`, vocab flag; parity (4a 0.0078, code0 exact, frame audibly identical) |
| Step 2 — streaming inference server | prompt→stream | ✅ exists; ⚠️ kernel not wired in | `inference_server/app.py` WS `/tts`; runs PyTorch path, not the kernel |
| Step 3 — Pipecat STT→LLM→TTS→audio | — | ✅ built + debugged | `pipecat-qwen/server_app/` working agent |
| Step 4 — e2e validation + numbers | see below | ⚠️ partial | have backbone tok/s + PyTorch RTF; missing TTFC, in-server RTF, e2e latency |
| Stream frame-by-frame (NOT buffered) | hard constraint | ❌ regressed | `qwen_ws_tts.py:62` buffers the WHOLE utterance (a Mac RTF~12 workaround) |
| Demo recording | required | ❌ not started | — |
| README (arch, kernel mods, run) | required | ✅ done | `megakernel/README.md` |
| Build instructions (single 5090) | required | ✅ done | `setup_and_verify.sh` + README |

### Performance targets vs current measurements
| Metric | Brief target (and relaxed deliverable target) | Measured so far |
|---|---|---|
| Decode tok/s (kernel) | report it | **1286 backbone steps/s (0.777 ms/step)** ✅ |
| TTFC | < 60 ms (deliverable: < 90 ms) | **not measured** |
| RTF | < 0.15 (deliverable: < 0.3) | **0.50** (full PyTorch path, 5090) ⚠️ over |
| End-to-end latency | report it | **not measured** |

---

## The crux: the RTF gap, and why the kernel doesn't close it

`benchmark.py` on the 5090:
- Kernel backbone: **0.777 ms/frame** → ~2% of per-frame time.
- Full talker.generate: **40 frames / 1.61 s → RTF 0.50** (frame ≈ 40 ms).
- So ~**98% of per-frame time is the code_predictor (5-layer ×15 per frame) + codec**,
  which the megakernel does **not** accelerate (the brief scopes the kernel to the
  *talker decoder*, not the codebook generator).

**Implication (state this plainly in the writeup):** wiring the kernel into the
server removes the backbone's ~2% — RTF goes ~0.50 → ~0.49. The kernel is correct
and ~103× faster than real-time on its part, but the part it accelerates was never
the bottleneck on a 5090. Hitting RTF < 0.15 requires accelerating the
**code_predictor + codec**, which is out of the kernel's stated scope. The brief
says: *"if you're way off, explain why"* — this is the why, with numbers.

This is a *performance-rigor* win, not a failure: we measured, found the real
bottleneck, and report it honestly instead of hand-waving.

---

## Execution plan (ordered)

### Phase A — Kernel actually serving (Step 2 completion)
1. Integrate the `kernel_in_loop` hook into `StreamingTTSEngine` behind
   `USE_KERNEL` (PyTorch fallback kept). Drop the per-step `cuda.synchronize()`.
   (Detailed in `2026-06-07-inference-integration-and-deploy.md` §2.)
2. Sanity: same line `USE_KERNEL=0` vs `1` → audio sounds identical (expected from
   the ear test); note any latency delta.

### Phase B — Streaming, not buffering (hard constraint)
3. Revert `qwen_ws_tts.py` Bug-6 buffering → **yield each PCM chunk as it arrives**
   (the code already had this; the buffer was a Mac workaround). On the 5090 at
   RTF 0.5 the source outpaces playback, so streaming won't underrun.
4. Confirm frame-by-frame delivery in the diagnostic log (audio_frames increments
   over time, not one big frame at the end).

### Phase C — Honest measurement (Step 4 + deliverable numbers)
5. Extend the bench / add server-side timing to capture:
   - **TTFC** — wall-clock from request to first PCM chunk emitted.
   - **RTF in the live server** (with kernel + streaming), both `USE_KERNEL` modes.
   - **End-to-end latency** — mic stop → first audio out, in the Pipecat round-trip
     (STT + LLM + TTS network included).
6. Tabulate against targets; explain the RTF gap (Phase B/C bottleneck = predictor).

### Phase D — Deploy + demo (deliverable)
7. Run the server on the 5090 (`--host 0.0.0.0`, expose 8000 — see deploy guide),
   point Pipecat's TTS at `ws://<vast-ip>:<port>/tts`.
8. Record the end-to-end demo: speak → transcribe → LLM → streamed TTS → playback.

### Phase E — Writeup (deliverable)
9. Update README with: final perf table (tok/s, TTFC, RTF, e2e), the RTF-gap
   explanation, kernel modifications (already documented), and a bonus section on
   megakernel speedup ideas found during integration.

---

## Bonus (brief asks for it): megakernel speedup ideas found
Capture these in the writeup as candidate optimizations (observed, not all tested):
- The LM-head grid (`LDG_LM_*`) is tuned for vocab 151936; for the talker's 3072 it
  massively over-subscribes blocks — retuning the grid for 3072 rows should cut the
  output-stage cost.
- `decode_from_hidden` lets the kernel skip the embed lookup — feeding the talker's
  summed embedding directly avoids a host-side gather.
- The biggest *system* win isn't the backbone at all (it's already ~free) — it's
  porting the **code_predictor** to the same megakernel approach, since it's 15× per
  frame and dominates RTF. Out of scope here but the clear next lever.

---

## What "done" looks like for submission
- Kernel serving TTS in the server (`USE_KERNEL=1`), streaming frame-by-frame.
- Perf table with real TTFC / RTF / e2e numbers + honest bottleneck analysis.
- Pipecat demo recording.
- README covering arch, kernel mods, how to run, and the speedup ideas.
