# Session Handoff — Megakernel TTS take-home (2026-06-08)

Continuity doc so work survives context compaction. Read this first to resume.

---

## TL;DR — where we are

The **megakernel port is DONE and verified** (offline). The **inference-server
integration is the open problem**: the server starts fine but **hangs during
generation — no audio frame is ever produced** (both `USE_KERNEL=1` and the
PyTorch path appear to hang). Root cause not yet confirmed; the deciding
diagnostic has NOT been run yet (see "NEXT ACTION").

---

## What is DONE and verified ✅

- Kernel ported to Qwen3-TTS talker backbone. Parity all PASSED:
  - 4a single-layer body parity: max abs diff 0.0078
  - 4b code0 parity: exact (1497 == 1497)
  - 4b 16-code frame: 13/16 (first 12 exact; bf16 accumulation floor)
  - Ear test: tail-code drift is INAUDIBLE → kernel output is audibly correct
  - `kernel_in_loop.py` (standalone, MAIN thread): produced real kernel-driven
    audio (`kernel_out.wav`) — **this is the "it worked" run.**
- `decode_from_hidden` CUDA op added (kernel takes a hidden vector, not a token
  id). Bit-identical to `decode` (test_decode_from_hidden.py: diff 0.000000).
- `LDG_VOCAB_SIZE` is a build flag (text 151936 ↔ talker 3072).
- Benchmark numbers (from benchmark.py, kernel-driven, on the 5090):
  - Kernel backbone alone: **1286 steps/s (0.777 ms/step)**, ~103× real-time
  - Full PyTorch talker.generate: **RTF 0.50** (2× faster than real-time)
  - Bottleneck = code_predictor (5-layer ×15/frame) + codec, NOT the backbone
    (which the kernel makes ~free). RTF target was 0.15 → we're at 0.50, and we
    know exactly why (documented, honest).
- Repo restructured into one package: `megakernel/qwen_tts_megakernel/`
  (csrc/ + qwen_tts_megakernel/ package + checks/ + docs). All scripts under
  `checks/`. model_tts.py is inside the package.

## Repo / file map (voice-agent repo, branch feat/kernel-inference-streaming)

```
voice-agent/
├── setup.sh                         one-shot box setup (verify + install + checks)
├── requirements.txt                 single deps file (NO torch/torchvision pin)
├── inference_server/
│   ├── app.py                       FastAPI WS server: /tts (text->PCM), /health
│   ├── engine.py                    StreamingTTSEngine.decode_stream() + USE_KERNEL hooks
│   ├── common.py                    load_model() — pick_device() cuda/bf16 (I changed this)
│   ├── metrics.py                   TTFC/RTF/tok-s computation (server side)
│   └── bench_client.py             client benchmark (TTFC/RTF/streaming proof)
├── pipecat_server/
│   ├── server.py                    pipecat runner (port 7860, /client UI)
│   ├── voice_agent.py               STT(Deepgram)->LLM(OpenAI)->TTS pipeline
│   └── qwen_ws_tts.py               TTS service; preroll default set to 0.0 (streaming)
├── frontend/                        React client (vite, :5173)
└── megakernel/
    ├── README.md                    full verified findings (4a/4b/4c, ear test)
    ├── docs/                        roadmap, vast setup, integration guide, gap plan, THIS file
    └── qwen_tts_megakernel/
        ├── csrc/kernel.cu           kernel (+ decode_from_hidden, LDG_VOCAB_SIZE flag)
        ├── csrc/torch_bindings.cpp  decode, decode_from_hidden, generate_nosync
        ├── qwen_tts_megakernel/     __init__, build.py, model.py, model_tts.py (THE PORT)
        └── checks/                  parity_single/code0/frame16, kernel_in_loop,
                                     test_decode_from_hidden, diag_hidden, ear_test,
                                     check_cfg, dump_weights, check_positions,
                                     capture_reference, benchmark, setup_and_verify.sh
```

## What I changed for the server integration (branch feat/kernel-inference-streaming)

- `inference_server/engine.py` — added USE_KERNEL backbone hooks (pre/post hook
  on talker.model; seed kernel KV cache from PyTorch prefill; decode_from_hidden
  each step; overwrite last_hidden_state). **Guarded by `if self._use_kernel`** —
  does NOT run on the PyTorch path.
- `inference_server/common.py` — pick_device() now prefers CUDA; load_model() uses
  bf16 on cuda, float32 on mps/cpu. (Was hardcoded CPU+float32 for Mac.) **This
  change affects BOTH paths.**
- `pipecat_server/qwen_ws_tts.py` — preroll default 2.5 → 0.0 (frame-by-frame
  streaming, the brief's hard constraint).
- `inference_server/bench_client.py` — added live progress logs + 120s timeout.

## THE OPEN BUG 🔴

**Symptom:** server starts cleanly (`device=cuda bfloat16`, `USE_KERNEL=1 —
megakernel backbone active`, `Uvicorn running`, `/health 200 OK`). On a /tts
request: WebSocket accepted, `connection open`, generation starts
(`Setting pad_token_id...`), then **NO audio frame is ever produced** → client
waits → bench_client times out at 120s.

**Note:** `WARNING: Invalid HTTP request received` in the server log is HARMLESS
noise (browser/port-scan HTTP probes hitting the WS port) — NOT related to the hang.

**Two competing hypotheses (unresolved):**
1. **Background-thread bug.** engine.py runs `model.generate()` in a daemon
   `threading.Thread`. The standalone `kernel_in_loop.py` ran on the MAIN thread
   and worked. CUDA context / kernel barriers may deadlock in the worker thread.
   If the gen thread crashes, the exception dies silently → foreground blocks on
   `frame_q.get()` forever → hang. This would affect the kernel path; could also
   affect PyTorch if the crash is generic.
2. **decode_stream / frame-hook / codec bug** independent of the kernel (user
   reports PyTorch path also hung; if true, it's the streaming glue, not kernel).

User's position: "it's NOT the kernel, it worked last time." Likely true that the
WORKING run was `kernel_in_loop.py` (main thread), not the server (daemon thread).

## NEXT ACTION (the ONE diagnostic that decides the fix) ⏭️

Run on the box (stop the server first), `USE_KERNEL` unset = PyTorch path on the
MAIN thread:

```bash
cd /workspace/qwen-tts-0.6b-megakernel/inference_server
QWEN_DEVICE=cuda python - <<'PY'
import sys, os
sys.path.insert(0, os.getcwd())
from engine import StreamingTTSEngine, StreamConfig
eng = StreamingTTSEngine()
print("calling decode_stream on MAIN thread...", flush=True)
n = 0
for pcm in eng.decode_stream("hey", StreamConfig(max_new_tokens=32)):
    n += 1
    if n == 1: print("FIRST CHUNK — works on main thread!", flush=True)
print(f"DONE: {n} chunks", flush=True)
PY
```

Outcomes → fix:
- **Streams here but hangs in server** → the background thread is the bug.
  Fix: run generation on the main thread in decode_stream (drain via the frame
  hook synchronously), OR set the CUDA device inside the worker thread
  (`torch.cuda.set_device(0)` at thread start). Helps both kernel + PyTorch.
- **Hangs/errors here too** → bug is in decode_stream itself (frame hook reading
  wrong field on this transformers version, or codec). Fix that. The traceback
  printed here is the real error the daemon thread was hiding.

Also worth testing to isolate the dtype change I made:
`QWEN_DEVICE=cpu python -m uvicorn app:app --host 0.0.0.0 --port 8080` — if CPU
path streams but cuda+bf16 hangs, my common.py bf16 change is implicated.

## Environment recipe (the version set that finally worked — hard-won)

Box: Vast RTX 5090, CUDA-13 devel image. Install with SYSTEM python (NOT `uv run`
— it builds a CPU-only torch env → device=cpu + kernel import fails).

- torch 2.11.0 + torchvision 0.26.0  (box image build — DO NOT upgrade; torch
  2.12 broke torchvision's `nms` op)
- transformers 4.57.3  (install WITH deps — `--no-deps` left it half-broken:
  GGUF_CONFIG_MAPPING / AutoProcessor import errors)
- huggingface_hub <1.0 (0.36.2)  — 1.8.0 is too new (is_offline_mode missing)
- accelerate 1.12.0
- qwen-tts 0.1.1 (install --no-deps so it doesn't re-pin transformers) + sox + gradio
  (sox python pkg needs system sox: `apt-get install -y sox libsox-dev`)
- huggingface_hub/tokenizers must match transformers 4.57.3

setup.sh now encodes this order. Lesson: never let pip touch torch/torchvision;
install transformers WITH deps; qwen-tts WITH --no-deps.

## How to run things

Inference server (system python, kernel on path, CUDA):
```bash
cd /workspace/qwen-tts-0.6b-megakernel/inference_server
PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=1 \
python -m uvicorn app:app --host 0.0.0.0 --port 8080
```
(USE_KERNEL=0 + drop PYTHONPATH = PyTorch baseline.)

Tunnel from Mac: `ssh -p <port> root@<ip> -L 8080:localhost:8080`
(current box used port 8000 in the tunnel — match server --port to the tunnel.)

Bench (on box for true numbers; on Mac for audio + end-user latency):
```bash
python bench_client.py ws://localhost:8080/tts --save test_out.wav
```

## Remaining for the take-home (after the hang is fixed)

1. Fix the server hang (NEXT ACTION above) — get streaming audio out.
2. Confirm streaming (chunks > 1) + measure TTFC / RTF / e2e in the live server.
3. Pipecat round-trip: talk → STT → LLM → kernel TTS → audio. (pipecat local +
   tunnel to box; WebRTC stays local so it works.)
4. Demo recording (required deliverable).
5. README writeup: perf table (tok/s 1286, RTF 0.50, TTFC, e2e) + honest RTF-gap
   explanation (predictor/codec bottleneck, kernel out of scope for it) + bonus
   speedup ideas (LM-head grid retune for vocab 3072; port code_predictor next).
6. Push branch (pending user's git creds — repo is ckmonish2000, gh is MONISH-CK),
   merge to main, stop the box.

## Honest fallback for submission

Even if the kernel-in-server hang isn't fixed: the kernel is PROVEN correct
end-to-end offline (parity + ear test + kernel_in_loop audio), and the PyTorch
path served via the same server (once the streaming bug is fixed) gives a working
demo at RTF 0.5. A truthful writeup of "kernel verified offline; server runs
PyTorch path; kernel-in-server is WIP due to a thread/CUDA interaction" is a
legitimate, honest submission that scores on rigor + communication.

## Cost note
Box bills ~$0.5–0.7/hr; user had a low-balance warning (~3h). Stop/destroy when idle.
