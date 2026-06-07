# Integrating the megakernel into the inference server + running on Vast.ai

**Date:** 2026-06-07
**Audience:** engineers wiring the verified `qwen_tts_megakernel` into the live TTS
inference server and deploying it on a rented RTX 5090.
**Status:** the kernel is verified correct end-to-end (see `../README.md`). The
engine swap below is **not yet implemented** — this is the execution plan.

---

## 0. Decide first: is it fast enough? (run the benchmark)

Before integrating, get the number:

```bash
cd voice-agent/megakernel/qwen_tts_megakernel/checks
LDG_VOCAB_SIZE=3072 python benchmark.py "The quick brown fox jumps over the lazy dog."
```

It reports:
1. **Kernel backbone alone** — steps/sec, ms/step (the kernel's true speed).
2. **Full PyTorch `talker.generate`** — frames/sec and **RTF** (real-time factor).

Real-time bar: the codec runs at **12.5 frames/sec → 80 ms/frame budget** (RTF < 1).

**Read it:**
- PyTorch **RTF < 1** on the 5090 → already real-time; the kernel adds margin.
- **RTF > 1 and backbone dominates** → kernel integration fixes it.
- **RTF > 1 and the code_predictor/codec dominate** → kernel alone won't be enough;
  the next win is accelerating the 5-layer `code_predictor` (15 passes/frame), which
  the kernel does **not** touch. (Amdahl's law: the kernel only speeds the backbone.)

Don't integrate before you know which case you're in.

---

## 1. The seam — where the kernel plugs in

The server is cleanly layered; the kernel touches exactly one layer:

```
client (WebSocket)
  → inference_server/app.py        @app.websocket("/tts")  — text in, PCM out
    → engine.py  StreamingTTSEngine.decode_stream()        — the streaming seam
      → model.generate()  [talker backbone + code_predictor]   ← KERNEL GOES HERE
        → speech_tokenizer.decode()  [codec → 24 kHz PCM]
```

**Only the talker BACKBONE inside `model.generate()` changes.** Untouched:
`app.py` (WebSocket protocol), the frame queue, the sliding-window codec decode, the
PCM streaming, Pipecat, the frontend. The `decode_stream()` contract is identical.

---

## 2. The integration (engine.py) — step by step

Port the proven hook from `checks/kernel_in_loop.py` into `StreamingTTSEngine`,
behind a flag, with the PyTorch path kept as fallback.

### 2a. Build the kernel decoder once, at engine init
In `StreamingTTSEngine.__init__`, when `USE_KERNEL`:
```python
from qwen_tts_megakernel.model_tts import build_talker_decoder
self._kdec, _ = build_talker_decoder(verbose=False)   # talker weights, packed
```
(Requires the env var `LDG_VOCAB_SIZE=3072` set before import so the kernel
JIT-compiles for the talker's codec vocab.)

### 2b. Install the per-step hook around the generate call
This is exactly `kernel_in_loop.py`'s mechanism, moved inside the engine:
- **pre-hook** on `self.model.talker.model`: capture each step's `inputs_embeds`.
- **post-hook**: on the first decode step, seed the kernel KV cache from PyTorch's
  prefill (`cache.layers[L].keys/.values` → `_k_cache/_v_cache`); each step run
  `decode_from_hidden(inputs_embeds)` at the current position, take the kernel's
  pre-norm `_hidden`, apply `talker.model.norm`, and overwrite `last_hidden_state`.

Reuse these verified details from `kernel_in_loop.py`:
- transformers 5.x cache API: `cache.layers[L].keys/.values` (shim included there).
- **No `cuda.synchronize()` per step** in production (the VERIFY-mode sync was for
  measurement — it stalls the pipeline). Sync only where correctness needs it.

### 2c. Keep the existing frame hook + codec untouched
`decode_stream()` already hooks `talker` for the (1,16) code frames and streams PCM.
The kernel hook is *additional* (on `talker.model`, the backbone), producing the
hidden the code path consumes. The two coexist.

### 2d. Flag + fallback
```python
USE_KERNEL = os.environ.get("USE_KERNEL", "0") == "1"
```
`USE_KERNEL=0` → today's pure-PyTorch path (guaranteed working). `USE_KERNEL=1` →
kernel-driven backbone. Ship with the flag so you can A/B and fall back instantly.

### 2e. Accuracy note (expected, documented)
The kernel backbone is bf16 and accumulates slightly differently over 28 layers
(~0.2 hidden diff → 13/16 frame codes), but the **ear test showed this is
inaudible**. So `USE_KERNEL=1` audio should match `USE_KERNEL=0` audio to the ear.
Verify once after wiring (generate the same line both ways, listen).

---

## 3. Running the inference server on Vast.ai

The server runs the same way regardless of `USE_KERNEL`.

### 3a. Bind to all interfaces (one required change)
`app.py` currently ends with `uvicorn.run(app, host="127.0.0.1", port=8000)` —
`127.0.0.1` is reachable only inside the box. To expose it, run uvicorn with
`--host 0.0.0.0` (don't rely on the `__main__` block):

```bash
cd /workspace/voice-agent
pip install -r requirements.txt
# kernel on (or USE_KERNEL=0 for the PyTorch path):
LDG_VOCAB_SIZE=3072 USE_KERNEL=1 \
  python -m uvicorn inference_server.app:app --host 0.0.0.0 --port 8000
```

### 3b. Expose port 8000 on Vast
Vast maps a container port to a public port. Two ways:
- **At rent time:** in the template's Docker options, add `-p 8000:8000` (same place
  as the `-e ...` env vars). Several stock templates already expose 8000/7860.
- **Check the mapping:** on the instance card, the public **IP** (e.g.
  `142.171.48.138`) plus the ports panel shows `8000 → <external_port>`. The
  reachable address is `http://<ip>:<external_port>`.

> If you can't change the port mapping after renting, re-rent with `-p 8000:8000`,
> OR tunnel out: `ssh -R` / `cloudflared tunnel` / `ngrok http 8000` from the box.

### 3c. Verify it's up
```bash
# from your laptop, using the Vast public IP + mapped port:
curl http://<vast-ip>:<external-port>/health      # -> {"status":"ok",...}
```

---

## 4. Interacting with the backend

The API is a **WebSocket** at `/tts` (plus `GET /health`):
- **send:** a JSON message with the text (see `app.py` / `ws_client_test.py` for the
  exact message shape).
- **receive:** binary PCM frames streaming back, then a `{"event":"done","metrics":…}`
  text message.

Clients:
- **Quick test:** `pipecat-qwen/server_app/ws_client_test.py` (point it at
  `ws://<vast-ip>:<external-port>/tts`).
- **Voice agent:** the Pipecat server's TTS service — change its TTS endpoint from
  `ws://localhost:8000/tts` to the Vast address. Nothing else in Pipecat changes
  (clean seam).

---

## 5. Order of operations (recommended)

1. **Run `benchmark.py`** → get the RTF. Decide if the kernel is worth wiring.
2. **Deploy the PyTorch server on Vast** (`USE_KERNEL=0`, `--host 0.0.0.0`, expose
   8000) → confirm `/health` + a `/tts` round-trip from your laptop. This is the
   guaranteed-working baseline, reachable remotely.
3. **Wire the kernel into `engine.py`** behind `USE_KERNEL` (section 2).
4. **A/B**: same line with `USE_KERNEL=0` vs `1` — confirm audio sounds identical
   (expected) and compare latency.
5. **Point Pipecat** at the Vast `/tts` and dogfood end-to-end.

---

## 6. Gotchas (learned the hard way — see ../README.md §6e)

- `LDG_VOCAB_SIZE=3072` **must** be set or the kernel reads past `codec_head` →
  illegal memory access.
- Load the PyTorch model on **CUDA + bf16** (`device_map="cuda"`).
- transformers 5.x cache API is `cache.layers[L].keys/.values`.
- First kernel run JIT-compiles (~1–2 min).
- Drop the per-step `cuda.synchronize()` in production — it was a VERIFY-mode
  measurement aid and kills throughput.
- `127.0.0.1` vs `0.0.0.0`: the server is unreachable from outside until you bind
  `0.0.0.0` **and** map the port on Vast.
