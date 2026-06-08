# Demo run guide — full voice agent (talk to it, hear kernel-driven speech)

## Architecture (where each piece runs)

```
  YOUR MAC                                 THE BOX (RTX 5090)
  ┌─────────────────────────┐             ┌──────────────────────────┐
  │ browser (mic + speaker)  │  WebRTC     │ inference server :8000    │
  │   http://localhost:7860  │◀──local────▶│   /tts  (kernel TTS)      │
  │ pipecat server :7860     │             │   USE_KERNEL=1            │
  │   STT (Deepgram cloud)   │  WS over    │                          │
  │   LLM (OpenAI cloud)     │  SSH tunnel │                          │
  │   TTS ──────────────────────┼─────────▶│ ws://localhost:8000/tts   │
  └─────────────────────────┘             └──────────────────────────┘
```

- The inference server (the megakernel) runs on the box (needs the GPU).
- Pipecat + the browser run on your Mac (WebRTC mic/speaker must be local).
- Pipecat reaches the inference server through an SSH tunnel.
- STT (Deepgram) and LLM (OpenAI) are cloud calls from your Mac — need API keys.

## Streaming vs buffering (important — two separate layers)

1. INFERENCE SERVER: genuinely streams. Proven: 19 chunks per ~6s utterance,
   arriving ~778 ms apart (not one blob at the end). This is the brief's hard
   constraint and it is met. (debug_tools/bench_engine.py: chunks=19, mean gap 778ms.)
2. BROWSER PLAYBACK: controlled by QWEN_TTS_PREROLL.
   - PREROLL=0  -> true streaming playback: each chunk plays the instant it
     arrives. Honest end-to-end streaming; proves no buffering anywhere. May
     stutter on long replies because at RTF~2.4 playback can outrun generation.
   - PREROLL>0  -> buffer that many seconds before playing (smoother audio, but
     that part is synthesize-then-play, NOT streaming — describe it honestly).

This demo uses PREROLL=0 (true streaming). Keep replies short (system prompt
already asks the LLM for 1–2 sentences) so playback keeps up.

## Reality check on latency (be honest in the demo)

Measured server RTF ≈ 2.4 (slower than real time): generating ~4 s of speech
takes ~10 s. The demo is FUNCTIONAL but not snappy, and with PREROLL=0 a longer
reply may break up — that's the honest consequence of RTF>1, not a bug.

---

## Step 1 — on the BOX: start the inference server (kernel)

```bash
cd /workspace/qwen-tts-0.6b-megakernel/inference_server
PYTHONPATH=/workspace/qwen-tts-0.6b-megakernel/megakernel/qwen_tts_megakernel \
QWEN_DEVICE=cuda LDG_VOCAB_SIZE=3072 USE_KERNEL=1 \
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
Wait for `[server] ready.`  (USE_KERNEL=0 + drop PYTHONPATH for the PyTorch baseline.)

## Step 2 — on your MAC: open the SSH tunnel to the box

```bash
ssh -p <BOX_PORT> root@<BOX_IP> -L 8000:localhost:8000
```
Leave this terminal open. Now ws://localhost:8000/tts on your Mac == the box's
inference server. Sanity check (new Mac terminal):
```bash
curl http://localhost:8000/health      # -> {"status":"ok","sample_rate":24000}
```

## Step 3 — on your MAC: secrets

Create repo-root `.env` (copy from `.env.example` if present):
```
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
```

## Step 4 — on your MAC: start the pipecat voice agent

```bash
cd /Users/monish/Desktop/HoloCron/voice-agent
QWEN_TTS_URI=ws://localhost:8000/tts \
python pipecat_server/server.py -t webrtc
```
(Use the SAME python env that has pipecat installed — NOT `uv run`.)
Starts in V1 buffered mode by default; the UI toggle switches V1<->V2 live, so no
QWEN_TTS_PREROLL needed (it's still honored as the buffered cushion size).

## Step 5 — on your MAC: start the React frontend (has the V1/V2 toggle)

```bash
cd /Users/monish/Desktop/HoloCron/voice-agent/frontend
npm install      # first time only
npm run dev
```
Open the printed URL (usually http://localhost:5173). NOTE: use this, not the
pipecat built-in :7860/client — only the React app has the V1/V2 toggle button.

## Step 6 — talk to it

Connect, then hold **Speak** (or spacebar) and talk. On connect it greets you.
Use the mode button to switch:
  - **V1 Buffered (smooth)** — accumulates the reply, then plays cleanly.
  - **V2 Realtime streaming** — each chunk plays as it arrives (may stutter at
    RTF>1; this is the honest realtime behavior).
The switch applies to the next utterance; the pipecat terminal logs
`[voice_agent] TTS mode -> ...` to confirm.

## Recording the demo
- Screen-record the browser + a terminal showing the server log (so the
  USE_KERNEL=1 + per-utterance metrics are visible).
- Do one short exchange. Mention the honest RTF in voiceover/caption.
- Optionally show one PyTorch-baseline run for contrast.

## Troubleshooting
- `curl 8000/health` fails → tunnel down or server not ready; re-open Step 2.
- Audio breaks up in V2 → expected at RTF>1; switch to V1, or raise the buffered
  cushion via QWEN_TTS_PREROLL.
- Toggle does nothing → make sure you're on the React app (:5173), not :7860/client;
  check the pipecat terminal for `[voice_agent] TTS mode -> ...`.
- Pipecat import errors on Mac → wrong env; use the venv with pipecat-ai==1.3.0.
- "missing DEEPGRAM/OPENAI key" → fill repo-root .env (Step 3).
