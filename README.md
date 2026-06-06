# Voice Agent (Qwen3-TTS + Pipecat + React)

A local voice agent: talk into your browser, an AI replies out loud. Same idea as
Vapi or Retell, running entirely on your own machine.

```
Browser (mic + speaker)  →  Pipecat server  →  Inference server (Qwen3-TTS)
     :5173                       :7860                  :8000
                                   │
                                   ├─ Deepgram  (speech → text)
                                   └─ OpenAI    (text → reply)
```

The browser only ever talks to the Pipecat server (:7860). The Pipecat server is
the only thing that knows about Deepgram, OpenAI, and the inference server.

## What's in here

```
voice-agent/
  inference_server/    Loads the Qwen3-TTS model, turns text into audio (port 8000)
    app.py             FastAPI WebSocket server
    engine.py          Streaming TTS engine (model decode loop)
    metrics.py         Latency / RTF reporting
    common.py          Shared Qwen3-TTS model loader (downloads from HuggingFace)
  pipecat_server/      The voice pipeline the browser connects to (port 7860)
    server.py          Process entrypoint (loads .env, starts the runner)
    voice_agent.py     The pipeline: mic → Deepgram → OpenAI → Qwen TTS → speaker
    qwen_ws_tts.py     Pipecat TTS service that calls the inference server
  frontend/            React web app (port 5173)
  docs/                Design doc + debugging log
  requirements.txt     Python dependencies
  .env.example         Copy to .env and add your API keys
```

## Prerequisites

- Python 3.14 (3.11+ should work)
- Node.js 20+ and npm
- A **Deepgram** API key and an **OpenAI** API key (both are paid cloud services)
- ~first run downloads the Qwen3-TTS model from HuggingFace automatically

## One-time setup

### 1. Python environment + dependencies

Use **uv** (recommended — much faster) or plain pip. Pick one.

**Option A — uv (recommended):**
```bash
uv venv --python 3.14                 # create .venv with Python 3.14
uv pip install -r requirements.txt    # install all Python deps
```
If Python 3.14 isn't found, uv offers to download it; or drop `--python 3.14`
to use your default, or pick another like `--python 3.13`.

**Option B — pip + venv:**
```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Secrets

```bash
cp .env.example .env
#    then edit .env and set DEEPGRAM_API_KEY and OPENAI_API_KEY
```

### 3. Frontend dependencies

```bash
cd frontend
npm install
cd ..
```

## Running it (three processes, three terminals)

Start them in this order. Keep each terminal open.

> The Python commands below use `.venv/bin/python`. If you set up with uv, you
> can instead prefix with `uv run` (e.g. `uv run uvicorn ...`, `uv run python
> ...`) and uv will use the venv automatically — no activation needed.

**Terminal 1 — inference server (the voice model), port 8000:**
```bash
.venv/bin/python -m uvicorn inference_server.app:app --port 8000
```
Wait until it prints `[server] ready.` (first run also downloads the model).

**Terminal 2 — Pipecat server (the pipeline), port 7860:**
```bash
.venv/bin/python pipecat_server/server.py -t webrtc
```
If it says a key is missing, your `.env` isn't filled in.

**Terminal 3 — frontend (the web app), port 5173:**
```bash
cd frontend
npm run dev
```
This opens http://localhost:5173 in your browser.

## Using it

1. In the browser, click **Connect** and allow microphone access.
2. Talk. After a short pause you'll hear the reply and see the transcript.

There are two ways to test, both served by the Pipecat server:
- **The React app:** http://localhost:5173 (what `npm run dev` opens)
- **Pipecat's built-in test UI:** http://localhost:7860/client (no frontend needed)

## Important notes

- **Replies are not instant.** On Apple Silicon (MPS) the TTS model runs about
  12x slower than real time, so after you finish speaking there is a pause
  (several seconds, longer for longer replies) while it generates, then the reply
  plays smoothly. This is expected on this hardware — see `docs/` for the full
  explanation and the plan to make it fast with a GPU kernel.
- **Use headphones** if the bot keeps interrupting itself — its own audio leaking
  into the mic can be detected as you starting to talk.
- After restarting the Pipecat server, **close the browser tab and open a fresh
  one** before reconnecting, or you may see "Peer connection not found".

## Tests

```bash
.venv/bin/python -m pytest pipecat_server/ inference_server/
```
(The pipeline assembly test runs without network or API keys.)

## Quick command reference

| What | Command |
|------|---------|
| Setup Python (uv) | `uv venv --python 3.14 && uv pip install -r requirements.txt` |
| Setup Python (pip) | `python -m venv .venv && .venv/bin/pip install -r requirements.txt` |
| Setup secrets | `cp .env.example .env` then edit `.env` |
| Setup frontend | `cd frontend && npm install` |
| Run inference server | `uv run uvicorn inference_server.app:app --port 8000` |
| Run Pipecat server | `uv run python pipecat_server/server.py -t webrtc` |
| Run frontend | `cd frontend && npm run dev` |
| Run tests | `uv run pytest pipecat_server/` |
| Open the app | http://localhost:5173 |

(For pip-based setup, replace `uv run` with `.venv/bin/python -m` / `.venv/bin/python`.)
